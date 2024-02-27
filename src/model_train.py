#%% Imports
import os
import datetime

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

from utils.preprocessing.transcript import *
from utils.trainer import Trainer
from utils.dataset import CustomDataset

#%% Training arguments
model_id = 'FacebookAI/roberta-base'
num_labels = 3
max_len = 512

batch_size = 16
learning_rate = 5e-5
num_epochs = 2
weight_decay = 0.01
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW
patience = 5
min_delta = 0.0001

#%% Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
#%% Load data

# Data folder
train_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_train"
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_labels.xlsx"

# Load data
data = load_data_with_labels(labels_path, train_data_path)
data["label"] = data["label"].astype(int)

dataset = CustomDataset(data, max_len=max_len, tokenizer=tokenizer)

# Create a list of indices from 0 to the length of the dataset
indices = list(range(len(dataset)))

# Shuffle the indices
np.random.shuffle(indices)

# Create a train and validation subset of variable dataset with torch
train_size = int(0.89 * len(dataset))
val_size = len(dataset) - train_size

# Split the indices into train and validation sets
train_indices = indices[:train_size]
val_indices = indices[train_size:]

# Use the Subset class for the train and validation subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

# Put train dataset into a loader with 2 batches and put test data in val loader
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

#%% Initialize model and trainer
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)
model.to(device)
trainer = Trainer()
trainer.compile(model, optimizer, learning_rate=learning_rate, loss_fn=loss_fn)

#%% Train model and save
trainer.fit(num_epochs=num_epochs, train_loader=train_loader, val_loader=val_loader, patience=patience, min_delta=min_delta)

# define the name for trained model based on set parameters and date
try:
    os.makedirs('trained_models', exist_ok=True)
    model_name = f"{model_id}_LR{learning_rate}_EPOCHS{num_epochs}_BATCHSIZE_{batch_size}_TIME_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    trainer.save('trained_models/'+model_name)
except:
    model_name = f"trained_model_{model_id}_LR{learning_rate}_EPOCHS{num_epochs}_TIME_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    trainer.save(model_name)

# Access the history
train_loss = trainer.history['train_loss']
val_loss = trainer.history['val_loss']

# plot the loss over epochs
plt.plot(train_loss, label='train_loss')
plt.plot(val_loss, label='val_loss')

# Add labels and title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Loss ov. epochs | {model_id} | LR = {learning_rate} \n {len(dataset)} samples | Batch size = {batch_size}')
plt.legend()

# Ensure the directory exists
os.makedirs('Figures', exist_ok=True)
plt.savefig(f'Figures/loss_plot_{model_name}.png')