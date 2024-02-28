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
model_id = 'roberta-base'
classes = ["Dismissing", "Secure", "Preoccupied"]
num_labels = len(classes)
max_len = 512

id2label = {i: label for i, label in enumerate(classes)}
label2id = {label: i for i, label in enumerate(classes)}

batch_size = 16
learning_rate = 5e-5
num_epochs = 5
weight_decay = 0.01
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW
patience = 2
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
data["label"] = data["label"].astype(int) - 1 # Convert labels to 0, 1, 2

dataset = CustomDataset(data, max_len=max_len, tokenizer=tokenizer)

# Create a list of indices from 0 to the length of the dataset
indices = list(range(len(dataset)))

# Shuffle the indices
np.random.shuffle(indices, random_state=42)

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
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id)
model.to(device)
trainer = Trainer()
trainer.compile(model, optimizer, learning_rate=learning_rate, loss_fn=loss_fn, model_name=model_id)

#%% Train model and save
trainer.fit(num_epochs=num_epochs, train_loader=train_loader, device=device, val_loader=val_loader, patience=patience, min_delta=min_delta)

# define the name for trained model based on set parameters and date
try:
    os.makedirs('trained_models', exist_ok=True)
    model_name = f"{model_id}_LR_{learning_rate}__EPOCHS_{num_epochs}__BATCHSIZE_{batch_size}__TIME_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    trainer.save('trained_models/'+model_name)
except:
    model_name = f"trained_model_{model_id}__LR_{learning_rate}__EPOCHS_{num_epochs}__TIME_{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
    trainer.save(model_name)

# Access the history
# Access the history
train_loss = trainer.history['train_loss']
val_loss = trainer.history['val_loss']
train_accuracy = trainer.history['train_accuracy']
val_accuracy = trainer.history['val_accuracy']
train_precision = trainer.history['train_precision']
val_precision = trainer.history['val_precision']
train_recall = trainer.history['train_recall']
val_recall = trainer.history['val_recall']

# Create a figure with 5 subplots
fig, axs = plt.subplots(5, sharex=True, figsize=(10, 15))

# Plot the loss
axs[0].plot(train_loss, label='Train Loss')
axs[0].plot(val_loss, label='Validation Loss')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot the accuracy
axs[1].plot(train_accuracy, label='Train Accuracy')
axs[1].plot(val_accuracy, label='Validation Accuracy')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

# Plot the precision
axs[2].plot(train_precision, label='Train Precision')
axs[2].plot(val_precision, label='Validation Precision')
axs[2].set_ylabel('Precision')
axs[2].legend()

# Plot the recall
axs[3].plot(train_recall, label='Train Recall')
axs[3].plot(val_recall, label='Validation Recall')
axs[3].set_ylabel('Recall')
axs[3].legend()

# Add labels and title
axs[4].set_xlabel('Epochs')
fig.suptitle(f'Metrics over epochs | {trainer.model_name} | LR = {learning_rate} \n {len(dataset)} samples | Batch size = {batch_size}')

# Ensure the directory exists
os.makedirs('Outputs/Figures', exist_ok=True)
plt.savefig(f'Outputs/Figures/metrics_plot_{trainer.model_name}.png')

# Plot the loss over epochs separately
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Loss over epochs | {trainer.model_name} | LR = {learning_rate} \n {len(dataset)} samples | Batch size = {batch_size}')
plt.legend()
plt.savefig(f'Outputs/Figures/loss_plot_{trainer.model_name}.png')

# Write metrics to a text file
with open(f'/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/REPORT__{model_name}.txt', 'w') as file:
    file.write(f'Train Loss: {train_loss[-1]}\n')
    file.write(f'Validation Loss: {val_loss[-1]}\n')
    file.write(f'Train Accuracy: {train_accuracy[-1]}\n')
    file.write(f'Validation Accuracy: {val_accuracy[-1]}\n')
    file.write(f'Train Precision: {train_precision[-1]}\n')
    file.write(f'Validation Precision: {val_precision[-1]}\n')
    file.write(f'Train Recall: {train_recall[-1]}\n')
    file.write(f'Validation Recall: {val_recall[-1]}\n')