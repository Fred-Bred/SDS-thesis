import os
import datetime

from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizerForSequenceClassification
import numpy as np

from utils.trainer import Trainer
from utils.preprocessing.transcript import load_data_with_labels
from utils.dataset import CustomDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "roberta-base"
model_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/trained_models/unspecified_checkpoint_EPOCH_2_SAMPLES_5629_BATCHSIZE_16.pt"

classes = ["Dismissing", "Secure", "Preoccupied"]
num_labels = len(classes)

id2label = {i: label for i, label in enumerate(classes)}
label2id = {label: i for i, label in enumerate(classes)}

# Instantiate the tokenizer and the model
tokenizer = AutoTokenizerForSequenceClassification.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id)
model.to(device)

# Instantiate the Trainer
trainer = Trainer()

# Data folder
train_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_train"
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_labels.xlsx"

# Load data
data = load_data_with_labels(labels_path, train_data_path)
data["label"] = data["label"].astype(int) - 1 # Convert labels to 0, 1, 2

max_len = 512
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
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Load the saved weights into the model
trainer.load(model_path)

# Initialize the metrics
accuracy = Accuracy()
precision = Precision(average='weighted')
recall = Recall(average='weighted')

# Make sure to switch the model to evaluation mode
trainer.model.eval()

# Disable gradient calculations
with torch.no_grad():
    for inputs, labels in val_loader:
        # Move inputs and labels to the right device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Get the model's predictions
        outputs = trainer.model(inputs)
        _, preds = torch.max(outputs, 1)

        # Update the metrics
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)

# Compute the final metrics
final_accuracy = accuracy.compute()
final_precision = precision.compute()
final_recall = recall.compute()

print(f"Accuracy: {final_accuracy}")
print(f"Precision: {final_precision}")
print(f"Recall: {final_recall}")