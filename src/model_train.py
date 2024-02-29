#%% Imports
import os
from datetime import datetime

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, Precision, Recall
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from utils.preprocessing.transcript import *
from utils.trainer import Trainer
from utils.dataset import CustomDataset

#%% Training arguments
model_id = 'roberta-base'
model_source = "base"
model_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/trained_models/unspecified_checkpoint_EPOCH_2_SAMPLES_5629_BATCHSIZE_16.pt"
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
train_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_train"
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_labels.xlsx"

# Load data
data = load_data_with_labels(labels_path, train_data_path)
data["label"] = data["label"].astype(int) - 1 # Convert labels to 0, 1, 2

dataset = CustomDataset(data, max_len=max_len, tokenizer=tokenizer)

# Create a list of indices from 0 to the length of the dataset
indices = list(range(len(dataset)))

# Shuffle the indices
np.random.seed(42)
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

# Instantiate the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id)
model.to(device) # Send to device

# Instantiate the Trainer
trainer = Trainer(num_labels=num_labels)
trainer.compile(model, optimizer, learning_rate=learning_rate, loss_fn=loss_fn, model_name=model_id)

# Load the saved weights into the model if model_source == fine-tuned
if model_source == "fine-tuned":
    model = trainer.load(model_path, "gpu" if device.type == "cuda" else "cpu")

#%% Train model and save
trainer.fit(num_epochs=num_epochs, train_loader=train_loader, device=device, val_loader=val_loader, patience=patience, min_delta=min_delta)

# define the name for trained model based on set parameters and date
try:
    os.makedirs('trained_models', exist_ok=True)
    model_name = f"{model_id}_LR_{learning_rate}__EPOCHS_{num_epochs}__BATCHSIZE_{batch_size}__TIME_{datetime.now().strftime('%Y-%m-%d_%H%M')}.pt"
    trainer.save('trained_models/'+model_name)
except:
    model_name = f"trained_model_{model_id}__LR_{learning_rate}__EPOCHS_{num_epochs}__TIME_{datetime.now().strftime('%Y-%m-%d_%H%M')}.pt"
    trainer.save(model_name)

#%% Evaluate model
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

# Initialize the metrics
accuracy = Accuracy(task="multiclass", average=None, num_classes=num_labels)
precision = Precision(task='multiclass', average=None, num_classes=num_labels)
recall = Recall(task='multiclass', average=None, num_classes=num_labels)

# Make sure to switch the model to evaluation mode
trainer.model.eval()

# Initialize lists to store the true and predicted labels
true_labels = []
pred_labels = []

print("\nComputing validation metrics...")

with torch.no_grad():
    # Create a progress bar
    progress_bar = tqdm(val_loader, desc="Validation", total=len(val_loader))

    for batch in progress_bar:
        inputs = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['targets'].to(device)

        # Get the model's predictions
        outputs = trainer.model(inputs, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, 1)

        # Update the metrics
        accuracy.update(preds, labels)
        precision.update(preds, labels)
        recall.update(preds, labels)

        # Store the true and predicted labels
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(preds.cpu().numpy())

        # Update the progress bar
        progress_bar.set_postfix({'accuracy': accuracy.compute().tolist(), 'precision': precision.compute().tolist(), 'recall': recall.compute().tolist()})
       
# Compute the confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("\nConfusion matrix:")
print(cm)

# Compute the final metrics
print("\nValidation metrics:")
final_accuracy = accuracy.compute()
final_precision = precision.compute()
final_recall = recall.compute()

print(f"Accuracy: {final_accuracy}")
print(f"Precision: {final_precision}")
print(f"Recall: {final_recall}")

# Print the classification report
print("\nClassification report:")
print(classification_report(true_labels, pred_labels, target_names=classes))

# Write metrics to text file
with open(f'/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/REPORT__{model_name}.txt', 'w') as f:
    f.write('\n\nValidation metrics:\n')
    f.write(f'Accuracy: {final_accuracy}\n')
    f.write(f'Precision: {final_precision}\n')
    f.write(f'Recall: {final_recall}\n')
    f.write('\nConfusion matrix:\n')
    f.write(str(cm))
    f.write('\nClassification report:\n')
    f.write(classification_report(true_labels, pred_labels, target_names=classes))

# Save the confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(f'Outputs/Figures/confusion_matrix_{model_name}.png')