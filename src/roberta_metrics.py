import os
import datetime

from torchmetrics import Accuracy, Precision, Recall, ConfusionMatrix
import torch
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

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
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id)
model.to(device)

# Data folders
val_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_val"
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_labels.xlsx"

# Load data
val_data = load_data_with_labels(labels_path, val_data_path)
val_data["label"] = val_data["label"].astype(int) - 1 # Convert labels to 0, 1, 2

max_len = 512
val_dataset = CustomDataset(val_data, max_len=max_len, tokenizer=tokenizer)

# Put datasets into loaders
batch_size = 16
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Instantiate the Trainer
trainer = Trainer()
# trainer.compile(model, torch.optim.AdamW, learning_rate=5e-5, loss_fn=torch.nn.CrossEntropyLoss())
trainer.model = model
trainer.val_loader = val_loader

# Load the saved weights into the model
model_state_dict = torch.load(model_path)
model.load_state_dict(model_state_dict)

# Initialize the metrics
accuracy = Accuracy(task="multiclass", average=None, num_classes=num_labels)
precision = Precision(task='multiclass', average=None, num_classes=num_labels)
recall = Recall(task='multiclass', average=None, num_classes=num_labels)

# Make sure to switch the model to evaluation mode
trainer.model.eval()

# Initialize lists to store the true and predicted labels
true_labels = []
pred_labels = []

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
print("\nFinal metrics:")
final_accuracy = accuracy.compute()
final_precision = precision.compute()
final_recall = recall.compute()

print(f"Accuracy: {final_accuracy}")
print(f"Precision: {final_precision}")
print(f"Recall: {final_recall}")

# Print the classification report
print("\nClassification report:")
print(classification_report(true_labels, pred_labels, target_names=classes))

# Create output folder
model_name = model_path.split("/")[-1].split(".")[0]
output_folder = f"Outputs/{model_name}"

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
plt.savefig(f'Outputs/{model_name}/confusion_matrix_{model_name}.png')