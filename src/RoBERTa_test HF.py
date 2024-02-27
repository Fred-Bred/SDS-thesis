#%%
# Imports
import os
import datetime

from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import evaluate
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.preprocessing.transcript import *
from utils.model import RoBERTaTorch
import utils.trainer
#%%
# Training arguments
training_args = TrainingArguments(
    output_dir="trained_models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01, # Check implementation against BatchNorm
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sequence length
max_length = 512

# Data folder
train_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_train"
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_labels.xlsx"

# Base model name
model_name = 'FacebookAI/roberta-base'

### NOTE:
# - Test Llama 2?
# - Consider using base BERT (it's smaller?)
# - Test domain-adapted model

# Label names
# id2label = {0: "Unclassified", 1: "Avoidant-1", 2: "Avoidant-2", 3: "Secure", 4: "Preoccupied-1", 5: "Preoccupied-2"}
# label2id = {"Unclassified": 0, "Avoidant-1": 1, "Avoidant-2": 2, "Secure": 3, "Preoccupied-1": 4, "Preoccupied-2": 5}
id2label = {1: "Dismissing", 2: "Secure", 3: "Preoccupied"}
label2id = {"Dismissing": 1, "Secure": 2, "Preoccupied": 3}

#%%
# Load data
data = load_data_with_labels(labels_path, train_data_path)
data["label"] = data["label"].astype(int)

# # Combine turns into one list
patient_turns = data["text"].to_list()

# Labels
labels = data["label"].to_list()

#%%
# Preprocess
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples, truncation=True)

# Tokenize texts and map the tokens to their word IDs.
tokenized_text = data["text"].map(preprocess_function)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Train/val split
train_text, val_text, train_labels, val_labels = train_test_split(tokenized_text, labels, test_size=0.15, random_state=42)

# Create datasets
train_dataset = TensorDataset(train_text, train_labels)
val_dataset = TensorDataset(val_text, val_labels)

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

#%%
# Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, _labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=_labels)

#%%
# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label2id)
#%%
# Transformers trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=val_dataloader,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
#%%
# Train
trainer.train()

print(trainer.evaluate())