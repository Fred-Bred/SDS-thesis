#%%
# Imports
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import evaluate
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from utils.preprocessing.transcript import load_patient_turns_from_folder, split_into_chunks
from utils.preprocessing.preprocessing import pad_tensors
from utils.model import RoBERTaTorch
#%%
# Training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_model",
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
folder_path = r'C:\Users\frbre\OneDrive\01 Dokumenter\01 Uni\SDS Thesis\data\test'

# Base model name
model_name = 'FacebookAI/roberta-base'

### NOTE:
# - Test Llama 2?
# - Consider using base BERT (it's smaller?)
# - Test domain-adapted model

# Label names
id2label = {0: "Unclassified", 1: "Avoidant-1", 2: "Avoidant-2", 3: "Secure", 4: "Preoccupied-1", 5: "Preoccupied-2"}
label2id = {"Unclassified": 0, "Avoidant-1": 1, "Avoidant-2": 2, "Secure": 3, "Preoccupied-1": 4, "Preoccupied-2": 5}

#%%
# Load data
patient_turns = load_patient_turns_from_folder(folder_path)

# Split into chunks
patient_chunks = split_into_chunks(patient_turns, max_length=max_length)

# Combine chunks into one list
patient_turns = [turn for patient in patient_chunks for turn in patient]

#%%
# Generate fake labels array
length = len(patient_turns)

# Generate a fake labels array
fake_labels = np.eye(6)[np.random.choice(6, length)]
# %%
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length, padding=True, truncation=True)

# Tokenize texts and map the tokens to their word IDs.
input_ids = []

for sent in patient_turns:
    encoded_text = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                   )
    input_ids.append(encoded_text)

# Pad input tokens
input_ids = pad_tensors(input_ids, max_length)

### NOTE:
#   - Try using the tokenizer to pad the input tokens??
#   - Consider padding with DataCollatorWithPadding from transformers to conform with the trainer

# Create attention masks
attention_masks = []

for sent in input_ids:
    
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]
    
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

#%%
# Make train/val split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, fake_labels, 
                                                            random_state=2018, test_size=0.1)
# Performing same steps on the attention masks
train_masks, validation_masks, _, _ = train_test_split(attention_masks, fake_labels,
                                             random_state=2018, test_size=0.1)

# Convert to tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Create dataloaders
batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#%%
#### NOTES ####
# Load labels

# Instantiate model:
#   - Try RoBERTaTorch from utils/model.py
#   - Try AutoModelForSequenceClassification from transformers
#   - Try TorchBaseModel from utils/model.py (requires instantiating a pre-trained model first)

# Instantiate trainer:
#   - Try Trainer from transformers
#   - Try Trainer from utils/trainer.py

#%%
# Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

#%%
# Model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6, output_attentions=True, output_hidden_states=True)
#%%
# Transformers trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=validation_dataloader,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)
#%%
# Train
trainer.train()

#%%
# utils trainer
