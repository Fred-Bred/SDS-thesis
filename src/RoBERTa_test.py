#%%
# Imports
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import evaluate

from utils.transcript import load_patient_turns_from_folder, split_into_chunks
from utils.model import RoBERTaTorch
#%%
# Training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01, # Check implementation against BatchNorm or LayerNorm
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

max_length = 512

folder_path = r'C:\Users\frbre\OneDrive\01 Dokumenter\01 Uni\SDS Thesis\data\test'

model_name = 'xlm-r-base'

id2label = {0: "Unclassified", 1: "Avoidant-1", 2: "Avoidant-2", 3: "Secure", 4: "Preoccupied-1", 5: "Preoccupied-2"}
label2id = {"Unclassified": 0, "Avoidant-1": 1, "Avoidant-2": 2, "Secure": 3, "Preoccupied-1": 4, "Preoccupied-2": 5}

##%
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, max_length=max_length, padding=True, truncation=True)

#%%
#### NOTES ####
# Load data
# Tokenize data

# Instantiate model:
#   - Try RoBERTaTorch from utils/model.py
#   - Try AutoModelForSequenceClassification from transformers
#   - Try TorchBaseModel from utils/model.py (requires instantiating a pre-trained model first)

#%%
# Metrics
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
#%%
# Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,

)
#%%
# Train
trainer.train()