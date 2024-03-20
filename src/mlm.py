from transformers import auto_tokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
from utils.preprocessing.preprocessing import preprocess_mlm, group_texts
import argparse

import math

# Arguments
parser = argparse.ArgumentParser(description="Train a masked language model")
parser.add_argument("--train_dir", type=str, help="Path to the training data directory")
parser.add_argument("--test_dir", type=str, help="Path to the test data directory")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--epochs", type=int, help="Number of epochs")
parser.add_argument("--weight_decay", type=float, help="Weight decay")

args = parser.parse_args()

# Hyperparameters
lr = 2e-5 if not args.lr else args.lr
epochs = 10 if not args.epochs else args.epochs
w_decay = 0.01 if not args.weight_decay else args.weight_decay

# Paths
train_files = []
train_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data" if not args.train_dir else args.train_dir
test_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data" if not args.train_dir else args.test_dir
output_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/trained_models/pretrained" if not args.output_dir else args.output_dir

# Load data
# dataset = load_dataset("text", data_files={"train": train_files, "test": test_files}, sample_by='paragraph')
therapy_train = load_dataset("text", data_dir=train_dir, sample_by='paragraph')
therapy_test = load_dataset("text", data_dir=test_dir, sample_by='paragraph')

# Preprocess data
tokenizer = auto_tokenizer("roberta-base")

train_dataset = therapy_train.map(preprocess_mlm, batched=True)
test_dataset = therapy_test.map(preprocess_mlm, batched=True)

block_size = 128
mlm_train_dataset = train_dataset.map(group_texts, batched=True)
mlm_test_dataset = test_dataset.map(group_texts, batched=True)

# Dynamic padding
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Model
model = AutoModelForMaskedLM.from_pretrained("roberta-base")

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=lr,
    num_train_epochs=epochs,
    weight_decay=w_decay,
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mlm_train_dataset,
    eval_dataset=mlm_test_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(output_dir)

eval_results = trainer.evaluate()

print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")