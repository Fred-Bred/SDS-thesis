import pandas as pd
from transformers import RobertaTokenizer
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset")
args = parser.parse_args()
dataset_path = args.dataset

# Determine format of dataset
if dataset_path.endswith(".csv"):
    df = pd.read_csv(args.dataset, sep="\t")
elif dataset_path.endswith(".txt"):
    # Read the text file
    with open(dataset_path, "r") as file:
        lines = file.readlines()
    # Create a DataFrame
    df = pd.DataFrame(lines, columns=["text"])

# Create a RoBERTa tokenizer instance
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize the text data and count the number of tokens
token_counts = []
for text in df["text"]:
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors="pt",
    )
    token_counts.append(len(inputs["input_ids"][0]))

print("Total number of tokens:", sum(token_counts))