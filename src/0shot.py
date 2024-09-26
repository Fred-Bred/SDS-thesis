import pandas as pd
import argparse
from transformers import pipeline, AutoConfig
from huggingface_hub import login

import os

# Login
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)
else:
    raise ValueError("Token not found. Please set the HF_TOKEN environment variable.")

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--length", type=int, required=True, help="Turn length to classify")
parser.add_argument("--mode", type=str, required=True, help="train or test")
parser.add_argument("--size", type=int, required=True, help="Size of the model")
args = parser.parse_args()
length = args.length
mode = args.mode
size = args.size

# Load the data
path = f"Data/PACS_varying_lengths/{mode}_combined_{length}.csv"

data = pd.read_csv(path, sep='\t')

# Login
login(token=token)

# Load and correct the model configuration
config = AutoConfig.from_pretrained(f"meta-llama/Meta-Llama-3.1-{size}B-Instruct")

# Ensure rope_scaling has only the required fields
if hasattr(config, 'rope_scaling'):
    config.rope_scaling = {"type": "linear", "factor": 8.0}

# Initialize the classifier with the correct configuration
classifier = pipeline(model=f"meta-llama/Meta-Llama-3.1-{size}B-Instruct", task="zero-shot-classification", config=config)

label_candidates = ["Avoidant attachment", "Secure attachment", "Preoccupied attachment"]

preds = []
for sample in data["text"]:
    pred = classifier(sample, label_candidates) # Classify the sample
    preds.append(pred["labels"][0]) # Append the predicted label to the list

label_preds = [1 if pred == "Dismissing attachment" else 2 if pred == "Secure attachment" else 3 for pred in preds]

data["predicted_label"] = label_preds

# Save the predictions
data.to_csv(f"Outputs/Llama31_preds_{size}b/{mode}_combined_{length}_predictions_{size}.csv", index=False)