# Imports
import sys

import pandas as pd

from utils.preprocessing.transcript import load_data_with_labels

# Arguments
target_length = sys.argv[1] # Target length of the instances

# Load data
pacs_train = load_data_with_labels("Data/PACS_labels.xlsx", "Data/PACS_train")
pacs_dev = load_data_with_labels("Data/PACS_labels.xlsx", "Data/PACS_val")
pacs_test = load_data_with_labels("Data/PACS_labels.xlsx", "Data/PACS_test")

# Combine turns within documents until the target length is reached
def combine_turns(data, target_length):
    combined_data = pd.DataFrame(columns=["text", "label", "document"])
    current_length = 0
    current_document = ""
    current_turn = ""
    current_label = ""
    for index, row in data.iterrows():
        turn_length = len(row["text"].split())
        if current_length + turn_length < target_length and (current_document == row["document"] or current_document == ""):
            current_document = row["document"]
            current_turn += row["text"] + " "
            current_label = row["label"]
            current_length += turn_length
        else:
            combined_data = combined_data.append({"text": current_turn, "label": current_label, "document": current_document}, ignore_index=True)
            current_length = turn_length
            current_document = row["document"]
            current_turn = row["text"] + " "
            current_label = row["label"]
    return combined_data

pacs_train_combined = combine_turns(pacs_train, int(target_length))
pacs_dev_combined = combine_turns(pacs_dev, int(target_length))
pacs_test_combined = combine_turns(pacs_test, int(target_length))

# Save the combined data
pacs_train_combined.to_csv(f"Data/PACS_varying_lengths/train_length_{target_length}.csv", index=False, sep="\t")
pacs_dev_combined.to_csv(f"Data/PACS_varying_lengths/val_length_{target_length}.csv", index=False, sep="\t")
pacs_test_combined.to_csv(f"Data/PACS_varying_lengths/test_length_{target_length}.csv", index=False, sep="\t")
