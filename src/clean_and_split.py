from utils.preprocessing.transcript import load_data_with_labels
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

# PACS dataset
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_labels.xlsx"
data_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_data"

# Load doc names and labels
pacs_labels = pd.read_excel(labels_path)
docs = pacs_labels["Document"]
labels = pacs_labels["Class3"]

# Stratified split for train/val/test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(
    docs, labels, test_size=0.25, stratify=labels, random_state=42, shuffle=True
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.60, stratify=y_val_test, random_state=42, shuffle=True
)

# Create directories
train_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/train_PACS"
val_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/val_PACS"
test_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/test_PACS"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Copy files to respective directories
for doc in X_train:
    shutil.copy2(os.path.join(data_dir, doc), os.path.join(train_dir, doc))
for doc in X_val:
    shutil.copy2(os.path.join(data_dir, doc), os.path.join(val_dir, doc))
for doc in X_test:
    shutil.copy2(os.path.join(data_dir, doc), os.path.join(test_dir, doc))

### Clean text and save data to csv in MaChAmp format
# Load data
train_data = load_data_with_labels(labels_path, train_dir)
val_data = load_data_with_labels(labels_path, val_dir)
test_data = load_data_with_labels(labels_path, test_dir)

# Remove tabs and newlines from text
train_data['text'] = train_data['text'].str.replace(r'\t', ' ', regex=True)
train_data['text'] = train_data['text'].str.replace(r'\n', ' ', regex=True)
val_data['text'] = val_data['text'].str.replace(r'\t', ' ', regex=True)
val_data['text'] = val_data['text'].str.replace(r'\n', ' ', regex=True)
test_data['text'] = test_data['text'].str.replace(r'\t', ' ', regex=True)
test_data['text'] = test_data['text'].str.replace(r'\n', ' ', regex=True)

# Save data
train_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/train_PACS.csv", index=False, sep="\t")
val_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/val_PACS.csv", index=False, sep="\t")
test_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/test_PACS.csv", index=False, sep="\t")

### Make varying length instances
# Combine turns within documents to reach the target length
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
            if current_length >= target_length:  # Add this line
                new_row = pd.DataFrame({"text": [current_turn], "label": [current_label], "document": [current_document]})
                combined_data = pd.concat([combined_data, new_row], ignore_index=True)
            current_length = turn_length
            current_document = row["document"]
            current_turn = row["text"] + " "
            current_label = row["label"]
    combined_data.reset_index(drop=True, inplace=True)
    return combined_data

# Combine and clean turns with min_length=100
train_combined_100 = combine_turns(train_data, 100)
val_combined_100 = combine_turns(val_data, 100)
test_combined_100 = combine_turns(test_data, 100)

# Combine and clean turns with min_length=150
train_combined_150 = combine_turns(train_data, 150)
val_combined_150 = combine_turns(val_data, 150)
test_combined_150 = combine_turns(test_data, 150)

# Combine and clean turns with min_length=250
train_combined_250 = combine_turns(train_data, 250)
val_combined_250 = combine_turns(val_data, 250)
test_combined_250 = combine_turns(test_data, 250)

# Save combined data 100
train_combined_100.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/train_combined_100.csv", index=False, sep="\t")
val_combined_100.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/val_combined_100.csv", index=False, sep="\t")
test_combined_100.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/test_combined_100.csv", index=False, sep="\t")

# Save combined data 150
train_combined_150.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/train_combined_150.csv", index=False, sep="\t")
val_combined_150.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/val_combined_150.csv", index=False, sep="\t")
test_combined_150.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/test_combined_150.csv", index=False, sep="\t")

# Save combined data 250
train_combined_250.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/train_combined_250.csv", index=False, sep="\t")
val_combined_250.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/val_combined_250.csv", index=False, sep="\t")
test_combined_250.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/test_combined_250.csv", index=False, sep="\t")