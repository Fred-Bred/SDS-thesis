from utils.preprocessing.transcript import load_data_with_labels, combine_turns
import os
import shutil
from sklearn.model_selection import train_test_split
import pandas as pd

# PACS dataset
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_labels_updated.xlsx"
data_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_data"

# Load doc names and labels
pacs_labels = pd.read_excel(labels_path)
docs = pacs_labels["Document"]
labels = pacs_labels["Class3"]

# Make 5 splits of the data
for i in range (1, 6):
    train_dir = f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/train_docs"
    val_dir = f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/val_docs"
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Stratified split for train/val sets
    X_train, X_val, y_train, y_val = train_test_split(
        docs, labels, test_size=0.40, stratify=labels, random_state=i, shuffle=True
    )

    # Copy files to respective directories
    for doc in X_train:
        shutil.copy2(os.path.join(data_dir, doc), os.path.join(train_dir, doc))
    for doc in X_val:
        shutil.copy2(os.path.join(data_dir, doc), os.path.join(val_dir, doc))

    ### Clean text and save data to csv in MaChAmp format
    # Load data
    train_data = load_data_with_labels(labels_path, train_dir)
    val_data = load_data_with_labels(labels_path, val_dir)

    # Remove tabs and newlines from text
    train_data['text'] = train_data['text'].str.replace(r'\t', ' ', regex=True)
    train_data['text'] = train_data['text'].str.replace(r'\n', ' ', regex=True)
    val_data['text'] = val_data['text'].str.replace(r'\t', ' ', regex=True)
    val_data['text'] = val_data['text'].str.replace(r'\n', ' ', regex=True)

    # Save data
    train_data.to_csv(f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/train_PACS.csv", index=False, sep="\t")
    val_data.to_csv(f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/val_PACS.csv", index=False, sep="\t")

# Make varying lengths of instances
for i in range (1, 6):
    for target_length in [50, 100, 150, 250]:
        train_data = pd.read_csv(f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/train_PACS.csv", sep="\t")
        val_data = pd.read_csv(f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/val_PACS.csv", sep="\t")

        # Combine turns to make varying target_length instances
        combined_train = combine_turns(train_data, target_length)
        combined_val = combine_turns(val_data, target_length)

        # Save data
        combined_train.to_csv(f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/train_{target_length}.csv", index=False, sep="\t")
        combined_val.to_csv(f"/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/k-folds/split{i}/val_{target_length}.csv", index=False, sep="\t")