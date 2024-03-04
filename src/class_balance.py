import pandas as pd
from utils.preprocessing.transcript import load_data_with_labels

# Data folder
train_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_train"
val_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_val"
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_labels.xlsx"

# Load training data
train_data = load_data_with_labels(labels_path, train_data_path)
train_data["label"] = train_data["label"].astype(int) - 1 # Convert labels to 0, 1, 2

# Load validation data
val_data = load_data_with_labels(labels_path, val_data_path)
val_data["label"] = val_data["label"].astype(int) - 1 # Convert labels to 0, 1, 2

# Load test data
test_data_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_test"
test_data = load_data_with_labels(labels_path, test_data_path)
test_data["label"] = test_data["label"].astype(int) - 1 # Convert labels to 0, 1, 2

print("Training data class balance:")
print(train_data["label"].value_counts())
print(train_data["label"].value_counts(normalize=True) * 100)

print("\nValidation data class balance:")
print(val_data["label"].value_counts())
print(val_data["label"].value_counts(normalize=True) * 100)

print("\nTest data class balance:")
print(test_data["label"].value_counts())
print(test_data["label"].value_counts(normalize=True) * 100)