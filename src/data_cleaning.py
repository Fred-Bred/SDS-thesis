from utils.preprocessing.transcript import load_data_with_labels

import pandas as pd

# PACS dataset
labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_labels.xlsx"

# Load training data
train_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_train"
train_data = load_data_with_labels(labels_path, train_path)

# Remove tabs and newlines from text
train_data['text'] = train_data['text'].str.replace(r'\t', ' ', regex=True)
train_data['text'] = train_data['text'].str.replace(r'\n', ' ', regex=True)

# Load validation data
val_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_val"
val_data = load_data_with_labels(labels_path, val_path)

# Remove tabs and newlines from text
val_data['text'] = val_data['text'].str.replace(r'\t', ' ', regex=True)
val_data['text'] = val_data['text'].str.replace(r'\n', ' ', regex=True)

# Load test data
test_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_test"
test_data = load_data_with_labels(labels_path, test_path)

# Remove tabs and newlines from text
test_data['text'] = test_data['text'].str.replace(r'\t', ' ', regex=True)
test_data['text'] = test_data['text'].str.replace(r'\n', ' ', regex=True)

# Save data
train_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_train.csv", index=False, sep="\t")
val_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_val.csv", index=False, sep="\t")
test_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_test.csv", index=False, sep="\t")

# AnnoMI dataset

# Load data
annomi = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/AnnoMI/AnnoMI-simple.csv")

# Clean text
timestamp = r'\b(\d{1,2}:[0-5][0-9]:[0-5][0-9])\b'
annomi['utterance_text'] = annomi['utterance_text'].str.replace(r'\[unintelligible ' + timestamp + r'\]', '<UNK>', regex=True) # remove unintelligible

# Select client speech turns
client = annomi[annomi['interlocutor'] == 'client']

# Save data
client.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/AnnoMI_cleaned_client.csv", index=False, sep="\t")