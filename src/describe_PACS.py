from utils.preprocessing.transcript import load_data_with_labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
pacs_labels_path = "Data/PACS_labels_updated.xlsx"
all_docs_path = "Data/PACS_data"

train_path = "Data/train_PACS.csv"
val_path = "Data/val_PACS.csv"
test_path = "Data/test_PACS.csv"

# Load data
full = load_data_with_labels(pacs_labels_path, all_docs_path)

train = pd.read_csv(train_path, sep="\t")
val = pd.read_csv(val_path, sep="\t")
test = pd.read_csv(test_path, sep="\t")

# Number of documents
all_docs = full["document"].unique()
train_docs = train["document"].unique()
val_docs = val["document"].unique()
test_docs = test["document"].unique()

# Class balance
full_classes = full["label"].value_counts(normalize=True)

train_classes = train["label"].value_counts(normalize=True)
val_classes = val["label"].value_counts(normalize=True)
test_classes = test["label"].value_counts(normalize=True)

# Turn lengths
full["turn_length"] = full["text"].apply(lambda x: len(x.split()))
avg_turn_length = np.mean(full["turn_length"])

train["turn_length"] = train["text"].apply(lambda x: len(x.split()))
train_avg_turn_length = np.mean(train["turn_length"])

val["turn_length"] = val["text"].apply(lambda x: len(x.split()))
val_avg_turn_length = np.mean(val["turn_length"])

test["turn_length"] = test["text"].apply(lambda x: len(x.split()))
test_avg_turn_length = np.mean(test["turn_length"])

print("Writing to txt...")

# Write to txt
with open("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/PACS_descriptives.txt", 'w') as f:
    f.write("-------")
    f.write("\n***Full PACS data set description***\n")
    f.write(f"\nNumber of documents loaded: {len(all_docs)}\n")
    f.write(f"Number of patient turns: {len(full)}\n")
    f.write(f"Average patient turns per document: {len(full)/len(all_docs)}\n")
    f.write(f"Average patient turn length: {avg_turn_length} words\n")
    f.write(f"\nClass balance:\n{full_classes}\n")
    f.write("-------")
    f.write("\n***Train set description***\n")
    f.write(f"\nNumber of documents loaded: {len(train_docs)}\n")
    f.write(f"Number of patient turns: {len(train)}\n")
    f.write(f"Average patient turns per document: {len(train)/len(train_docs)}\n")
    f.write(f"Average patient turn length: {train_avg_turn_length} words\n")
    f.write(f"\nClass balance:\n{train_classes}\n")
    f.write("-------")
    f.write("\n***Validation set description***\n")
    f.write(f"\nNumber of documents loaded: {len(val_docs)}\n")
    f.write(f"Number of patient turns: {len(val)}\n")
    f.write(f"Average patient turns per document: {len(val)/len(val_docs)}\n")
    f.write(f"Average patient turn length: {val_avg_turn_length} words\n")
    f.write(f"\nClass balance:\n{val_classes}\n")
    f.write("-------")
    f.write("\n***Test set description***\n")
    f.write(f"\nNumber of documents loaded: {len(test_docs)}\n")
    f.write(f"Number of patient turns: {len(test)}\n")
    f.write(f"Average patient turns per document: {len(test)/len(test_docs)}\n")
    f.write(f"Average patient turn length: {test_avg_turn_length} words\n")
    f.write(f"\nClass balance:\n{test_classes}\n")
    f.write("-------")

print("Writing to txt... Done!")

print("\nPlotting...")

# Plot class balance
plt.figure(figsize=(10, 5))
sns.barplot(x=full['label'].value_counts().index, y=full['label'].value_counts().values)
plt.title("Class balance in full PACS data set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/full_class_balance.png")

plt.figure(figsize=(10, 5))
sns.barplot(x=train['label'].value_counts().index, y=train['label'].value_counts().values)
plt.title("Class balance in train set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/train_class_balance.png")

plt.figure(figsize=(10, 5))
sns.barplot(x=val['label'].value_counts().index, y=val['label'].value_counts().values)
plt.title("Class balance in validation set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/val_class_balance.png")

plt.figure(figsize=(10, 5))
sns.barplot(x=test['label'].value_counts().index, y=test['label'].value_counts().values)
plt.title("Class balance in test set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/test_class_balance.png")

# Plot turn length distributions
plt.figure(figsize=(10, 5))
sns.histplot(full["turn_length"], bins=50)
plt.title("Turn length distribution in full PACS data set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/full_turn_length_distribution.png")

plt.figure(figsize=(10, 5))
sns.histplot(train["turn_length"], bins=50)
plt.title("Turn length distribution in train set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/train_turn_length_distribution.png")

plt.figure(figsize=(10, 5))
sns.histplot(val["turn_length"], bins=50)
plt.title("Turn length distribution in validation set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/val_turn_length_distribution.png")

plt.figure(figsize=(10, 5))
sns.histplot(test["turn_length"], bins=50)
plt.title("Turn length distribution in test set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/test_turn_length_distribution.png")

# Plot turn length distributions by class
# Define the color palette
palette = {1: 'blue', 2: 'orange', 3: 'green'}

plt.figure(figsize=(10, 5))
sns.histplot(data=full, x="turn_length", bins=50, hue="label", palette=palette, multiple="dodge", legend=True)
plt.title("Turn length distribution by class in full PACS data set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/turn_length_distribution_by_class.png")

plt.figure(figsize=(10, 5))
sns.histplot(data=train, x="turn_length", bins=50, hue="label", palette=palette, multiple="dodge", legend=True)
plt.title("Turn length distribution by class in train set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/train_turn_length_distribution_by_class.png")

plt.figure(figsize=(10, 5))
sns.histplot(data=val, x="turn_length", bins=50, hue="label", palette=palette, multiple="dodge", legend=True)
plt.title("Turn length distribution by class in validation set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/val_turn_length_distribution_by_class.png")

plt.figure(figsize=(10, 5))
sns.histplot(data=test, x="turn_length", bins=50, hue="label", palette=palette, multiple="dodge", legend=True)
plt.title("Turn length distribution by class in test set")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/test_turn_length_distribution_by_class.png")

print("Plotting... Done!")

# varying lengths
print("Loading varying lengths...")

train_combined_50 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/train_combined_50.csv", sep="\t")
val_combined_50 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/val_combined_50.csv", sep="\t")
test_combined_50 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/test_combined_50.csv", sep="\t")

train_combined_100 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/train_combined_100.csv", sep="\t")
val_combined_100 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/val_combined_100.csv", sep="\t")
test_combined_100 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/test_combined_100.csv", sep="\t")

train_combined_150 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/train_combined_150.csv", sep="\t")
val_combined_150 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/val_combined_150.csv", sep="\t")
test_combined_150 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/test_combined_150.csv", sep="\t")

train_combined_250 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/train_combined_250.csv", sep="\t")
val_combined_250 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/val_combined_250.csv", sep="\t")
test_combined_250 = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_varying_lengths/test_combined_250.csv", sep="\t")

# Turn lengths
train_combined_50["turn_length"] = train_combined_50["text"].apply(lambda x: len(x.split()))
val_combined_50["turn_length"] = val_combined_50["text"].apply(lambda x: len(x.split()))
test_combined_50["turn_length"] = test_combined_50["text"].apply(lambda x: len(x.split()))

train_combined_100["turn_length"] = train_combined_100["text"].apply(lambda x: len(x.split()))
train_combined_150["turn_length"] = train_combined_150["text"].apply(lambda x: len(x.split()))
train_combined_250["turn_length"] = train_combined_250["text"].apply(lambda x: len(x.split()))

val_combined_100["turn_length"] = val_combined_100["text"].apply(lambda x: len(x.split()))
val_combined_150["turn_length"] = val_combined_150["text"].apply(lambda x: len(x.split()))
val_combined_250["turn_length"] = val_combined_250["text"].apply(lambda x: len(x.split()))

test_combined_100["turn_length"] = test_combined_100["text"].apply(lambda x: len(x.split()))
test_combined_150["turn_length"] = test_combined_150["text"].apply(lambda x: len(x.split()))
test_combined_250["turn_length"] = test_combined_250["text"].apply(lambda x: len(x.split()))

print("Writing to txt...")

# Write to txt
with open("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Outputs/Descriptives/PACS_varying_lengths_descriptives.txt", 'w') as f:
    f.write("Minimum length: 50\n")
    f.write("Train set min_len 50: \n")
    f.write(train_combined_50["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(train_combined_50["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Validation set min_len 50: \n")
    f.write(val_combined_50["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(val_combined_50["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Test set min_len 50: \n")
    f.write(test_combined_50["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(test_combined_50["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("--------------------\n\n")
    f.write("Minimum length: 100\n")
    f.write("Train set min_len 100: \n")
    f.write(train_combined_100["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(train_combined_100["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Validation set min_len 100: \n")
    f.write(val_combined_100["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(val_combined_100["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Test set min_len 100: \n")
    f.write(test_combined_100["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(test_combined_100["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("--------------------\n\n")
    f.write("Minimum length: 150\n")
    f.write("Train set min_len 150: \n")
    f.write(train_combined_150["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(train_combined_150["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Validation set min_len 150: \n")
    f.write(val_combined_150["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(val_combined_150["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Test set min_len 150: \n")
    f.write(test_combined_150["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(test_combined_150["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("--------------------\n\n")
    f.write("Minimum length: 250\n")
    f.write("Train set min_len 250: \n")
    f.write(train_combined_250["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(train_combined_250["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Validation set min_len 250: \n")
    f.write(val_combined_250["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(val_combined_250["label"].value_counts(normalize=True).to_string())
    f.write("\n\n")
    f.write("Test set min_len 250: \n")
    f.write(test_combined_250["turn_length"].describe().to_string())
    f.write("\n")
    f.write("Class balance:\n")
    f.write(test_combined_250["label"].value_counts(normalize=True).to_string())

print("Writing to txt... Done!")