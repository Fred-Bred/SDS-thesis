from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
from collections import Counter
import os

# Parse arguments
parser = argparse.ArgumentParser(description="Compute metrics for a model")
parser.add_argument("--model_name", type=str, help="Model name (e.g. roberta-base_50)")
parser.add_argument("--mode", type=str, help="Mode (val or test)", default="val")

args = parser.parse_args()

# Define arguments
model_name = args.model_name # Model name (e.g. roberta-base_50)
min_length = model_name.split("_")[-1] # Minimum length of the instances
mode = args.mode # Mode (val or test)

# Define output path
output_folder = f"Outputs/trained_models/k-folds/{model_name}"
assert os.path.exists(output_folder), f"Folder {output_folder} does not exist. Exiting..."

# Define the classes
classes = ["Avoidant", "Secure", "Anxious"]

# Iniitialize list to store metrics
model_accuracies = []

# loop through the k-folds
for i in range(1, 6):
    # Load data
    if mode == "val":
        if min_length != "0":
            targets_path = f"Data/k-folds/split{i}/val_{min_length}.csv"
        else:
            targets_path = f"Data/k-folds/split{i}/val_PACS.csv"

    elif mode == "test":
        if min_length != "0":
            targets_path = f"Data/PACS_varying_lengths/test_combined_{min_length}.csv"
        else:
            targets_path = f"Data/test_PACS.csv"


    targets = pd.read_csv(targets_path, sep="\t")
    true_labels = targets.iloc[:, 1].tolist()

    # Load predictions
    predictions_path = f"{output_folder}/split{i}_{mode}_preds.csv"
    predictions = pd.read_csv(predictions_path, sep="\t")
    pred_labels = predictions.iloc[:, 1].tolist()

    # Compute metrics
    accuracy = round(accuracy_score(true_labels, pred_labels) *100, 4)
    precision = round(precision_score(true_labels, pred_labels, average="macro") * 100, 4)
    recall = round(recall_score(true_labels, pred_labels, average="macro") * 100, 4)
    f1 = round(f1_score(true_labels, pred_labels, average="macro") * 100, 4)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_norm = confusion_matrix(true_labels, pred_labels, normalize="all")
    cm_norm_df = pd.DataFrame(cm_norm, index=classes, columns=classes)

    # Save metrics
    model_accuracies.append(accuracy)

    # Save metrics to file
    with open(f"{output_folder}/split{i}_{mode}_metrics.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n\n")
        f.write(f"Confusion matrix:\n")
        f.write(f"{cm_df}\n\n")
        f.write(f"Normalized confusion matrix:\n")
        f.write(f"{cm_norm_df}\n")

# Compute average accuracy
average_accuracy = round(np.mean(model_accuracies), 4)
std_accuracy = round(np.std(model_accuracies), 4)

best_model = np.argmax(model_accuracies) + 1
best_metrics_txt = f"{output_folder}/split{best_model}_{mode}_metrics.txt"
with open(best_metrics_txt, "r") as f:
    best_metrics = f.read()

# Save average accuracy to file
with open(f"{output_folder}/{mode}_average_accuracy.txt", "w") as f:
    f.write(f"Average accuracy: {average_accuracy}\n")
    f.write(f"Standard deviation: {std_accuracy}\n")
    f.write(f"Accuracies: {model_accuracies}\n")
    f.write(f"Model: {model_name}\n")
    f.write(f"Mode: {mode}\n")
    f.write(f"Min length: {min_length}\n")
    f.write(f"Classes: {classes}\n\n")
    f.write(f"Best model: {best_model}\n")
    f.write(f"{best_metrics}\n")
    f.write(f"Metrics saved to {output_folder}\n")

print(f"Average accuracy: {average_accuracy}")
print(f"Standard deviation: {std_accuracy}")
print(f"Accuracies: {model_accuracies}")
print(f"Best model: {model_name}")