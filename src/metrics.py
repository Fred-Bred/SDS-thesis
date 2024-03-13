from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

# Parse arguments
model_date = sys.argv[1]
model_number = sys.argv[2]

# Define paths
output_folder = f"Outputs/trained_models/{model_date}"

# Define the classes
classes = ["Dismissing", "Secure", "Preoccupied"]

# Load predictions 
preds = pd.read_csv(f'{output_folder}/pacs.csv', sep='\t')
pred_labels = preds.iloc[:, 1].tolist()

# Load true labels
targets = pd.read_csv('Data/PACS_val.csv', sep='\t')
true_labels = targets.iloc[:, 1].tolist()

# Compute metrics
cm = confusion_matrix(true_labels, pred_labels)
cr = classification_report(true_labels, pred_labels, target_names=classes, zero_division=0)

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average='macro')
recall = recall_score(true_labels, pred_labels, average='macro')

# Print metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print('\nConfusion matrix:')
print(cm)
print('\nClassification report:')
print(cr)

# Plot and save the confusion matrix
plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix | Model {model_number} | {model_date}')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.savefig(f'{output_folder}/confusion_matrix_{model_date}.png')

# Write metrics to text file
with open(f'{output_folder}/metrics_model_{model_number}.txt', 'w') as f:
    f.write('\n\nValidation metrics:\n')
    f.write(f'Accuracy: {accuracy}\n')
    f.write(f'Precision: {precision}\n')
    f.write(f'Recall: {recall}\n')
    f.write('\nConfusion matrix:\n')
    f.write(str(cm))
    f.write('\n\nClassification report:\n')
    f.write(str(cr))

# Add a new column to the dataframes representing the length of the instances
preds['length'] = preds.iloc[:, 0].apply(lambda x: len(x.split()))
targets['length'] = targets.iloc[:, 0].apply(lambda x: len(x.split()))

# Group the data by length
grouped_preds = preds.groupby('length')
grouped_targets = targets.groupby('length')

# Initialize lists to store the metrics for each length
lengths = []
accuracies = []
precisions = []
recalls = []

# Compute the metrics for each length
for length in grouped_preds.groups.keys():
    pred_labels = grouped_preds.get_group(length).iloc[:, 1].tolist()
    true_labels = grouped_targets.get_group(length).iloc[:, 1].tolist()
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='macro')
    recall = recall_score(true_labels, pred_labels, average='macro')
    
    lengths.append(length)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)

# Plot the results
plt.figure(figsize=(10, 10))
plt.plot(lengths, accuracies, label='Accuracy')
plt.plot(lengths, precisions, label='Precision')
plt.plot(lengths, recalls, label='Recall')
plt.xlabel('Sentence Length (words)')
plt.ylabel('Metric Value')
plt.legend()
plt.show()