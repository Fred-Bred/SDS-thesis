"""Copied-in trainer class from previous project - ADJUST"""

# Import libraries
import os
from datetime import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, Precision, Recall

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 3

# class Trainer:
#     def __init__(self):
#         self.model = None
#         self.optimizer = None
#         self.loss_fn = None
#         self.train_loader = None
#         self.val_loader = None
#         self.source = None
#         self.history = {'train_loss': [], 'val_loss': [], 'train_loss_batch': [], 'val_loss_batch': []}

#     def compile(self, model, optimizer, learning_rate, loss_fn, weight_decay=0.01, model_name=None):
#         self.model = model
#         self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay = weight_decay)
#         self.loss_fn = loss_fn
#         if model_name is None:
#             self.model_name = "unspecified"
#         else:
#             self.model_name = model_name
    
#     def calculate_accuracy(self, logits, labels):
#         preds = torch.argmax(logits, dim=1)
#         return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
#     def calculate_f1(self, logits, labels):
#         preds = torch.argmax(logits, dim=1)
#         return f1_score(labels, preds, average='weighted')
    
#     def calculate_precision(self, logits, labels):
#         preds = torch.argmax(logits, dim=1)
#         return precision_score(labels, preds, average='weighted')
    
#     def calculate_recall(self, logits, labels):
#         preds = torch.argmax(logits, dim=1)
#         return recall_score(labels, preds, average='weighted')

#     def fit(self, num_epochs, train_loader, device, val_loader=None, patience=5, min_delta=0.0001):
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.patience = patience
#         self.best_val_loss = float('inf')
#         self.current_patience = 0
#         self.epochs_without_improvement = 0
#         self.min_delta = min_delta
#         figure_num = 0

#         for epoch in range(num_epochs):
#             self.model.train()
#             total_loss = 0.0
#             with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
#                 for batch in self.train_loader:
#                     texts = batch['input_ids'].to(device)
#                     attention_mask = batch['attention_mask'].to(device)
#                     labels = batch['targets'].to(device)
#                     self.optimizer.zero_grad()
#                     outputs = self.model(texts, attention_mask=attention_mask)
#                     logits = outputs.logits
#                     loss = self.loss_fn(logits, labels)
#                     loss.backward()
#                     self.optimizer.step()
#                     total_loss += loss.item()

#                     # Update the progress bar
#                     pbar.update(1)
#             avg_loss = total_loss / len(self.train_loader)
#             self.history['train_loss'].append(avg_loss)
#             print(f"Training Loss: {avg_loss}")

#             if self.val_loader is not None:
#                 val_loss = self.evaluate(self.val_loader, device, "Validation")
#                 self.history['val_loss'].append(val_loss)

#                 # Plot the loss
#                 plt.plot(self.history['train_loss'], label='train_loss')
#                 plt.plot(self.history['val_loss'], label='val_loss')
#                 plt.xlabel('Epochs')
#                 plt.ylabel('Loss')
#                 plt.title('Loss over epochs')
#                 plt.legend()
#                 os.makedirs('Outputs/Figures', exist_ok=True)
#                 plt.savefig(f'Outputs/Figures/loss_plot{str(figure_num)}.png')
#                 plt.clf()

#                 # Check for early stopping
#                 if val_loss < self.best_val_loss:
#                     self.best_val_loss = val_loss
#                     self.current_patience = 0
#                 else:
#                     self.current_patience += 1
#                     if self.current_patience >= self.patience:
#                         print(f"Early stopping after {epoch + 1} epochs.")
#                         break

#             ### save checkpoint of model and plot
#             ### MODEL
#             try:
#                 os.makedirs('Outputs/trained_models', exist_ok=True)
#                 model_name = f"{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.train_loader.dataset)}_BATCHSIZE_{self.train_loader.batch_size}.pt"
#                 torch.save(self.model.state_dict(), 'Outputs/trained_models/' + model_name)
#                 print(f'Checkpoint after epoch {epoch+1} saved successfully')
#             except Exception as e:
#                 print(f"Error saving checkpoint with name: {e}")
#                 model_name = f"checkpoint_TIME_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
#                 torch.save(self.model.state_dict(), 'Outputs/trained_models/' + model_name)
#                 print('Unnamed checkpoint saved successfully')

#             ### PLOT
#             try:
#                 history = self.history
#                 # Plot the loss history
#                 plt.plot(history['train_loss'], label='Train Loss')
#                 plt.plot(history['val_loss'], label='Validation Loss')
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Loss')
#                 plt.title('Training History')
#                 plt.legend()
#                 plt.savefig(f"Outputs/trained_models/{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.train_loader.dataset)}_BATCHSIZE_{self.train_loader.batch_size}.png")
#             except:
#                 print('Error generating plot')

#     def evaluate(self, data_loader, device, mode="Test"):
#         self.model.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for batch in data_loader:
#                 # Move the data to the GPU
#                 texts = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 labels = batch['targets'].to(device)
#                 outputs = self.model(texts, attention_mask=attention_mask)
#                 logits = outputs.logits
#                 loss = self.loss_fn(logits, labels)
#                 total_loss += loss.item()

#         avg_loss = total_loss / len(data_loader)
#         print(f"{mode} Loss: {avg_loss}")
#         return avg_loss
    
#     def save(self, filepath):
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'optimizer_state_dict': self.optimizer.state_dict(),
#             'history': self.history
#         }, filepath)
#         print(f"Model and training history saved to {filepath}")

#     def load(self, filepath, source):
#         if source == "cpu":
#             checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
#         else:
#             checkpoint = torch.load(filepath)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.history = checkpoint['history']
#         print(f"Model and training history loaded from {filepath}")
    
class Trainer:
    def __init__(self, num_labels):
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.source = None
        self.num_labels = num_labels
        self.accuracy = Accuracy(task="multiclass", average=None, num_classes=self.num_labels)
        self.precision = Precision(task="multiclass", average=None, num_classes=self.num_labels)
        self.recall = Recall(task="multiclass", average=None, num_classes=self.num_labels)
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_loss_batch': [],
            'val_loss_batch': [],
            'train_acc': [],
            'val_acc': [],
            'train_precision': [],
            'val_precision': [],
            'train_recall': [],
            'val_recall': []
        }

    def compile(self, model, optimizer, learning_rate, loss_fn, weight_decay=0.01, model_name=None):
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), lr=learning_rate, weight_decay = weight_decay)
        self.loss_fn = loss_fn
        if model_name is None:
            self.model_name = "unspecified"
        else:
            self.model_name = model_name

    def fit(self, num_epochs, train_loader, device, val_loader=None, patience=5, min_delta=0.0001):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.patience = patience
        self.best_val_loss = float('inf')
        self.current_patience = 0
        self.epochs_without_improvement = 0
        self.min_delta = min_delta
        figure_num = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
                for batch in self.train_loader:
                    # Make predictions
                    texts = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['targets'].to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(texts, attention_mask=attention_mask)
                    logits = outputs.logits

                    # Calculate train metrics
                    loss = self.loss_fn(logits, labels)
                    train_acc = self.accuracy(logits.softmax(dim=1), labels)
                    train_precision = self.precision(logits.softmax(dim=1), labels)
                    train_recall = self.recall(logits.softmax(dim=1), labels)

                    # Backpropagation
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    # Update the progress bar
                    pbar.update(1)
            
            # Compute metrics
            avg_loss = total_loss / len(self.train_loader)
            train_acc = self.accuracy.compute()
            train_precision = self.precision.compute()
            train_recall = self.recall.compute()

            # Add metrics to history
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(train_acc)
            self.history['train_precision'].append(train_precision)
            self.history['train_recall'].append(train_recall)

            # Print metrics
            print(f"Training Loss: {avg_loss}")
            print(f"Training Accuracy: {train_acc}")
            print(f"Training Precision: {train_precision}")
            print(f"Training Recall: {train_recall}")

            # Reset metrics
            self.accuracy.reset()
            self.precision.reset()
            self.recall.reset()
            
            if self.val_loader is not None:
                # Calculate validation loss
                val_loss = self.evaluate(self.val_loader, device, "Validation")

                # Add val loss to history
                self.history['val_loss'].append(val_loss)

                # Plot the loss
                plt.plot(self.history['train_loss'], label='train_loss')
                plt.plot(self.history['val_loss'], label='val_loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Loss over epochs')
                plt.legend()
                os.makedirs('Outputs/Figures', exist_ok=True)
                plt.savefig(f'Outputs/Figures/loss_plot{str(figure_num)}.png')
                plt.clf()

                # Check for early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.current_patience = 0
                else:
                    self.current_patience += 1
                    if self.current_patience >= self.patience:
                        print(f"Early stopping after {epoch + 1} epochs.")
                        break

            ### save checkpoint of model and plot
            ### MODEL
            try:
                os.makedirs('Outputs/trained_models', exist_ok=True)
                model_name = f"{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.train_loader.dataset)}_BATCHSIZE_{self.train_loader.batch_size}.pt"
                # torch.save(self.model.state_dict(), 'Outputs/trained_models/' + model_name)
                self.save("Outputs/trained_models/" + model_name)
                print(f'Checkpoint after epoch {epoch+1} saved successfully')
            except Exception as e:
                print(f"Error saving checkpoint with name: {e}")
                model_name = f"checkpoint_TIME_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
                # torch.save(self.model.state_dict(), 'Outputs/trained_models/' + model_name)
                self.save("Outputs/trained_models/" + model_name)
                print('Unnamed checkpoint saved successfully')

            ### PLOT
            try:
                history = self.history
                # Plot the loss history
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training History')
                plt.legend()
                plt.savefig(f"Outputs/trained_models/{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.train_loader.dataset)}_BATCHSIZE_{self.train_loader.batch_size}.png")
            except:
                print('Error generating plot')

    def evaluate(self, data_loader, device, mode="Test"):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                # Move the data to the GPU
                texts = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['targets'].to(device)

                # Make predictions
                outputs = self.model(texts, attention_mask=attention_mask)
                logits = outputs.logits

                # Calculate metrics
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item()
                val_acc = self.accuracy(logits.softmax(dim=1), labels)
                val_precision = self.precision(logits.softmax(dim=1), labels)
                val_recall = self.recall(logits.softmax(dim=1), labels)

        # Compute metrics
        avg_loss = total_loss / len(data_loader)
        val_acc = self.accuracy.compute()
        val_precision = self.precision.compute()
        val_recall = self.recall.compute()

        if mode == "Validation":
            # Add metrics to history
            self.history['val_acc'].append(val_acc)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)

        # Print metrics
        print(f"{mode} Loss: {avg_loss}")
        print(f"{mode} Accuracy: {val_acc}")
        print(f"{mode} Precision: {val_precision}")
        print(f"{mode} Recall: {val_recall}")

        # Reset metrics
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()

        return avg_loss
    
    def save(self, filepath):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }, filepath)
        print(f"Model and training history saved to {filepath}")

    def load(self, filepath, source):
        if source == "cpu":
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Model and training history loaded from {filepath}")