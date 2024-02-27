"""Copied-in trainer class from previous project - ADJUST"""

# Import libraries
import os
import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
 
#     def fit(self, num_epochs, train_loader, val_loader=None, patience=5, min_delta=0.0001):
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
#                 for texts, labels in self.train_loader:
#                     texts = texts.to(device)
#                     labels = labels.to(device)
#                     self.optimizer.zero_grad()
#                     outputs = self.model(texts)
#                     loss = self.loss_fn(outputs, labels)
#                     loss.backward()
#                     self.optimizer.step()
#                     total_loss += loss.item()

#                     # Update the progress bar
#                     pbar.update(1)
#             avg_loss = total_loss / len(self.train_loader)
#             self.history['train_loss'].append(avg_loss)
#             print(f"Training Loss: {avg_loss}")

#             if self.val_loader is not None:
#                 val_loss = self.evaluate(self.val_loader, "Validation")
#                 self.history['val_loss'].append(val_loss)

#                 # Plot the loss
#                 plt.plot(self.history['train_loss'], label='train_loss')
#                 plt.plot(self.history['val_loss'], label='val_loss')
#                 plt.xlabel('Epochs')
#                 plt.ylabel('Loss')
#                 plt.title('Loss over epochs')
#                 plt.legend()
#                 plt.savefig(f'plots/loss_plot{str(figure_num)}.png')
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
#                 os.makedirs('../trained_models', exist_ok=True)
#                 model_name = f"{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.data_loader.dataset)}_BATCHSIZE_{self.data_loader.batch_size}.pt"
#                 torch.save(self.model.state_dict(), '../trained_models/' + model_name)
#                 print(f'Checkpoint after epoch {epoch+1} saved successfully')
#             except Exception as e:
#                 print(f"Error saving checkpoint with name: {e}")
#                 model_name = f"checkpoint_TIME_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
#                 torch.save(self.model.state_dict(), '../trained_models/' + model_name)
#                 print('Unnamed checkpoint saved successfully')

            
#             try:
#                 ### PLOT
#                 history = self.history
#                 # Plot the loss history
#                 plt.plot(history['train_loss'], label='Train Loss')
#                 plt.plot(history['val_loss'], label='Validation Loss')
#                 plt.xlabel('Epoch')
#                 plt.ylabel('Loss')
#                 plt.title('Training History')
#                 plt.legend()
#                 plt.savefig(f"../trained_models/{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.data_loader.dataset)}_BATCHSIZE_{self.data_loader.batch_size}.png")
#             except:
#                 print('Error generating plot')

#     def evaluate(self, data_loader, mode="Test"):
#         self.model.eval()
#         total_loss = 0.0
#         with torch.no_grad():
#             for texts, labels in data_loader:
#                 # Move the data to the GPU
#                 texts = texts.to(device)
#                 labels = labels.to(device) 

#                 outputs = self.model(texts)
#                 loss = self.loss_fn(outputs, labels)
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
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.train_loader = None
        self.val_loader = None
        self.source = None
        self.history = {'train_loss': [], 'val_loss': [], 'train_loss_batch': [], 'val_loss_batch': []}

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
                    texts = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['targets'].to(device)
                    self.optimizer.zero_grad()
                    outputs = self.model(texts)
                    loss = self.loss_fn(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                    # Update the progress bar
                    pbar.update(1)
            avg_loss = total_loss / len(self.train_loader)
            self.history['train_loss'].append(avg_loss)
            print(f"Training Loss: {avg_loss}")

            if self.val_loader is not None:
                val_loss = self.evaluate(self.val_loader, "Validation")
                self.history['val_loss'].append(val_loss)

                # Plot the loss
                plt.plot(self.history['train_loss'], label='train_loss')
                plt.plot(self.history['val_loss'], label='val_loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.title('Loss over epochs')
                plt.legend()
                plt.savefig(f'plots/loss_plot{str(figure_num)}.png')
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
                os.makedirs('../trained_models', exist_ok=True)
                model_name = f"{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.data_loader.dataset)}_BATCHSIZE_{self.data_loader.batch_size}.pt"
                torch.save(self.model.state_dict(), '../trained_models/' + model_name)
                print(f'Checkpoint after epoch {epoch+1} saved successfully')
            except Exception as e:
                print(f"Error saving checkpoint with name: {e}")
                model_name = f"checkpoint_TIME_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.pt"
                torch.save(self.model.state_dict(), '../trained_models/' + model_name)
                print('Unnamed checkpoint saved successfully')

            
            try:
                ### PLOT
                history = self.history
                # Plot the loss history
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training History')
                plt.legend()
                plt.savefig(f"../trained_models/{self.model_name}_checkpoint_EPOCH_{epoch}_SAMPLES_{len(self.data_loader.dataset)}_BATCHSIZE_{self.data_loader.batch_size}.png")
            except:
                print('Error generating plot')

    def evaluate(self, data_loader, device mode="Test"):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in data_loader:
                # Move the data to the GPU
                texts = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['targets'].to(device) 

                outputs = self.model(texts)
                loss = self.loss_fn(outputs, labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"{mode} Loss: {avg_loss}")
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