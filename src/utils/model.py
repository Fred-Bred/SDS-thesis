"""
Model building and training utilities.
"""

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn


# Keras BERT model
class BERTKeras(keras.Model):
    def __init__(self, num_classes, hidden_size=768, dropout_prob=0.25, **kwargs):
        super().__init__()

        # BERT model
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.bert.build(input_shape=(None, 512))
        # Set trainable to False to freeze the weights
        self.bert.trainable = False

        # Intermediate layers
        self.fc1 = layers.Dense(hidden_size, activation='relu')
        self.fc2 = layers.Dense(hidden_size, activation='relu')
        self.fc3 = layers.Dense(hidden_size, activation='relu')

        # Dropout layer
        self.dropout = layers.Dropout(dropout_prob)

        # Output layer
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        # BERT model
        bert_output = self.bert(inputs)[0]

        # Intermediate layers
        fc1_output = self.fc1(bert_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)

        # Output layer
        output = self.output_layer(fc3_output)

        return output

# Torch BERT model
class BERTTorch(nn.Module):
    def __init__(self, num_classes, hidden_size=768, dropout_prob=0.25, **kwargs):
        super().__init__()

        # BERT model
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        
        # Freeze bert layers
        for param in self.bert.parameters():
            param.requires_grad = False

        # Get output size of BERT
        bert_output_size = self.bert.pooler.dense.out_features

        # Intermediate layers
        self.fc1 = nn.Linear(bert_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        # BERT model
        bert_output = self.bert(inputs)[0]

        # Intermediate layers
        fc1_output = self.fc1(bert_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)

        # Output layer
        output = self.output_layer(fc3_output)

        return output
    
