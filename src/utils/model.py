"""
Model building and training utilities.
"""
# Imports
import os

import numpy as np

from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F

# Torch BERT model
class BERTTorch(nn.Module):
    def __init__(self, num_classes, hidden_size=768, dropout_prob=0.25, train_bert=False, **kwargs):
        super().__init__()

        # BERT model
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-base-multilingual-cased',
            num_labels=1,
            output_attentions=True,
            output_hidden_states=True,)

        # Get output size of last layer before removing it
        in_features = self.bert.classifier.in_features

        # Replace last layer with identity function
        self.bert.classifier = nn.Identity()

        # Remove last bert classifier layer
        # self.bert = nn.Sequential(*list(self.bert.children())[:-1])

        if not train_bert:
            # Freeze bert layers
            for param in self.bert.parameters():
                param.requires_grad = False 

        # Intermediate layers
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, attention_mask=None):
        # BERT model
        bert_output = self.bert(inputs, attention_mask=attention_mask)[0]

        # Intermediate layers
        fc1_output = self.fc1(bert_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)

        # Output layer
        output = F.softmax(self.output_layer(fc3_output))

        return output
    

# RoBERTa model
class RoBERTaTorch(nn.Module):
    def __init__(self, num_classes, hidden_size=768, dropout_prob=0.25, train_bert=False, **kwargs):
        super().__init__()

        # BERT model
        self.bert = RobertaForSequenceClassification.from_pretrained(
            'xlm-r-base',
            num_labels=1,
            output_attentions=True,
            output_hidden_states=True,)

        # Get output size of last layer before removing it
        in_features = self.bert.classifier.in_features

        # Replace last layer with identity function
        self.bert.classifier = nn.Identity()

        # Remove last bert classifier layer
        # self.bert = nn.Sequential(*list(self.bert.children())[:-1])

        if not train_bert:
            # Freeze bert layers
            for param in self.bert.parameters():
                param.requires_grad = False 

        # Intermediate layers
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, attention_mask=None):
        # BERT model
        bert_output = self.bert(inputs, attention_mask=attention_mask)[0]

        # Intermediate layers
        fc1_output = self.fc1(bert_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)

        # Output layer
        output = F.softmax(self.output_layer(fc3_output))

        return output
    
# Generic torch pre-trained model
class TorchBaseModel(nn.Module):
    def __init__(self, base_model, num_classes, hidden_size=768, dropout_prob=0.25, train_base=False, **kwargs):
        super().__init__()

        # base model
        self.base = base_model

        # Get output size of last layer before removing it
        in_features = self.base.classifier.in_features

        # Replace last layer with identity function
        self.base.classifier = nn.Identity()

        # Remove last base classifier layer
        # self.base = nn.Sequential(*list(self.base.children())[:-1])

        if not train_base:
            # Freeze bert layers
            for param in self.base.parameters():
                param.requires_grad = False 

        # Intermediate layers
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs, attention_mask=None):
        # base model
        base_output = self.base(inputs, attention_mask=attention_mask)[0]

        # Intermediate layers
        fc1_output = self.fc1(base_output)
        fc2_output = self.fc2(fc1_output)
        fc3_output = self.fc3(fc2_output)

        # Output layer
        output = F.softmax(self.output_layer(fc3_output))

        return output