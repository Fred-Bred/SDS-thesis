#%%
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
from keras import layers
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import utils.model as model
#%%
# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#%%
# Keras BERT model
keras_model = model.BERTKeras(num_classes=5, hidden_size=768, dropout_prob=0.25)
# %%
