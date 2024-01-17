#%%
# Importing libraries
import numpy as np
from transformers import BertTokenizer
from tensorflow import keras
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

from utils.preprocessing.transcript import load_patient_turns_from_folder, split_into_chunks
from utils.model import BERTTorch

#%%
max_length = 512
folder_path = r'C:\Users\frbre\OneDrive\01 Dokumenter\01 Uni\SDS Thesis\data\test'

#%%
test_turns = load_patient_turns_from_folder(folder_path)

#%%
split_turns = split_into_chunks(test_turns)

#%%
all_turns = [item for sublist in split_turns for item in sublist]

#%%
# Generate fake labels array
length = len(all_turns)

# Generate a fake labels array
fake_labels = np.eye(6)[np.random.choice(6, length)]
#%%
# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', max_length=max_length, padding=True, truncation=True)
#%%
# Tokenize texts and map the tokens to their word IDs.
input_ids = []

for sent in all_turns:
    encoded_text = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                   )
    input_ids.append(encoded_text)

#%%
# # Pad our input tokens
input_ids = keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post", value=0)

# Create attention masks
attention_masks = []

for sent in input_ids:
    
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]
    
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)
#%%
# Make train/val split
train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, fake_labels, 
                                                            random_state=2018, test_size=0.1)
# Performing same steps on the attention masks
train_masks, validation_masks, _, _ = train_test_split(attention_masks, fake_labels,
                                             random_state=2018, test_size=0.1)

#%%
# Convert to tensors
train_inputs = torch.tensor(train_inputs)
validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

# Create dataloaders
batch_size = 32

# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

#%%
torch_model = BERTTorch(num_classes=6, hidden_size=768, dropout_prob=0.25, train_bert=False)

#%%
torch_result = torch_model(train_inputs, attention_mask=train_masks)
print(f"\n\nShape of result: {torch_result.shape}")
print(f"\n\nResult: {torch_result}")