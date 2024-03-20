import torch
from torch import nn
import numpy as np

import os

def pad_tensors(input_ids, max_length):
    """Pad input_ids tensors to max_length and truncate if needed"""
    # Convert input_ids elements to PyTorch tensor if needed
    input_ids = [tensor.to('cpu') if isinstance(tensor, np.ndarray) else tensor for tensor in input_ids]
    
    # Create a packed sequence
    packed_sequence = nn.utils.rnn.pack_padded_sequence(input_ids, [len(tensor) for tensor in input_ids], batch_first=True)

    # Pad sequences
    padded_sequence, _ = nn.utils.rnn.pad_packed_sequence(packed_sequence, total_length=max_length, batch_first=True, padding_value=0)

    return padded_sequence

def preprocess_mlm(samples, tokenizer):
    return tokenizer([" ".join(sample) for sample in samples["text"]])


def group_texts(examples):
    """
    Concatenate all texts and return a dictionary with keys as the input keys and values as lists of lists of tokens.
    
    Args:
        examples: dict, containing input keys as keys and lists of tokens as values.
        """
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
   
    total_length = (total_length // block_size) * block_size
   
    # Add padding to make total_length a multiple of block_size
    remainder = total_length % block_size
    if remainder > 0:
        padding = [tokenizer.eos_token_id] * (block_size - remainder) # if this doesn't work replace eos_token_id with pad_token_id
        concatenated_examples = {k: examples[k] + padding for k in examples.keys()}
    
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

# def sendto_txt(input, output_dir):
#     for file in os.listdir(output_dir):
