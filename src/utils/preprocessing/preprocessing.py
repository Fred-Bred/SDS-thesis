import torch
from torch import nn
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

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

# if __name__ == "__main__":
#     tokenizer = AutoTokenizer.from_pretrained("roberta-base")

#     def preprocess_mlm(samples):
#         return tokenizer([" ".join(sample) for sample in samples["text"]])


def group_texts(examples, block_size):
    """
    Concatenate all texts and return a dictionary with keys as the input keys and values as lists of lists of tokens.
    
    Args:
        examples: dict, containing input keys as keys and lists of tokens as values.
        block_size: int, the size of the blocks to split the concatenated examples into.
    """
    # Concatenate all texts.
    concatenated_examples = {k: [item for sublist in examples[k] for item in sublist] for k in examples.keys()}
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

def csv_to_txtlist(input_dir):
    texts = []
    for file in os.listdir(input_dir):
        df = pd.read_csv(os.path.join(input_dir, file))
        text = df['text'].tolist()
        texts.extend(text)
    return texts

def sendto_txt(input, output_dir, dataset_name, input_dir=None, save_txt=False):
    """
    Send the cleaned dataset to a txt file.

    Args:
        input: list or pandas dataframe, containing the cleaned dataset. Mutually exclusive with input_dir.
        input_dir: str, the input directory. Mutually exclusive with input.
        output_dir: str, the output directory.
        dataset_name: str, the name of the dataset.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if input_dir:
        text = csv_to_txtlist(input_dir)

    if input is not None:
        if isinstance(input, pd.DataFrame):
            try:
                text = input['text'].tolist()
            except KeyError:
                try:
                    text = input['Text'].tolist()
                except KeyError:
                    try:
                        text = input['utterance_text'].tolist()
                    except KeyError:
                        text = input['Utterance'].tolist()
        
        elif isinstance(input, list):
            text = [str(turn) for turn in input]

    if save_txt:
        with open(os.path.join(output_dir, f"{dataset_name}.txt"), 'w') as f:
            for turn in text:
                f.write(str(turn) + '\n')
        print(f"{dataset_name} saved to {output_dir}/{dataset_name}.txt")
    else:
        return text