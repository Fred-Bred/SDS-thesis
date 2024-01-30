# Description: This file contains the code for the description of the data set.
#%%
# Imports
import pandas as pd
import numpy as np

from utils.preprocessing.transcript_linux import *
#%%
# Load data
folder_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Session PACS validation study"
patient_turns = load_patient_turns_from_folder(folder_path=folder_path) # load patient speech turns from all documents in folder

# Split into chunks of 150 words
chunks = split_into_chunks(patient_turns, chunk_size=150)

n_turns = 0
for lst in patient_turns:
    length = len(lst)
    n_turns += length

n_chunks = 0
for lst in chunks:
    length = len(lst)
    n_chunks += length

print(f"\nNumber of documents loaded: {len(patient_turns)}\n")
print(f"Number of patient turns: {n_turns}\n")
print(f"Number of chunks over 150 words: {n_chunks}\n")