# Description: This file contains the code for the description of the data set.
#%%
# Imports
import pandas as pd
import numpy as np

from utils.preprocessing.transcript import *
#%%
# Folder path
folder_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/PACS_docx"

# load patient speech turns from all documents in folder
patient_turns = load_patient_turns_from_folder(folder_path=folder_path)

# Different splits of patient turns
patient_chunks = split_into_chunks(patient_turns, chunk_size=150) # Split into chunks of 150 words
count_filtered = filter_by_word_count(patient_turns, min_word_count=150) # Filter out turns with less than 150 words

n_turns = 0
for lst in patient_turns:
    length = len(lst)
    n_turns += length

n_filtered = 0
for lst in count_filtered:
    length = len(lst)
    n_filtered += length

n_patient_chunks = 0
for lst in patient_chunks:
    length = len(lst)
    n_patient_chunks += length

avg_turn_length = average_word_count(patient_turns)

# Load and chunk (250+ words) all turns from all documents in folder
all_chunks = load_and_chunk_speech_turns(folder_path=folder_path)

n_all_chunks = 0
for lst in all_chunks:
    length = len(lst)
    n_all_chunks += length

# Print results
print(f"\nNumber of documents loaded: {len(patient_turns)}\n")
print(f"Number of patient turns: {n_turns}\n")
print(f"Average patient turn length: {avg_turn_length}\n")
print(f"Number of arbitrary patient chunks with at least 150 words: {n_patient_chunks}\n")
print(f"Number of patient turns with at least 150 words: {n_filtered}\n")
print(f"Number of combined chunks with at least 250 words: {n_all_chunks}\n")