# Description: This file contains the code for the description of the data set.
#%%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocessing.transcript import *
#%% PACS data set description
# PACS folder path
PACS_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Original data UNTOUCHED/PACS_corrected"

# load patient speech turns from all documents in folder
patient_turns = load_patient_turns_from_folder(folder_path=PACS_path)

# Different splits of patient turns
patient_chunks = split_into_chunks(patient_turns, chunk_size=150) # Split into chunks of 150 words
count_filtered = filter_by_word_count(patient_turns, min_word_count=150) # Filter out turns with less than 150 words

all_patient_turns = [item for sublist in patient_turns for item in sublist]

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
all_chunks = load_and_chunk_speech_turns(folder_path=PACS_path)

n_all_chunks = 0
for lst in all_chunks:
    length = len(lst)
    n_all_chunks += length

#%% Plot PACS results
# Bar plot of turns per document
turns_per_doc = [len(lst) for lst in patient_turns]
plt.figure(figsize=(10, 6))
sns.histplot(turns_per_doc, bins=20)
plt.xlabel('Turn Length')
plt.ylabel('Frequency')
plt.title('PACS Distribution of Patient Turns per Document')
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Figures/PACS_Turns_per_document.png")

# Bar plot of patient turn length
plt.figure(figsize=(10, 6))
sns.histplot([len(turn.split()) for turn in all_patient_turns], bins=20, color="green")
plt.title("PACS Distribution of Patient Turn Length")
plt.xlabel("Turn Length")
plt.ylabel("Frequency")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Figures/PACS_Patient_turn_length.png")

#%% Anno-MI description
# Load anno-mi data
annomi = pd.read_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/AnnoMI/AnnoMI-simple.csv")

# Clean text
timestamp = r'\b(\d{1,2}:[0-5][0-9]:[0-5][0-9])\b'
annomi['utterance_text'] = annomi['utterance_text'].str.replace(r'\[unintelligible ' + timestamp + r'\]', '<UNK>', regex=True)

# Filter out client turns
annomi_client = annomi[annomi['interlocutor'] == 'client']

annomi_client_turns = annomi_client['utterance_text'].tolist()
annomi_client_chunks = split_into_chunks(annomi_client_turns, chunk_size=150) # Split into chunks of 150 words
annomi_count_filtered = filter_by_word_count(annomi_client_turns, min_word_count=150) # Filter out turns with less than 150 words

#%% Plot Anno-MI results
# Bar plot of turn length
plt.figure(figsize=(10, 6))
sns.histplot([len(turn.split()) for turn in annomi_client_turns], bins=20, color="orange")
plt.title("Anno-MI Distribution of Client Turn Length")
plt.xlabel("Turn Length")
plt.ylabel("Frequency")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Figures/AnnoMI_Client_turn_length.png")

#%% DAIC-WOZ description
DAIC_WOZ_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/DAIC-WOZ"
daic_woz_files = os.listdir(DAIC_WOZ_path)

# Load patient speech turns from all documents in folder
daic_woz = pd.read_csv(os.path.join(DAIC_WOZ_path, daic_woz_files[0]))
for file in daic_woz_files[1:]:
    df = pd.read_csv(os.path.join(DAIC_WOZ_path, file))
    daic_woz = daic_woz.concat(df)

#%% HOPE description
# HOPE folder path
HOPE_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/HOPE"
hope_files = os.listdir(HOPE_path)


# Load patient speech turns from all documents in folder
hope = pd.read_csv(os.path.join(HOPE_path, hope_files[0]))
for file in hope_files[1:]:
    df = pd.read_csv(os.path.join(HOPE_path, file))
    hope = hope.concat(df)

# Different splits of patient turns
hope_patient = hope[hope["type"] == "P"]
hope_patient_turns = hope_patient["utterance"].tolist()
hope_patient_chunks = split_into_chunks(hope_patient_turns, chunk_size=150) # Split into chunks of 150 words
hope_count_filtered = filter_by_word_count(hope_patient_turns, min_word_count=150) # Filter out turns with less than 150 words

#%% Plot HOPE results
# Bar plot of turn length
plt.figure(figsize=(10, 6))

sns.histplot([len(turn.split()) for turn in hope_patient_turns], bins=20, color="steelblue")
plt.title("HOPE Distribution of Patient Turn Length")
plt.xlabel("Turn Length")
plt.ylabel("Frequency")
plt.savefig("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Figures/HOPE_Patient_turn_length.png")

#%%
# Print results
print("-------")
print("\n***PACS data set description***\n")
print(f"\nNumber of documents loaded: {len(patient_turns)}\n")
print(f"Number of patient turns: {n_turns}\n")
print(f"Average patient turns per document: {n_turns/len(patient_turns)}\n")
print(f"Average patient turn length: {avg_turn_length} words\n")
print(f"Number of patient turns with at least 150 words: {n_filtered}\n")
print(f"Number of arbitrary patient chunks with at least 150 words: {n_patient_chunks}\n")
print(f"Number of combined chunks with at least 250 words: {n_all_chunks}\n")
print("-------")

print("\n***HOPE data set description***\n")
print(f"\nNumber of documents loaded: {len(hope_files)}\n")
print(f"\nNumber of patient turns per document: {len(hope_patient)/len(hope_files)}\n")
print(f"\nNumber of patient turns: {len(hope_patient)}\n")
print(f"Number of patient chunks: {len(hope_patient_chunks)}\n")
print(f"Number of patient turns with at least 150 words: {len(hope_count_filtered)}\n")
print("-------")

print("\n***Anno-MI data set description***\n")
print(f"\nNumber of client turns: {len(annomi_client_turns)}\n")
print(f"Number of client chunks: {len(annomi_client_chunks)}\n")
print(f"Number of client turns with at least 150 words: {len(annomi_count_filtered)}\n")

print("-------")

print("\n***DAIC-WOZ data set description***\n")
print(f"\nNumber of documents loaded: {len(daic_woz_files)}\n")
print(f"\nTotalt number of utterances: {len(daic_woz)}\n")
print(f"\nAverage number of utterances per document: {len(daic_woz)/len(daic_woz_files)}\n")
print("-------")