#%%
import os

import pandas as pd

from utils.preprocessing.preprocessing import csv_to_txtlist, sendto_txt

#%%
# Paths
annomi = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/AnnoMI/AnnoMI-simple.csv"
hope_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/HOPE"
daicwoz_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/DAIC-WOZ/Transcripts"

#%%
# Load and clean annomi data
annomi_df = pd.read_csv(annomi)
annomi_client = annomi_df[annomi_df['interlocutor'] == 'client']

# Remove unintelligible utterances
timestamp = r'\b(\d{1,2}:[0-5][0-9]:[0-5][0-9])\b'
annomi_client.loc[:, 'utterance_text'] = annomi_client['utterance_text'].str.replace(r'\[unintelligible ' + timestamp + r'\]', '<UNK>', regex=True)

# make train/test split
annomi_train = annomi_client.sample(frac=0.8, random_state=42)
annomi_test = annomi_client.drop(annomi_train.index)

# Write to txt
sendto_txt(annomi_train, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_train", "annomi_train", save_txt=True)
sendto_txt(annomi_test, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_test", "annomi_test", save_txt=True)

#%%
# Load and clean daicwoz data
daic_list = os.listdir(daicwoz_dir)
daic = pd.read_csv(os.path.join(daicwoz_dir, daic_list[0]))
for file in daic_list[1:]:
    df = pd.read_csv(os.path.join(daicwoz_dir, file))
    daic = pd.concat([daic, df], axis=0)

daic = daic.reset_index(drop=True)

# Make train/test split
daic_train = daic.sample(frac=0.8, random_state=42)
daic_test = daic.drop(daic_train.index)

# Write to txt
sendto_txt(daic_train, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_train", "daic-woz_train", save_txt=True)
sendto_txt(daic_test, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_test", "daic-woz_test", save_txt=True)

#%%
# Load and clean hope data
hope_list = os.listdir(hope_dir)
hope = pd.read_csv(os.path.join(hope_dir, hope_list[0]), index_col=0)
for file in hope_list[1:]:
    df = pd.read_csv(os.path.join(hope_dir, file), index_col=0)
    hope = pd.concat([hope, df], axis=0)

hope = hope.reset_index(drop=True)
hope_client = hope[hope['Type'] == 'P']

# Make train/test split
hope_train = hope_client.sample(frac=0.8, random_state=42)
hope_test = hope_client.drop(hope_train.index)

# Write to txt
sendto_txt(hope_train, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_train", "hope_train", save_txt=True)
sendto_txt(hope_test, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_test", "hope_test", save_txt=True)

#%%
# Send PACS training data to txt
pacs_train_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_train.csv"

pacs_train = pd.read_csv(pacs_train_path, sep='\t')
sendto_txt(pacs_train, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data", "pacs_mlm_train", save_txt=True)

pacs_test_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_test.csv"
pacs_test = pd.read_csv(pacs_test_path, sep='\t')
sendto_txt(pacs_test, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data", "pacs_mlm_test", save_txt=True)