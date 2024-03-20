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
annomi_client['utterance_text'] = annomi_client['utterance_text'].str.replace(r'\[unintelligible ' + timestamp + r'\]', '<UNK>', regex=True)

# Write to txt
sendto_txt(annomi_client, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data", "annomi", save_txt=True)

#%%
# Load and clean daicwoz data
daic_list = os.listdir(daicwoz_dir)
daic = pd.read_csv(os.path.join(daicwoz_dir, daic_list[0]))
for file in daic_list[1:]:
    df = pd.read_csv(os.path.join(daicwoz_dir, file))
    daic = pd.concat([daic, df], axis=0)

daic = daic.reset_index(drop=True)

# Write to txt
sendto_txt(daic, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data", "daic-woz", save_txt=True)

#%%
# Load and clean hope data
hope_list = os.listdir(hope_dir)
hope = pd.read_csv(os.path.join(hope_dir, hope_list[0]), index_col=0)
for file in hope_list[1:]:
    df = pd.read_csv(os.path.join(hope_dir, file), index_col=0)
    hope = pd.concat([hope, df], axis=0)

hope = hope.reset_index(drop=True)
hope_client = hope[hope['Type'] == 'P']

# Write to txt
sendto_txt(hope_client, "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data", "hope", save_txt=True)
