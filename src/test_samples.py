from utils.preprocessing.transcript import load_data_with_labels

import pandas as pd

labels_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_labels.xlsx"

train_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_train"
train_data = load_data_with_labels(labels_path, train_path)

val_path = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_val"
val_data = load_data_with_labels(labels_path, val_path)

train_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/train_data.csv", index=False, sep="\t")
val_data.to_csv("/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/val_data.csv", index=False, sep="\t")