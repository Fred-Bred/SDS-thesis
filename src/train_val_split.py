import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
train_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_train"
val_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/PACS_val"

# Create split
file_list = os.listdir(train_dir)
train_files, val_files = train_test_split(file_list, test_size=0.15, random_state=42)

# Copy files to respective directories
for file in val_files:
    shutil.copy2(os.path.join(train_dir, file), os.path.join(val_dir, file))