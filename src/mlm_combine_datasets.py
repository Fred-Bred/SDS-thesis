import os

train_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_train"
test_dir = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/pretraining_test"

train_out = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/mlm_combined_train.txt"
test_out = "/home/unicph.domain/wqs493/ucph/securegroupdir/SAMF-SODAS-PACS/Data/mlm_combined_test.txt"

for file in os.listdir(train_dir):
    with open(os.path.join(train_dir, file), 'r') as f:
        text = f.readlines()
    with open(train_out, 'a') as f:
        for line in text:
            f.write(line + '\n')

for file in os.listdir(test_dir):
    with open(os.path.join(test_dir, file), 'r') as f:
        text = f.readlines()
    with open(test_out, 'a') as f:
        for line in text:
            f.write(line + '\n')