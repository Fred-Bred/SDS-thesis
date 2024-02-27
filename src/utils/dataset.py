import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class CustomDataset(Dataset):
    def __init__(self, texts, labels,):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label