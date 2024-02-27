import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len

        # Tokenize the texts
        self.encodings = self.tokenizer.batch_encode_plus(
            self.data.text.tolist(),
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        self.targets = torch.tensor(self.data.label.values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = {key: val[index] for key, val in self.encodings.items()}
        item['targets'] = self.targets[index]
        return item
    
def create_data_loader(data, tokenizer, max_len, batch_size):
    ds = CustomDataset(
        dataframe=data,
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )