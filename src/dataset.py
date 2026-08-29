import torch
from torch.utils.data import Dataset, DataLoader

class TimeMachineDataset(Dataset):
    def __init__(self, corpus, seq_len):
        self.corpus = torch.tensor(corpus, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.corpus) - self.seq_len

    def __getitem__(self, i):
        x = self.corpus[i:i+self.seq_len]
        y = self.corpus[i+1:i+self.seq_len+1]
        return x, y

def load_data(batch_size, corpus, seq_len, num_workers=0):
    split = int(0.8 * len(corpus))
    train_dataset = TimeMachineDataset(corpus[:split], seq_len)
    val_dataset = TimeMachineDataset(corpus[split:], seq_len)

    return (
        DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
        )