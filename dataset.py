import torch
from torch.utils.data import Dataset

class NFLDataset(Dataset):
    def __init__(self, x, y):
        self.X = torch.from_numpy(x)
        self.Y = torch.from_numpy(y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]