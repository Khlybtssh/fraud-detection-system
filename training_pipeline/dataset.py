import torch
from torch.utils.data import Dataset

class FraudDataset(Dataset):
    def __init__(self, X, y):
        # Handle X as a numpy array, which it will be after scaling
        self.X = torch.tensor(X, dtype=torch.float32)
        # Handle y as a Series or DataFrame which it is after smote
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
