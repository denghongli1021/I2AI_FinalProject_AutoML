"""資料切割與 PyTorch Dataset/DataLoader 工具。"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from .config import SEED, N_SPLITS


def get_folds(y: np.ndarray) -> list:
    """回傳 5-Fold stratified 切割索引列表。"""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(y)), y))


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def make_loader(
    X: np.ndarray,
    y: np.ndarray = None,
    batch_size: int = 128,
    shuffle: bool = False,
) -> DataLoader:
    ds = TabularDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=shuffle)
