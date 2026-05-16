"""資料切割與 PyTorch Dataset/DataLoader 工具。"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, TimeSeriesSplit

from .config import SEED


def get_folds(y: np.ndarray, n_splits: int = 5, n_repeats: int = 1, random_state: int = SEED) -> list:
    """回傳 5-Fold stratified 切割索引列表。"""
    if n_repeats > 1:
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(np.zeros(len(y)), y))


def get_ts_folds(n_samples: int, n_splits: int = 5) -> list:
    """
    時間序列 Walk-forward 驗證切割。
    使用 TimeSeriesSplit 確保訓練集永遠在驗證集之前（無 Look-ahead Bias）。
    越靠近測試集的 fold（fold index 越大）代表越新的時段。
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    dummy = np.zeros((n_samples, 1))
    return list(tscv.split(dummy))


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
