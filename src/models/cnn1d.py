"""
1D ResNet 分類模型。
將輸入的 F 維特徵視為長度 F 的一維訊號進行卷積操作。
架構參數（n_blocks、channels、kernel_size）完全由外部傳入。
"""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    """單個殘差塊：Conv → BN → GELU → Dropout → Conv → BN，加上殘差連接。"""

    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.conv_block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv_block(x))


class CNN1D(nn.Module):
    """
    Parameters
    ----------
    in_features : int
        輸入維度（被視為訊號長度）。
    n_blocks : int
        殘差塊數量（由 NAS/HPO 決定）。
    channels : int
        卷積通道數（由 NAS/HPO 決定）。
    kernel_size : int
        卷積核大小（由 NAS/HPO 決定，建議奇數）。
    dropout : float
    n_classes : int
    """

    def __init__(
        self,
        in_features: int,
        n_blocks: int,
        channels: int,
        kernel_size: int,
        dropout: float,
        n_classes: int,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[ResBlock1D(channels, kernel_size, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F] → 當作 1D 訊號 [B, 1, F]
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)
