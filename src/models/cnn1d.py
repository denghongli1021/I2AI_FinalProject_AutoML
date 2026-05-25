"""
1D CNN / ResNet / TCN 分類模型。
將輸入的 F 維特徵視為長度 F 的一維訊號進行卷積操作。
架構參數（n_blocks、channels、kernel_size）完全由外部傳入。

TCN（Temporal Convolutional Network）使用因果卷積（Causal Conv）+ 擴張卷積（Dilated Conv），
確保預測時間 t 的模型絕對不會看到 t+1 之後的資訊。
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


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.GELU()
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
            
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.act2(out)
        return out


class ResNet1D_18(nn.Module):
    def __init__(self, in_features: int, channels: int = 64, dropout: float = 0.2, n_classes: int = 15):
        super().__init__()
        self.in_channels = channels
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(channels, 2, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(channels * 2, 2, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(channels * 4, 2, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(channels * 8, 2, stride=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels * 8, n_classes)
        )
        
    def _make_layer(self, out_channels, blocks, stride, dropout):
        layers = []
        layers.append(BasicBlock1D(self.in_channels, out_channels, stride, dropout))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_channels, out_channels, 1, dropout))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


# ── TCN（Temporal Convolutional Network）────────────────────────────────────

class _CausalConv1d(nn.Module):
    """因果擴張 1D 卷積：只看當前與過去的資料，不看未來。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        # 左側補零使輸出長度與輸入相同，且不含未來資訊
        self.causal_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            dilation=dilation,
            padding=self.causal_pad,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # 裁去右側多餘的 causal_pad 個時間步（防止未來資訊進入）
        if self.causal_pad > 0:
            out = out[:, :, :-self.causal_pad]
        return out


class _TCNBlock(nn.Module):
    """單個 TCN 殘差塊：兩層因果擴張卷積 + 殘差連接。"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.residual(x))


class TCN(nn.Module):
    """
    Temporal Convolutional Network（TCN）。

    每層的擴張因子以 2 的冪次遞增（dilation = 2^i），
    使感受野（Receptive Field）隨深度指數增長，以少量層數捕捉長距離依賴。
    因果卷積保證零未來洩漏。

    Parameters
    ----------
    in_features : int
        輸入維度（視為訊號長度）。
    n_blocks : int
        TCN 殘差塊數量（dilation = 2^0, 2^1, ..., 2^(n_blocks-1)）。
    channels : int
        所有層的卷積通道數。
    kernel_size : int
        卷積核大小（奇數為宜）。
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
        self.stem = nn.Conv1d(1, channels, kernel_size=1)

        blocks = []
        for i in range(n_blocks):
            dilation = 2 ** i
            blocks.append(_TCNBlock(channels, channels, kernel_size, dilation, dropout))
        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)   # [B, F] → [B, 1, F]
        x = self.stem(x)      # [B, channels, F]
        x = self.blocks(x)    # [B, channels, F]
        return self.head(x)   # [B, n_classes]
