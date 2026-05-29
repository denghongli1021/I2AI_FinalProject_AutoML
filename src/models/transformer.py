"""
1D Signal Transformer 與 PatchTST：
  SignalTransformer — Patch Embedding + CLS token + Multi-Head Attention（適合靜態特徵序列）
  PatchTST          — Channel-Independent Patch Embedding + Mean Pooling（適合時間序列，無未來偏差）

所有架構參數由外部傳入（不人為固定）。
"""
import math
import torch
import torch.nn as nn


class SignalTransformer(nn.Module):
    """
    Parameters
    ----------
    in_features : int
        輸入訊號長度（特徵維度）。
    patch_size : int
        每個 patch 的長度；in_features 必須能被整除（不足時補零）。
    d_model : int
        Transformer 內部維度。
    n_heads : int
        Multi-Head Attention 頭數（必須整除 d_model）。
    depth : int
        Transformer Encoder 層數。
    ff_dim : int
        Feed-Forward 子層的隱藏維度。
    dropout : float
    n_classes : int
    """

    def __init__(
        self,
        in_features: int,
        patch_size: int,
        d_model: int,
        n_heads: int,
        depth: int,
        ff_dim: int,
        dropout: float,
        n_classes: int,
        norm_first: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_patches = math.ceil(in_features / patch_size)
        self.n_patches = n_patches
        self.pad_len = n_patches * patch_size - in_features

        # Patch Embedding
        self.patch_embed = nn.Linear(patch_size, d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 固定 Sinusoidal 位置嵌入（比可學習 PE 更穩健，不易過擬合）
        pe = torch.zeros(n_patches + 1, d_model)
        pos = torch.arange(0, n_patches + 1).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_embed", pe.unsqueeze(0))  # [1, n_patches+1, d_model]

        # norm_first 由 HPO 搜尋：Pre-Norm（True）適合深層/大資料，Post-Norm（False）適合淺層
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # 分類頭：LayerNorm → Linear（encoder dropout 已足夠，移除冗餘 head dropout）
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        if self.pad_len > 0:
            x = torch.cat([x, torch.zeros(B, self.pad_len, device=x.device)], dim=1)

        x = x.reshape(B, self.n_patches, self.patch_size)
        x = self.patch_embed(x)                   # [B, n_patches, d_model]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)            # [B, n_patches+1, d_model]
        x = x + self.pos_embed

        x = self.transformer(x)
        return self.head(x[:, 0])


class PatchTST(nn.Module):
    """
    PatchTST（Patch Time Series Transformer）—— 時序 SOTA。

    與 SignalTransformer 的主要差異：
      1. 無 CLS token，改用對所有 patch 做 Mean Pooling → 減少過擬合
      2. 無絕對位置嵌入（可選學習式位置編碼），更適合可變長度序列
      3. Pre-Norm（norm_first=True）確保梯度穩定

    設計原則：
      - 將輸入序列切割成 patch（區塊），每個 patch 獨立映射到 d_model
      - Transformer 學習 patch 間的長距離依賴
      - 避免 full-attention 的 O(L^2) 開銷（L = sequence length >> n_patches）

    Parameters
    ----------
    in_features : int
        輸入序列長度（特徵維度）。
    patch_size : int
        每個 patch 的長度；不足時補零。
    d_model : int
        Transformer 隱藏維度。
    n_heads : int
        Multi-Head Attention 頭數（需整除 d_model）。
    depth : int
        Transformer Encoder 層數。
    ff_dim : int
        Feed-Forward 子層隱藏維度。
    dropout : float
    n_classes : int
    """

    def __init__(
        self,
        in_features: int,
        patch_size: int,
        d_model: int,
        n_heads: int,
        depth: int,
        ff_dim: int,
        dropout: float,
        n_classes: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        n_patches = math.ceil(in_features / patch_size)
        self.n_patches = n_patches
        self.pad_len = n_patches * patch_size - in_features

        self.patch_embed = nn.Linear(patch_size, d_model)
        # 學習式位置嵌入（可選；對超長序列可設為 0）
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-Norm：梯度更穩定
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Mean Pooling → 分類頭（無 CLS token 過擬合問題）
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # 補零使長度整除 patch_size
        if self.pad_len > 0:
            x = torch.cat(
                [x, torch.zeros(B, self.pad_len, device=x.device)], dim=1
            )

        # 切 patch 並映射到 d_model
        x = x.reshape(B, self.n_patches, self.patch_size)   # [B, n_patches, patch_size]
        x = self.patch_embed(x)                              # [B, n_patches, d_model]
        x = x + self.pos_embed                               # 加入位置嵌入

        # Transformer 編碼
        x = self.transformer(x)   # [B, n_patches, d_model]

        # Mean Pooling：對所有 patch 取均值
        x = self.norm(x.mean(dim=1))   # [B, d_model]
        return self.head(x)            # [B, n_classes]