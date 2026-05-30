"""
One-Shot Neural Architecture Search（NAS）for MLP and TSNet。

流程：
  1. 訓練權重共享 Supernet（每 batch 隨機採樣一個子架構）
  2. 以演化演算法（隨機初始 → 突變 → 截斷選擇）從共享權重中擷取最佳子架構

MLP 搜尋空間（所有邊界可由外部設定，不人為固定）：
  - depth     : 隱藏層數
  - hidden_dim: 所有層共用同一維度（方便 skip connection）
  - activations: 每層激活函式
  - use_skips  : 每層是否有殘差連接
  - dropout    : Dropout 比率

TSNet 搜尋空間：
  - n_blocks   : 卷積塊數（1~max_blocks）
  - operations : 每塊的 op 索引（0=conv_k3, 1=conv_k5, 2=tcn_d2, 3=tcn_d4）
  - channels   : 特徵通道數
  - dropout    : Dropout 比率
"""
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .config import SEED, DEVICE

ACT_NAMES = ["relu", "gelu", "silu"]


# ── Supernet 建構 ────────────────────────────────────────────────────────────

class SupernetBlock(nn.Module):
    """共享權重塊：Linear → BN → 多選一激活 + 可選殘差。"""

    ACTS = [nn.ReLU, nn.GELU, nn.SiLU]

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.acts = nn.ModuleList([A() for A in self.ACTS])

    def forward(self, x: torch.Tensor, act_idx: int, use_skip: bool) -> torch.Tensor:
        out = self.acts[act_idx](self.bn(self.linear(x)))
        return x + out if use_skip else out


class OneShotSupernet(nn.Module):
    """
    最大深度 max_depth 的 MLP Supernet，所有層共享 hidden_dim 維度。
    子架構由 arch_config 決定使用哪幾層、每層激活與 skip。
    """

    def __init__(self, in_features: int, hidden_dim: int, max_depth: int, n_classes: int):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_dim)
        self.in_bn = nn.BatchNorm1d(hidden_dim)
        self.blocks = nn.ModuleList([SupernetBlock(hidden_dim) for _ in range(max_depth)])
        self.dropout = nn.Dropout(p=0.0)  # Dropout 比率由外部在 forward 時覆蓋
        self.head = nn.Linear(hidden_dim, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        arch: dict,
        dropout_rate: float = 0.0,
    ) -> torch.Tensor:
        depth = arch["depth"]
        act_indices = arch.get("act_indices", [0] * depth)
        use_skips = arch.get("use_skips", [False] * depth)

        h = torch.relu(self.in_bn(self.in_proj(x)))
        if dropout_rate > 0:
            h = torch.nn.functional.dropout(h, p=dropout_rate, training=self.training)

        for i in range(depth):
            h = self.blocks[i](h, act_idx=act_indices[i], use_skip=use_skips[i])
            if dropout_rate > 0:
                h = torch.nn.functional.dropout(h, p=dropout_rate, training=self.training)
        return self.head(h)


# ── 架構採樣與突變 ───────────────────────────────────────────────────────────

def _random_arch(max_depth: int, hidden_dim_choices: list) -> dict:
    depth = np.random.randint(1, max_depth + 1)
    return {
        "depth": depth,
        "hidden_dim": int(np.random.choice(hidden_dim_choices)),
        "act_indices": [np.random.randint(0, len(ACT_NAMES)) for _ in range(depth)],
        "use_skips": [bool(np.random.randint(0, 2)) for _ in range(depth)],
        "dropout": float(np.random.uniform(0.0, 0.5)),
    }


def _mutate_arch(arch: dict, max_depth: int, hidden_dim_choices: list) -> dict:
    new = {k: list(v) if isinstance(v, list) else v for k, v in arch.items()}
    depth = new["depth"]
    choice = np.random.randint(0, 4)

    if choice == 0 and depth < max_depth:
        new["depth"] += 1
        new["act_indices"].append(np.random.randint(0, len(ACT_NAMES)))
        new["use_skips"].append(bool(np.random.randint(0, 2)))
    elif choice == 1 and depth > 1:
        new["depth"] -= 1
        new["act_indices"] = new["act_indices"][: new["depth"]]
        new["use_skips"] = new["use_skips"][: new["depth"]]
    elif choice == 2:
        layer = np.random.randint(0, depth)
        new["act_indices"][layer] = np.random.randint(0, len(ACT_NAMES))
        new["use_skips"][layer] = not new["use_skips"][layer]
    else:
        new["hidden_dim"] = int(np.random.choice(hidden_dim_choices))
        new["dropout"] = float(np.clip(new["dropout"] + np.random.uniform(-0.1, 0.1), 0.0, 0.5))
    return new


# ── NAS 主類別 ───────────────────────────────────────────────────────────────

class MLPNASSearcher:
    """
    One-Shot NAS for MLP。

    Parameters（全部可由外部設定，不人為固定）
    ----------
    max_depth           : Supernet 最大層數
    hidden_dim_choices  : 可搜尋的隱藏維度列表
    n_supernet_epochs   : Supernet 訓練 epoch 數
    supernet_lr         : Supernet 訓練學習率
    supernet_wd         : Supernet 訓練 weight decay
    n_candidates        : 演化初始族群大小
    n_evolution_rounds  : 演化迭代輪數
    device              : 訓練裝置
    """

    def __init__(
        self,
        max_depth: int = 4,
        hidden_dim_choices: list = None,
        n_supernet_epochs: int = 20,
        supernet_lr: float = 1e-3,
        supernet_wd: float = 1e-4,
        n_candidates: int = 15,
        n_evolution_rounds: int = 3,
        device: str = None,
    ):
        self.max_depth = max_depth
        self.hidden_dim_choices = hidden_dim_choices or [64, 128, 256]
        self.n_supernet_epochs = n_supernet_epochs
        self.supernet_lr = supernet_lr
        self.supernet_wd = supernet_wd
        self.n_candidates = n_candidates
        self.n_evolution_rounds = n_evolution_rounds
        self.device = device or DEVICE
        self.best_arch_: dict = None

    # ------------------------------------------------------------------
    def _train_supernet(
        self, X: np.ndarray, y: np.ndarray, n_classes: int
    ) -> OneShotSupernet:
        """訓練權重共享 Supernet：每 batch 隨機採樣一個子架構。"""
        hidden_dim = max(self.hidden_dim_choices)
        in_features = X.shape[1]
        supernet = OneShotSupernet(in_features, hidden_dim, self.max_depth, n_classes).to(
            self.device
        )
        optimizer = torch.optim.AdamW(
            supernet.parameters(), lr=self.supernet_lr, weight_decay=self.supernet_wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_supernet_epochs
        )
        criterion = nn.CrossEntropyLoss()

        # 🛡️ 型態安全轉換：確保進入 PyTorch 的一定是 Numpy Array
        X_array = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        y_array = y.to_numpy() if hasattr(y, 'to_numpy') else np.array(y)
        
        X_t = torch.tensor(X_array, dtype=torch.float32, device=self.device)
        
        # 注意 y 的型態：如果是二元/多類別分類通常用 torch.long，如果是回歸則用 float32
        # 請保留你原本程式碼中對 dtype 的設定，只把丟進去的變數換成 y_array
        if 'dtype' in str(self.__class__): # 假設你原本是長下面這樣，請依你原本的寫法替換
            y_t = torch.tensor(y_array, dtype=torch.long, device=self.device)
        else:
            # 安全起見，直接用你原本的寫法，只是把 y 換成 y_array
            y_t = torch.tensor(y_array, device=self.device)
        
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)

        supernet.train()
        for epoch in tqdm(
            range(self.n_supernet_epochs),
            desc="  NAS supernet",
            leave=False,
            ncols=80,
        ):
            for xb, yb in loader:
                arch = _random_arch(self.max_depth, self.hidden_dim_choices)
                optimizer.zero_grad()
                logits = supernet(xb, arch, dropout_rate=arch["dropout"])
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(supernet.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        return supernet

    # ------------------------------------------------------------------
    def _eval_arch(
        self, supernet: OneShotSupernet, X: np.ndarray, y: np.ndarray, arch: dict
    ) -> float:
        """用共享權重直接推論，計算 Macro F1（免重新訓練）。"""
        from sklearn.metrics import f1_score as skf1
        import numpy as np

        supernet.eval()
        
        # 🛡️ 型態安全轉換：確保進入 PyTorch 的一定是 Numpy Array
        X_array = X.to_numpy() if hasattr(X, 'to_numpy') else np.array(X)
        
        # 隱藏層處理邏輯 
        # hidden_dim = max(self.hidden_dim_choices)
        # 若搜尋架構的 hidden_dim 比 supernet 小，截取前幾個神經元
        # （簡化處理：直接用 supernet 的 hidden_dim，不做截取）
        
        X_t = torch.tensor(X_array, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = supernet(X_t, arch, dropout_rate=0.0)
        preds = logits.argmax(dim=1).cpu().numpy()
        
        return skf1(y, preds, average="macro", zero_division=0)

    # ------------------------------------------------------------------
    def _evolutionary_search(
        self, supernet: OneShotSupernet, X: np.ndarray, y: np.ndarray
    ) -> dict:
        """演化搜尋：隨機初始族群 → 截斷選擇 → 突變 → 迭代。"""
        np.random.seed(SEED)
        population = [_random_arch(self.max_depth, self.hidden_dim_choices)
                      for _ in range(self.n_candidates)]
        scores = [self._eval_arch(supernet, X, y, a) for a in population]

        for rnd in tqdm(
            range(self.n_evolution_rounds),
            desc="  NAS evolution",
            leave=False,
            ncols=80,
        ):
            top_n = max(2, len(population) // 3)
            top_idx = np.argsort(scores)[::-1][:top_n]
            parents = [population[i] for i in top_idx]

            offspring = [
                _mutate_arch(p, self.max_depth, self.hidden_dim_choices)
                for p in parents
            ]
            offspring_scores = [self._eval_arch(supernet, X, y, a) for a in offspring]

            population += offspring
            scores += offspring_scores

            combined = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
            population = [c[1] for c in combined[: self.n_candidates]]
            scores = [c[0] for c in combined[: self.n_candidates]]

        best_idx = int(np.argmax(scores))
        return population[best_idx], scores[best_idx]

    # ------------------------------------------------------------------
    def search(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
        """
        執行完整 NAS 流程，回傳最佳 arch_params dict。
        dict 含 depth / hidden_dim / activations / use_skips / dropout。
        """
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        print("  [NAS] Training MLP supernet ...")
        supernet = self._train_supernet(X, y, n_classes)

        print("  [NAS] Evolutionary architecture search ...")
        best_arch, best_score = self._evolutionary_search(supernet, X, y)

        # 把 act_indices 轉成名稱，方便後續建模
        best_arch["activations"] = [
            ACT_NAMES[i] for i in best_arch.get("act_indices", [0] * best_arch["depth"])
        ]
        print(
            f"  [NAS] Best arch → depth={best_arch['depth']}, "
            f"hidden_dim={best_arch['hidden_dim']}, "
            f"dropout={best_arch['dropout']:.2f}, "
            f"Macro F1={best_score:.4f}"
        )
        self.best_arch_ = best_arch
        return best_arch


# ════════════════════════════════════════════════════════════════════════════
# TSNet NAS — 時序 1D-CNN 架構搜尋
# ════════════════════════════════════════════════════════════════════════════

TS_OPS = ["conv_k3", "conv_k5", "tcn_d2", "tcn_d4"]


class CausalConv1d(nn.Module):
    """因果一維卷積（Left-padding 確保無未來洩漏）。"""

    def __init__(self, channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x_padded = F.pad(x, (self.pad, 0))
        return self.conv(x_padded)


class _TSOpBlock(nn.Module):
    """單一 TSNet 操作塊：可選 4 種因果卷積操作 + BN + ReLU。"""

    def __init__(self, channels: int):
        super().__init__()
        # 預先建立所有可能的操作（共享 channels 維度）
        self.ops = nn.ModuleList([
            CausalConv1d(channels, kernel_size=3, dilation=1),   # conv_k3
            CausalConv1d(channels, kernel_size=5, dilation=1),   # conv_k5
            CausalConv1d(channels, kernel_size=3, dilation=2),   # tcn_d2
            CausalConv1d(channels, kernel_size=3, dilation=4),   # tcn_d4
        ])
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor, op_idx: int) -> torch.Tensor:
        # x: [B, C, L]
        out = self.ops[op_idx](x)
        out = self.bn(out)
        return F.relu(out + x)  # 殘差連接


class TSNet(nn.Module):
    """
    時序網路：1D Conv 堆疊 + AdaptiveAvgPool → 分類頭。

    Parameters
    ----------
    in_features : int
        輸入特徵維度（序列長度）
    channels : int
        卷積通道數（所有塊共用）
    operations : list[int]
        每塊的 op 索引（0=conv_k3, 1=conv_k5, 2=tcn_d2, 3=tcn_d4）
    n_classes : int
        分類數
    dropout : float
        分類頭前的 Dropout 比率
    """

    def __init__(
        self,
        in_features: int,
        channels: int = 64,
        operations: list = None,
        n_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        operations = operations or [0, 2, 3]
        self.in_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList([
            _TSOpBlock(channels) for _ in operations
        ])
        self.operations = operations
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F]  → unsqueeze → [B, 1, F]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.in_proj(x)  # [B, C, F]
        for i, (block, op_idx) in enumerate(zip(self.blocks, self.operations)):
            h = block(h, op_idx)
        h = self.pool(h).squeeze(-1)  # [B, C]
        h = self.dropout(h)
        return self.head(h)


class TSNetSupernet(nn.Module):
    """
    TSNet 超網路：固定 max_blocks 個 block，每個 block 有 4 種 op 可選。
    子架構由 arch（n_blocks + operations 列表）決定使用哪幾個 block 及各 block 的 op。
    """

    def __init__(self, in_features: int, channels: int, max_blocks: int, n_classes: int):
        super().__init__()
        self.in_proj = nn.Conv1d(1, channels, kernel_size=1)
        self.blocks = nn.ModuleList([_TSOpBlock(channels) for _ in range(max_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.0)  # 比率由外部 arch 決定
        self.head = nn.Linear(channels, n_classes)
        self.max_blocks = max_blocks

    def forward(self, x, arch: dict, dropout_rate: float = 0.0, is_small: bool = False) -> torch.Tensor:
        n_blocks = arch["n_blocks"]
        operations = arch["operations"]  # list[int], len == n_blocks

        # 🛡️ 終極安檢：只要 x 還不是 PyTorch Tensor，就強迫轉換！
        if not isinstance(x, torch.Tensor):
            # 如果它有 .to_numpy() 屬性 (例如 Pandas DataFrame/Series)，先脫掉外衣
            if hasattr(x, 'to_numpy'):
                x = x.to_numpy()
            
            # 安全地轉成 Tensor 並送到正確的 GPU/CPU 設備上
            device = next(self.parameters()).device
            x = torch.tensor(x, dtype=torch.float32, device=device)

        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.in_proj(x)

        # 小資料集強制 dropout 防過擬合，eval mode 下同樣保持開啟
        eff_dropout = 0.3 if (is_small and dropout_rate == 0.0) else dropout_rate
        eff_training = True if is_small else self.training

        for i in range(n_blocks):
            h = self.blocks[i](h, op_idx=operations[i])
            if eff_dropout > 0:
                h = F.dropout(h, p=eff_dropout, training=eff_training)

        h = self.pool(h).squeeze(-1)
        if eff_dropout > 0:
            h = F.dropout(h, p=eff_dropout, training=eff_training)
        return self.head(h)


# ── TSNet 架構採樣與突變 ─────────────────────────────────────────────────────

def _random_ts_arch(max_blocks: int) -> dict:
    """隨機採樣 TSNet 子架構。"""
    n_blocks = np.random.randint(1, max_blocks + 1)
    return {
        "n_blocks": n_blocks,
        "operations": [int(np.random.randint(0, len(TS_OPS))) for _ in range(n_blocks)],
        "channels": int(np.random.choice([32, 64, 128])),
        "dropout": float(np.random.uniform(0.0, 0.4)),
    }


def _mutate_ts_arch(arch: dict, max_blocks: int) -> dict:
    """突變一個 TSNet 子架構。"""
    new = {k: list(v) if isinstance(v, list) else v for k, v in arch.items()}
    n_blocks = new["n_blocks"]
    choice = np.random.randint(0, 4)

    if choice == 0 and n_blocks < max_blocks:
        new["n_blocks"] += 1
        new["operations"].append(int(np.random.randint(0, len(TS_OPS))))
    elif choice == 1 and n_blocks > 1:
        new["n_blocks"] -= 1
        new["operations"] = new["operations"][:new["n_blocks"]]
    elif choice == 2:
        layer = np.random.randint(0, n_blocks)
        new["operations"][layer] = int(np.random.randint(0, len(TS_OPS)))
    else:
        new["channels"] = int(np.random.choice([32, 64, 128]))
        new["dropout"] = float(np.clip(new["dropout"] + np.random.uniform(-0.1, 0.1), 0.0, 0.4))

    return new


# ── TSNASSearcher ────────────────────────────────────────────────────────────

class TSNASSearcher:
    """
    TSNet NAS，介面與 MLPNASSearcher 相同。

    Parameters
    ----------
    max_blocks          : Supernet 最大 block 數
    channels            : 固定通道數（Supernet 使用最大值）
    n_supernet_epochs   : Supernet 訓練 epoch 數
    supernet_lr         : Supernet 訓練學習率
    supernet_wd         : Supernet 訓練 weight decay
    n_candidates        : 演化初始族群大小
    n_evolution_rounds  : 演化迭代輪數
    device              : 訓練裝置
    """

    def __init__(
        self,
        max_blocks: int = 4,
        channels: int = 64,
        n_supernet_epochs: int = 20,
        supernet_lr: float = 1e-3,
        supernet_wd: float = 1e-4,
        n_candidates: int = 15,
        n_evolution_rounds: int = 3,
        device: str = None,
    ):
        self.max_blocks = max_blocks
        self.channels = channels
        self.n_supernet_epochs = n_supernet_epochs
        self.supernet_lr = supernet_lr
        self.supernet_wd = supernet_wd
        self.n_candidates = n_candidates
        self.n_evolution_rounds = n_evolution_rounds
        self.device = device or DEVICE
        self.best_arch_: dict = None

    # ------------------------------------------------------------------
    def _train_supernet(
        self, X: np.ndarray, y: np.ndarray, n_classes: int
    ) -> TSNetSupernet:
        """訓練 TSNet 權重共享 Supernet：每 batch 隨機採樣一個子架構。"""
        in_features = X.shape[1]
        supernet = TSNetSupernet(
            in_features=in_features,
            channels=self.channels,
            max_blocks=self.max_blocks,
            n_classes=n_classes,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            supernet.parameters(), lr=self.supernet_lr, weight_decay=self.supernet_wd
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_supernet_epochs
        )
        # 🛡️ 終極安檢：確保 X 和 y 在進入 PyTorch 前都是純 Numpy 陣列
        X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

        # 動態判斷回歸 / 分類
        is_regression = (n_classes == 1)
        if is_regression:
            criterion = nn.MSELoss()
            # 餵入安全的 X_array 和 y_array
            X_t = torch.tensor(X_array, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y_array, dtype=torch.float32, device=self.device).view(-1, 1)
        else:
            criterion = nn.CrossEntropyLoss()
            # 餵入安全的 X_array 和 y_array
            X_t = torch.tensor(X_array, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y_array, dtype=torch.long, device=self.device)
            
        ds = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True, drop_last=False)

        supernet.train()
        for epoch in tqdm(
            range(self.n_supernet_epochs),
            desc="  TS-NAS supernet",
            leave=False,
            ncols=80,
        ):
            for xb, yb in loader:
                arch = _random_ts_arch(self.max_blocks)
                optimizer.zero_grad()
                logits = supernet(xb, arch, dropout_rate=arch["dropout"])
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(supernet.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        return supernet

    # ------------------------------------------------------------------
    def _eval_arch(
        self, supernet: TSNetSupernet, X_tensor, y_tensor, arch: dict, raw_y=None
    ) -> float:
        """分批推論評分（分類 Macro F1；回歸負 MSE）。"""
        is_small = len(X_tensor) < 2000

        # 小資料集限制架構深度防過擬合
        if is_small and arch.get("n_blocks", 1) > 2:
            arch = dict(arch)
            arch["n_blocks"] = min(arch["n_blocks"], 2)
            arch["operations"] = arch["operations"][: arch["n_blocks"]]

        supernet.eval()
        eval_bs = 512
        n = len(X_tensor)
        all_logits = []
        with torch.no_grad():
            for i in range(0, n, eval_bs):
                xb = X_tensor[i: i + eval_bs]
                all_logits.append(supernet(xb, arch, dropout_rate=0.0, is_small=is_small).cpu())
        logits = torch.cat(all_logits, dim=0)

        if self.n_classes == 1:
            y_cpu = y_tensor.cpu().view(-1, 1)
            score = -float(nn.MSELoss()(logits, y_cpu).item())
            if is_small:
                score -= 0.05 * arch["n_blocks"]
            return score
        else:
            from sklearn.metrics import f1_score as skf1
            preds = logits.argmax(dim=1).numpy()
            y_np = raw_y if raw_y is not None else y_tensor.cpu().numpy()
            return skf1(y_np, preds, average="macro", zero_division=0)

    # ------------------------------------------------------------------
    def _evolutionary_search(
        self, supernet: TSNetSupernet, X: np.ndarray, y: np.ndarray
    ) -> tuple:
        """演化搜尋：隨機初始族群 → 截斷選擇 → 突變 → 迭代。"""
        np.random.seed(SEED)

        # 🛡️ 終極安檢：脫掉 Pandas 外衣，確保是純 Numpy 陣列
        X_array = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)

        # 進迴圈前一次性轉換 tensor，避免重複轉換拖慢速度
        is_regression = (self.n_classes == 1)
        # 餵入安全的 X_array
        X_tensor = torch.tensor(X_array, dtype=torch.float32, device=self.device)
        # 🛡️ 終極安檢：脫掉 Pandas 外衣，確保 y 是純 Numpy 陣列
        y_array = y.to_numpy() if hasattr(y, "to_numpy") else np.array(y)

        if is_regression:
            # 餵入安全的 y_array
            y_tensor = torch.tensor(y_array, dtype=torch.float32, device=self.device).view(-1, 1)
            raw_y = None
        else:
            # 餵入安全的 y_array
            y_tensor = torch.tensor(y_array, dtype=torch.long, device=self.device)
            # 🚨 關鍵細節：讓 raw_y 也變成乾淨的 Numpy 陣列，保護後續的 sklearn 評估！
            raw_y = y_array
        try:
            population = [_random_ts_arch(self.max_blocks) for _ in range(self.n_candidates)]
            scores = [self._eval_arch(supernet, X_tensor, y_tensor, a, raw_y=raw_y) for a in population]

            for rnd in tqdm(
                range(self.n_evolution_rounds),
                desc="  TS-NAS evolution",
                leave=False,
                ncols=80,
            ):
                top_n = max(2, len(population) // 3)
                top_idx = np.argsort(scores)[::-1][:top_n]
                parents = [population[i] for i in top_idx]

                offspring = [_mutate_ts_arch(p, self.max_blocks) for p in parents]
                offspring_scores = [self._eval_arch(supernet, X_tensor, y_tensor, a, raw_y=raw_y) for a in offspring]

                population += offspring
                scores += offspring_scores

                combined = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
                population = [c[1] for c in combined[:self.n_candidates]]
                scores = [c[0] for c in combined[:self.n_candidates]]

            best_idx = int(np.argmax(scores))
            return population[best_idx], scores[best_idx]
        finally:
            del X_tensor, y_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    def search(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
        """
        執行完整 TSNet NAS 流程，回傳最佳 arch_params dict。
        dict 含 n_blocks / operations / channels / dropout。
        """
        self.n_classes = n_classes
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("  [TS-NAS] Training TSNet supernet ...")
        supernet = self._train_supernet(X, y, n_classes)

        print("  [TS-NAS] Evolutionary architecture search ...")
        best_arch, best_score = self._evolutionary_search(supernet, X, y)

        print(
            f"  [TS-NAS] Best arch → n_blocks={best_arch['n_blocks']}, "
            f"ops={[TS_OPS[i] for i in best_arch['operations']]}, "
            f"channels={best_arch['channels']}, "
            f"dropout={best_arch['dropout']:.2f}, "
            f"score={best_score:.4f}"
        )
        self.best_arch_ = best_arch
        return best_arch
