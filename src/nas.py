"""
One-Shot Neural Architecture Search（NAS）for MLP。

流程：
  1. 訓練權重共享 Supernet（每 batch 隨機採樣一個子架構）
  2. 以演化演算法（隨機初始 → 突變 → 截斷選擇）從共享權重中擷取最佳子架構

搜尋空間（所有邊界可由外部設定，不人為固定）：
  - depth     : 隱藏層數
  - hidden_dim: 所有層共用同一維度（方便 skip connection）
  - activations: 每層激活函式
  - use_skips  : 每層是否有殘差連接
  - dropout    : Dropout 比率
"""
import numpy as np
import torch
import torch.nn as nn
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
        max_depth: int = 6,
        hidden_dim_choices: list = None,
        n_supernet_epochs: int = 30,
        supernet_lr: float = 1e-3,
        supernet_wd: float = 1e-4,
        n_candidates: int = 30,
        n_evolution_rounds: int = 5,
        device: str = None,
    ):
        self.max_depth = max_depth
        self.hidden_dim_choices = hidden_dim_choices or [64, 128, 256, 512]
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

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)
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

        supernet.eval()
        hidden_dim = max(self.hidden_dim_choices)
        # 若搜尋架構的 hidden_dim 比 supernet 小，截取前幾個神經元
        # （簡化處理：直接用 supernet 的 hidden_dim，不做截取）
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
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
