"""
神經架構搜尋（Neural Architecture Search, NAS）模組
以「One-Shot NAS」+「權重共享 supernet」為核心，輔以演化／隨機搜尋
從 PyTorch MLP 架構空間中尋找最佳子網路。

搜尋空間：層數深度、每層的活化函式、是否啟用 skip connection、dropout。
"""
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Supernet 基礎建構區塊
# ---------------------------------------------------------------------------

class SupernetBlock(nn.Module):
    """單層 supernet：包含 Linear → BN → 多選一活化 → 可選 skip connection。"""

    def __init__(self, in_features: int, out_features: int, activations: list):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        # 多種活化函式同時保存，由 forward 階段依架構設定挑選
        self.activations = nn.ModuleList(activations)
        # skip connection：若輸入輸出維度不同需用 1×1 投影（這裡用無 bias 的 Linear）
        self.skip = nn.Linear(in_features, out_features, bias=False) if in_features != out_features else nn.Identity()

    def forward(self, x, act_idx=0, use_skip=False):
        # 先 Linear → BN → 指定活化
        out = self.bn(self.linear(x))
        out = self.activations[act_idx](out)
        # 若架構決定使用 skip，加上殘差路徑
        if use_skip:
            out = out + self.skip(x)
        return out


class OneShotSupernet(nn.Module):
    """
    One-Shot Supernet：在一張大網路中包含所有可能的子架構，所有子架構共享權重。
    最大深度 = len(hidden_sizes)；任何子架構都是它的子路徑。
    """

    def __init__(self, in_features: int, hidden_sizes: list, n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.n_classes = n_classes
        self.dropout_rate = dropout

        # 依 hidden_sizes 串出每一層 SupernetBlock
        sizes = [in_features] + hidden_sizes
        self.blocks = nn.ModuleList()
        for i in range(len(hidden_sizes)):
            block = SupernetBlock(
                in_features=sizes[i],
                out_features=sizes[i + 1],
                activations=[nn.ReLU(), nn.Tanh(), nn.GELU()],
            )
            self.blocks.append(block)

        self.dropout = nn.Dropout(dropout)
        # 最終分類/回歸頭
        self.head = nn.Linear(hidden_sizes[-1], n_classes)

    def forward(self, x, arch_config: dict):
        """
        依 arch_config 動態挑選使用哪幾層、每層的活化、是否 skip：
            depth        : 使用前幾層
            act_indices  : 每層使用哪個活化函式（index 對應 [ReLU, Tanh, GELU]）
            use_skips    : 每層是否啟用 skip connection
        """
        depth = arch_config["depth"]
        act_indices = arch_config.get("act_indices", [0] * depth)
        use_skips = arch_config.get("use_skips", [False] * depth)

        h = x
        for i in range(depth):
            h = self.blocks[i](h, act_idx=act_indices[i], use_skip=use_skips[i])
            h = self.dropout(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# 架構搜尋（Random / Mutation）
# ---------------------------------------------------------------------------

def _random_arch(max_depth: int) -> dict:
    """隨機抽樣一個架構設定（深度、各層活化、各層 skip）。"""
    depth = np.random.randint(1, max_depth + 1)
    return {
        "depth": depth,
        "act_indices": [np.random.randint(0, 3) for _ in range(depth)],
        "use_skips": [bool(np.random.randint(0, 2)) for _ in range(depth)],
    }


def _mutate_arch(arch: dict, max_depth: int) -> dict:
    """對既有架構做一次突變：加深一層 / 砍一層 / 隨機改某層的活化與 skip。"""
    # 深複製以避免改到原架構
    new = {k: list(v) if isinstance(v, list) else v for k, v in arch.items()}
    choice = np.random.randint(0, 3)
    depth = new["depth"]
    if choice == 0 and depth < max_depth:
        # 變異 0：加深一層（深度未達上限時）
        new["depth"] += 1
        new["act_indices"].append(np.random.randint(0, 3))
        new["use_skips"].append(bool(np.random.randint(0, 2)))
    elif choice == 1 and depth > 1:
        # 變異 1：砍掉最後一層（深度大於 1 時）
        new["depth"] -= 1
        new["act_indices"] = new["act_indices"][: new["depth"]]
        new["use_skips"] = new["use_skips"][: new["depth"]]
    else:
        # 變異 2：隨機選一層改活化函式並翻轉 skip
        layer = np.random.randint(0, depth)
        new["act_indices"][layer] = np.random.randint(0, 3)
        new["use_skips"][layer] = not new["use_skips"][layer]
    return new


class NeuralArchitectureSearcher:
    """
    One-Shot NAS：先訓練權重共享的 supernet，再用演化搜尋找出最佳子架構。
    """

    # 所有層使用相同隱藏維度，這樣任何深度的子架構都能共用同一個輸出頭
    HIDDEN_SIZES = [128, 128, 128, 128]

    def __init__(self, task="classification", n_supernet_epochs=30,
                 n_arch_candidates=20, n_evolution_rounds=5,
                 dropout=0.2, device=None):
        self.task = task
        self.n_supernet_epochs = n_supernet_epochs      # supernet 訓練 epoch 數
        self.n_arch_candidates = n_arch_candidates      # 演化族群大小
        self.n_evolution_rounds = n_evolution_rounds    # 演化迭代輪數
        self.dropout = dropout
        # 若未指定 device，自動使用 GPU（若可用）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.best_arch_ = None
        self.supernet_ = None
        self.label_encoder_ = None
        self.n_classes_ = None
        self.in_features_ = None

    # ------------------------------------------------------------------
    def _encode_y(self, y: np.ndarray):
        """分類任務做 LabelEncoder；回歸轉 float32。"""
        if self.task == "classification":
            self.label_encoder_ = LabelEncoder()
            return self.label_encoder_.fit_transform(y)
        return y.astype(np.float32)

    def _decode_y(self, y: np.ndarray):
        """將模型輸出的整數類別還原成原始標籤。"""
        if self.task == "classification" and self.label_encoder_:
            return self.label_encoder_.inverse_transform(y)
        return y

    # ------------------------------------------------------------------
    def _train_supernet(self, X: np.ndarray, y_enc: np.ndarray):
        """訓練權重共享的 supernet：每個 batch 隨機抽一個子架構訓練。"""
        n_classes = self.n_classes_
        in_feat = self.in_features_

        supernet = OneShotSupernet(in_feat, self.HIDDEN_SIZES, n_classes, self.dropout).to(self.device)
        # 分類用 CrossEntropy、回歸用 MSE
        criterion = nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()
        optimizer = torch.optim.Adam(supernet.parameters(), lr=1e-3, weight_decay=1e-4)
        # Cosine 學習率退火，平滑收斂
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_supernet_epochs)

        # 將 numpy 轉成 tensor 一次搬到 device 上（資料量適中時較有效率）
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.task == "classification":
            y_t = torch.tensor(y_enc, dtype=torch.long).to(self.device)
        else:
            y_t = torch.tensor(y_enc, dtype=torch.float32).unsqueeze(1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

        supernet.train()
        for epoch in range(self.n_supernet_epochs):
            for xb, yb in loader:
                # 每個 batch 隨機抽一個子架構，逼迫 supernet 各路徑都被訓練
                arch = _random_arch(len(self.HIDDEN_SIZES))
                optimizer.zero_grad()
                logits = supernet(xb, arch)
                if self.task == "regression":
                    loss = criterion(logits, yb)
                else:
                    loss = criterion(logits, yb)
                loss.backward()
                # 梯度裁剪避免 supernet 不同路徑梯度互相衝突造成爆炸
                nn.utils.clip_grad_norm_(supernet.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        return supernet

    # ------------------------------------------------------------------
    def _eval_arch(self, supernet, X: np.ndarray, y_enc: np.ndarray, arch: dict) -> float:
        """評估某個子架構的表現：分類算 accuracy、回歸算 R^2。"""
        supernet.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = supernet(X_t, arch)
        if self.task == "classification":
            preds = logits.argmax(dim=1).cpu().numpy()
            return (preds == y_enc).mean()
        else:
            preds = logits.squeeze(1).cpu().numpy()
            ss_res = ((y_enc - preds) ** 2).sum()
            ss_tot = ((y_enc - y_enc.mean()) ** 2).sum()
            return 1 - ss_res / (ss_tot + 1e-8)

    # ------------------------------------------------------------------
    def _evolutionary_search(self, supernet, X: np.ndarray, y_enc: np.ndarray) -> dict:
        """演化搜尋：取菁英 → 突變 → 截斷選擇，反覆迭代。"""
        max_depth = len(self.HIDDEN_SIZES)
        # 初始族群：完全隨機產生
        population = [_random_arch(max_depth) for _ in range(self.n_arch_candidates)]
        scores = [self._eval_arch(supernet, X, y_enc, a) for a in population]

        for _ in range(self.n_evolution_rounds):
            # 取前 1/3 表現最好的個體作為親代
            top_n = max(2, len(population) // 3)
            top_indices = np.argsort(scores)[::-1][:top_n]
            parents = [population[i] for i in top_indices]

            # 對每個親代各做一次突變產生子代
            offspring = []
            for p in parents:
                offspring.append(_mutate_arch(p, max_depth))
            offspring_scores = [self._eval_arch(supernet, X, y_enc, a) for a in offspring]

            # 親代 + 子代混合
            population += offspring
            scores += offspring_scores

            # 截斷選擇：保留前 n_arch_candidates 個
            combined = sorted(zip(scores, population), key=lambda x: x[0], reverse=True)
            population = [c[1] for c in combined[: self.n_arch_candidates]]
            scores = [c[0] for c in combined[: self.n_arch_candidates]]

        # 回傳最終最佳個體
        best_idx = int(np.argmax(scores))
        return population[best_idx], scores[best_idx]

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralArchitectureSearcher":
        """完整 NAS 流程：編碼 y → 訓練 supernet → 演化搜尋最佳架構。"""
        self.in_features_ = X.shape[1]
        y_enc = self._encode_y(y)

        # 分類取類別數，回歸固定 1
        if self.task == "classification":
            self.n_classes_ = len(np.unique(y_enc))
        else:
            self.n_classes_ = 1

        print(f"  [NAS] Training supernet ({self.n_supernet_epochs} epochs) on {self.device}...")
        self.supernet_ = self._train_supernet(X, y_enc)
        print(f"  [NAS] Evolutionary search ({self.n_arch_candidates} candidates, "
              f"{self.n_evolution_rounds} rounds)...")
        self.best_arch_, best_score = self._evolutionary_search(self.supernet_, X, y_enc)
        print(f"  [NAS] Best arch: depth={self.best_arch_['depth']}, score={best_score:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """以最佳架構推論：分類回傳類別、回歸回傳預測值。"""
        self.supernet_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.supernet_(X_t, self.best_arch_)
        if self.task == "classification":
            preds = logits.argmax(dim=1).cpu().numpy()
            return self._decode_y(preds)
        else:
            return logits.squeeze(1).cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """分類專用：回傳 softmax 後的機率分佈。"""
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        self.supernet_.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.supernet_(X_t, self.best_arch_)
        return torch.softmax(logits, dim=1).cpu().numpy()
