"""
架構搜尋與堆疊集成（Stacking Ensemble）模組
    L1 Base Layer : 多模型分別產生 OOF（Out-of-Fold）預測，作為 meta-features
    L2 Meta Layer : 以一個 meta-learner 將 OOF 結果再學一次得到最終輸出
"""
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

from .nas import NeuralArchitectureSearcher

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 工具函式：依 (name, params) 建構模型
# ---------------------------------------------------------------------------

def _build_clf(name: str, params: dict):
    """依名稱建立分類器（已套用 random_state、停用冗餘輸出）。"""
    if name == "xgb":
        return xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
    elif name == "lgb":
        return lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {name}")


def _build_reg(name: str, params: dict):
    """依名稱建立回歸器。"""
    if name == "xgb":
        return xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
    elif name == "lgb":
        return lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# NAS 包裝器：對 NeuralArchitectureSearcher 包一層 sklearn 風格的 API，方便 OOF
# ---------------------------------------------------------------------------

class NASWrapper:
    """將 NeuralArchitectureSearcher 包裝成 fit/predict/predict_proba 介面。"""

    def __init__(self, task="classification", n_supernet_epochs=20):
        self.task = task
        self.n_supernet_epochs = n_supernet_epochs
        self._nas = None

    def fit(self, X, y):
        # OOF 階段使用較小的搜尋預算（候選 15、演化 3 輪）以加速
        self._nas = NeuralArchitectureSearcher(
            task=self.task,
            n_supernet_epochs=self.n_supernet_epochs,
            n_arch_candidates=15,
            n_evolution_rounds=3,
        )
        self._nas.fit(X, y)
        return self

    def predict(self, X):
        return self._nas.predict(X)

    def predict_proba(self, X):
        return self._nas.predict_proba(X)


# ---------------------------------------------------------------------------
# L1 Base Layer
# ---------------------------------------------------------------------------

class L1BaseLayer:
    """
    訓練多個 base learner，並以交叉驗證產生 OOF 預測；OOF 預測即為 meta-features。
    Stacking 的核心：用「未見過該樣本的模型」之預測，避免資料洩漏。
    """

    def __init__(self, task="classification", n_folds=5, top_k_configs=2):
        self.task = task
        self.n_folds = n_folds
        self.top_k_configs = top_k_configs      # 每個模型只取 HPO 前 k 組設定
        self.models_ = []                       # [(tag, fitted_model), ...]
        self.label_encoder_ = None

    def fit_predict(self, X: np.ndarray, y: np.ndarray, hpo_configs: dict,
                    use_nas: bool = True) -> np.ndarray:
        """
        對所有 base learner 跑交叉驗證並回傳 OOF meta-features。
        回傳 shape：(n_samples, sum_of_each_model_output_dim)
        """
        # 分類用 Stratified、回歸用普通 KFold
        cv = (
            StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            if self.task == "classification"
            else KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        )

        if self.task == "classification":
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y)
            n_classes = len(np.unique(y_enc))
        else:
            y_enc = y.astype(np.float64)
            n_classes = 1

        all_oofs = []           # 收集每個模型的 OOF 矩陣，最後 hstack
        self.models_ = []

        # ── L1：樹模型（用 HPO 找到的前 top_k_configs 組設定）──
        for model_name in ["xgb", "lgb"]:
            configs = hpo_configs.get(model_name, [])[:self.top_k_configs]
            for rank, (score, params) in enumerate(configs):
                tag = f"{model_name}_{rank}"        # 例如 xgb_0、xgb_1、lgb_0...
                print(f"  [L1] Generating OOF for {tag}...")
                try:
                    if self.task == "classification":
                        model = _build_clf(model_name, params)
                        # 二元與多元分類都使用 predict_proba：保留資訊量給 L2
                        if n_classes == 2:
                            oof = cross_val_predict(model, X, y_enc, cv=cv, method="predict_proba")
                        else:
                            oof = cross_val_predict(model, X, y_enc, cv=cv, method="predict_proba")
                    else:
                        model = _build_reg(model_name, params)
                        # 回歸 OOF 是一維值，補成 (n,1) 方便 hstack
                        oof = cross_val_predict(model, X, y_enc, cv=cv).reshape(-1, 1)

                    # OOF 算完後再用全資料 refit 一次（推論時使用）
                    model.fit(X, y_enc)
                    self.models_.append((tag, model))
                    all_oofs.append(oof)
                except Exception as e:
                    # 個別模型失敗不影響整體，跳過即可
                    print(f"  [L1] {tag} failed: {e}")

        # ── L1：NAS 神經網路（可選）──
        if use_nas:
            try:
                print(f"  [L1] Generating OOF for NAS MLP...")
                nas_oof = self._nas_oof(X, y_enc, cv, n_classes)
                # 與樹模型相同：OOF 算完後 refit 一個全資料模型供推論
                nas_model = NASWrapper(task=self.task, n_supernet_epochs=20)
                nas_model.fit(X, y_enc)
                self.models_.append(("nas_mlp", nas_model))
                all_oofs.append(nas_oof)
            except Exception as e:
                print(f"  [L1] NAS failed: {e}")

        if not all_oofs:
            # 若所有 base learner 都炸掉，向上層拋例外
            raise RuntimeError("All L1 models failed")

        # 將所有模型的 OOF 沿著欄方向接起來，得到 meta-feature 矩陣
        meta_features = np.hstack(all_oofs)
        return meta_features

    def _nas_oof(self, X, y_enc, cv, n_classes):
        """為 NAS 模型手動跑交叉驗證產生 OOF（因為 NAS 無法直接套 cross_val_predict）。"""
        n = len(X)
        if self.task == "classification":
            oof = np.zeros((n, n_classes))
        else:
            oof = np.zeros((n, 1))

        # 對每一折分別訓練一個 NAS，預測屬於該折的驗證樣本
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_enc if self.task == "classification" else None)):
            print(f"    [NAS OOF] fold {fold_idx + 1}...")
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y_enc[train_idx]

            # OOF 階段為了速度，使用較精簡的 NAS 設定
            nas = NeuralArchitectureSearcher(
                task=self.task,
                n_supernet_epochs=15,
                n_arch_candidates=10,
                n_evolution_rounds=2,
            )
            nas.fit(X_tr, y_tr)

            if self.task == "classification":
                oof[val_idx] = nas.predict_proba(X_val)
            else:
                oof[val_idx, 0] = nas.predict(X_val)

        return oof

    def predict(self, X: np.ndarray) -> np.ndarray:
        """推論階段：用所有 refit 後的 base learner 產生測試集 meta-features。"""
        preds = []
        for tag, model in self.models_:
            if self.task == "classification":
                p = model.predict_proba(X)
            else:
                p = model.predict(X).reshape(-1, 1)
            preds.append(p)
        return np.hstack(preds)


# ---------------------------------------------------------------------------
# L2 Ensemble Layer
# ---------------------------------------------------------------------------

class L2EnsembleLayer:
    """
    L2 meta-learner：吃 L1 的 OOF meta-features，輸出最終預測。
    分類用 LogisticRegression、回歸用 Ridge（簡單、不易過擬合）。
    """

    def __init__(self, task="classification"):
        self.task = task
        self.meta_learner_ = None
        self.label_encoder_ = None

    def fit(self, meta_features: np.ndarray, y: np.ndarray):
        if self.task == "classification":
            # L2 自己再做一次 LabelEncoder（與 L1 獨立保存）
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y)
            self.meta_learner_ = LogisticRegression(
                C=1.0, max_iter=1000, random_state=42, multi_class="auto"
            )
            self.meta_learner_.fit(meta_features, y_enc)
        else:
            self.meta_learner_ = Ridge(alpha=1.0)
            self.meta_learner_.fit(meta_features, y.astype(np.float64))
        return self

    def predict(self, meta_features: np.ndarray) -> np.ndarray:
        preds = self.meta_learner_.predict(meta_features)
        # 分類需把整數 index 還原成原始類別
        if self.task == "classification" and self.label_encoder_:
            return self.label_encoder_.inverse_transform(preds.astype(int))
        return preds

    def predict_proba(self, meta_features: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        return self.meta_learner_.predict_proba(meta_features)


# ---------------------------------------------------------------------------
# 完整堆疊集成（L1 + L2 對外統一介面）
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """將 L1 Base Layer 與 L2 Meta Layer 合而為一，提供統一的 fit/predict 介面。"""

    def __init__(self, task="classification", n_folds=5, top_k_configs=2, use_nas=True):
        self.task = task
        self.n_folds = n_folds
        self.top_k_configs = top_k_configs
        self.use_nas = use_nas
        self.l1_ = L1BaseLayer(task=task, n_folds=n_folds, top_k_configs=top_k_configs)
        self.l2_ = L2EnsembleLayer(task=task)

    def fit(self, X: np.ndarray, y: np.ndarray, hpo_configs: dict):
        """先訓練 L1 並產生 OOF meta-features，再用 meta-features 訓練 L2。"""
        print("  [Ensemble] Training L1 Base Layer...")
        meta_train = self.l1_.fit_predict(X, y, hpo_configs, use_nas=self.use_nas)
        print(f"  [Ensemble] Meta-feature shape: {meta_train.shape}")
        print("  [Ensemble] Training L2 Meta-Learner...")
        self.l2_.fit(meta_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 先讓所有 L1 模型對測試集產生 meta-features，再交給 L2 輸出最終預測
        meta_test = self.l1_.predict(X)
        return self.l2_.predict(meta_test)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_test = self.l1_.predict(X)
        return self.l2_.predict_proba(meta_test)
