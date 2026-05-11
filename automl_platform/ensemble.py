"""
架構搜尋與堆疊集成（Stacking Ensemble）模組
    L1 Base Layer : 多模型分別產生 OOF（Out-of-Fold）預測，作為 meta-features
    L2 Meta Layer : 多策略 blending（LR / 加權平均 / 幾何平均 / 單一最佳），用 OOF micro F1 自動選最佳
"""
import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb

from .nas import NeuralArchitectureSearcher

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 工具函式：依 (name, params) 建構模型
# ---------------------------------------------------------------------------

def _build_clf(name: str, params: dict, random_state: int = 42):
    if name == "xgb":
        return xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )
    elif name == "lgb":
        return lgb.LGBMClassifier(**params, random_state=random_state, n_jobs=-1, verbose=-1)
    elif name == "et":
        return ExtraTreesClassifier(**params, random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def _build_reg(name: str, params: dict, random_state: int = 42):
    if name == "xgb":
        return xgb.XGBRegressor(**params, random_state=random_state, n_jobs=-1, verbosity=0)
    elif name == "lgb":
        return lgb.LGBMRegressor(**params, random_state=random_state, n_jobs=-1, verbose=-1)
    elif name == "et":
        return ExtraTreesRegressor(**params, random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


# ---------------------------------------------------------------------------
# NAS 包裝器：對 NeuralArchitectureSearcher 包一層 sklearn 風格的 API
# ---------------------------------------------------------------------------

class NASWrapper:
    """將 NeuralArchitectureSearcher 包裝成 fit/predict/predict_proba 介面。"""

    def __init__(self, task="classification", n_supernet_epochs=20,
                 n_arch_candidates=15, n_evolution_rounds=3, seed=42):
        self.task = task
        self.n_supernet_epochs = n_supernet_epochs
        self.n_arch_candidates = n_arch_candidates
        self.n_evolution_rounds = n_evolution_rounds
        self.seed = seed
        self._nas = None

    def fit(self, X, y):
        # 設定隨機種子使每個 NASWrapper 實例的 supernet 訓練、演化搜尋路徑不同
        import torch
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self._nas = NeuralArchitectureSearcher(
            task=self.task,
            n_supernet_epochs=self.n_supernet_epochs,
            n_arch_candidates=self.n_arch_candidates,
            n_evolution_rounds=self.n_evolution_rounds,
        )
        self._nas.fit(X, y)
        return self

    def predict(self, X):
        return self._nas.predict(X)

    def predict_proba(self, X):
        return self._nas.predict_proba(X)


# ---------------------------------------------------------------------------
# 標準化包裝（給 KNN 使用）
# ---------------------------------------------------------------------------

class _ScaledKNN:
    """KNN + 內建 StandardScaler，避免外層 pipeline 對 KNN 的尺度敏感性。"""

    def __init__(self, task="classification", n_neighbors=15, weights="distance"):
        self.task = task
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.scaler_ = None
        self.model_ = None

    def fit(self, X, y):
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        if self.task == "classification":
            self.model_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors, weights=self.weights, n_jobs=-1,
            )
        else:
            self.model_ = KNeighborsRegressor(
                n_neighbors=self.n_neighbors, weights=self.weights, n_jobs=-1,
            )
        self.model_.fit(Xs, y)
        return self

    def predict(self, X):
        return self.model_.predict(self.scaler_.transform(X))

    def predict_proba(self, X):
        return self.model_.predict_proba(self.scaler_.transform(X))


# ---------------------------------------------------------------------------
# L1 Base Layer
# ---------------------------------------------------------------------------

class L1BaseLayer:
    """
    訓練多個 base learner，並以交叉驗證產生 OOF 預測；OOF 預測即為 meta-features。
    本版額外保留：
        - 各模型的 OOF 預測（依模型分塊保存，供 L2 多策略 blending 使用）
        - 每折的 fitted model（fold-bagging 用，test 時對 fold-models predict_proba 取平均）
        - 每個模型的 OOF micro F1 / accuracy（供 L2 加權使用）
    """

    def __init__(self, task="classification", n_folds=5, top_k_configs=2,
                 n_nas_seeds=1, use_extratrees=True, use_knn=True):
        self.task = task
        self.n_folds = n_folds
        self.top_k_configs = top_k_configs
        self.n_nas_seeds = n_nas_seeds         # NAS 多 seed 集成數
        self.use_extratrees = use_extratrees
        self.use_knn = use_knn

        # 訓練後填入：
        self.model_tags_ = []                  # ["xgb_0", "lgb_0", ..., "nas_seed42", "et", "knn15"]
        self.fold_models_ = {}                 # {tag: [m_fold1, m_fold2, ...]}（NAS 為 [m_full]）
        self.full_models_ = {}                 # {tag: m_full}（refit-on-all 模型）
        self.oof_per_model_ = {}               # {tag: oof_proba (n, n_classes) 或 (n, 1)}
        self.oof_score_per_model_ = {}         # {tag: OOF micro F1 / R²}
        self.label_encoder_ = None
        self.classes_ = None                   # 分類任務的類別集合（int 編碼）

    # ------------------------------------------------------------------
    def _build_classification_meta(self, oof_per_model):
        """把每個模型的 OOF 沿欄方向 concat，作為 LR meta-learner 的輸入。"""
        return np.hstack([oof_per_model[tag] for tag in self.model_tags_])

    def _model_oof_metric(self, oof, y_enc):
        """計算單一模型的 OOF 表現分數（分類用 micro F1，回歸用 R²）。"""
        if self.task == "classification":
            preds = oof.argmax(axis=1) if oof.ndim == 2 and oof.shape[1] > 1 else oof.ravel()
            return f1_score(y_enc, preds.astype(int), average="micro")
        # 回歸：R²（越高越好）
        preds = oof.ravel()
        ss_res = ((y_enc - preds) ** 2).sum()
        ss_tot = ((y_enc - y_enc.mean()) ** 2).sum()
        return float(1 - ss_res / (ss_tot + 1e-8))

    # ------------------------------------------------------------------
    def fit_predict(self, X: np.ndarray, y: np.ndarray, hpo_configs: dict,
                    use_nas: bool = True) -> np.ndarray:
        """
        對所有 base learner 跑 cross_val_predict 取 OOF，並在每折保留模型供 fold-bagging。
        回傳 (n_samples, total_meta_dim) 的 OOF meta-feature 矩陣。
        """
        cv = (
            StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            if self.task == "classification"
            else KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        )

        if self.task == "classification":
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y)
            self.classes_ = np.unique(y_enc)
            n_classes = len(self.classes_)
        else:
            y_enc = y.astype(np.float64)
            n_classes = 1

        self.model_tags_ = []
        self.fold_models_ = {}
        self.full_models_ = {}
        self.oof_per_model_ = {}
        self.oof_score_per_model_ = {}

        # ── L1：樹模型（XGB / LGB），HPO top-k 組設定 ──
        for model_name in ["xgb", "lgb"]:
            configs = hpo_configs.get(model_name, [])[: self.top_k_configs]
            for rank, (_score, params) in enumerate(configs):
                tag = f"{model_name}_{rank}"
                print(f"  [L1] OOF for {tag}...")
                try:
                    oof, fold_ms, full_m = self._oof_with_fold_models(
                        model_name, params, X, y_enc, cv, n_classes
                    )
                    self._register(tag, oof, fold_ms, full_m, y_enc)
                except Exception as e:
                    print(f"  [L1] {tag} failed: {e}")

        # ── L1：ExtraTrees（固定設定，無需 HPO） ──
        if self.use_extratrees:
            tag = "et"
            print(f"  [L1] OOF for {tag} (ExtraTrees)...")
            try:
                et_params = {"n_estimators": 400, "max_depth": None, "max_features": "sqrt"}
                oof, fold_ms, full_m = self._oof_with_fold_models(
                    "et", et_params, X, y_enc, cv, n_classes
                )
                self._register(tag, oof, fold_ms, full_m, y_enc)
            except Exception as e:
                print(f"  [L1] {tag} failed: {e}")

        # ── L1：KNN（標準化後，distance-weighted） ──
        if self.use_knn and self.task == "classification":
            for k in [15, 50]:
                tag = f"knn{k}"
                print(f"  [L1] OOF for {tag}...")
                try:
                    oof, fold_ms, full_m = self._oof_knn(k, X, y_enc, cv, n_classes)
                    self._register(tag, oof, fold_ms, full_m, y_enc)
                except Exception as e:
                    print(f"  [L1] {tag} failed: {e}")

        # ── L1：NAS（多 seed） ──
        if use_nas:
            seeds = [42, 7, 123][: max(1, self.n_nas_seeds)]
            for seed in seeds:
                tag = f"nas_seed{seed}"
                print(f"  [L1] OOF for {tag}...")
                try:
                    oof = self._nas_oof(X, y_enc, cv, n_classes, seed=seed)
                    full_m = NASWrapper(
                        task=self.task,
                        n_supernet_epochs=20, n_arch_candidates=15, n_evolution_rounds=3,
                        seed=seed,
                    )
                    full_m.fit(X, y_enc)
                    self._register(tag, oof, [full_m], full_m, y_enc)
                except Exception as e:
                    print(f"  [L1] {tag} failed: {e}")

        if not self.model_tags_:
            raise RuntimeError("All L1 models failed")

        # 印出每個 base 模型的 OOF 分數，方便診斷
        print("  [L1] OOF scores per base model:")
        for tag in self.model_tags_:
            print(f"        {tag:18s}: {self.oof_score_per_model_[tag]:.4f}")

        return self._build_classification_meta(self.oof_per_model_)

    # ------------------------------------------------------------------
    def _register(self, tag, oof, fold_ms, full_m, y_enc):
        """登記一個 base learner 的 OOF / fold models / 全資料模型 / OOF 分數。"""
        self.model_tags_.append(tag)
        self.oof_per_model_[tag] = oof
        self.fold_models_[tag] = fold_ms
        self.full_models_[tag] = full_m
        self.oof_score_per_model_[tag] = self._model_oof_metric(oof, y_enc)

    # ------------------------------------------------------------------
    def _oof_with_fold_models(self, model_name, params, X, y_enc, cv, n_classes):
        """
        對樹模型 / ExtraTrees 做手動 fold loop，產生 OOF 並保留 fold-models。
        回傳: (oof, fold_models, full_model)
        """
        n = len(X)
        if self.task == "classification":
            oof = np.zeros((n, n_classes), dtype=np.float32)
        else:
            oof = np.zeros((n, 1), dtype=np.float32)

        fold_models = []
        for fold_idx, (tr_idx, va_idx) in enumerate(
            cv.split(X, y_enc if self.task == "classification" else None)
        ):
            X_tr, y_tr = X[tr_idx], y_enc[tr_idx]
            X_va = X[va_idx]
            if self.task == "classification":
                m = _build_clf(model_name, params)
                m.fit(X_tr, y_tr)
                # 對齊類別欄：訓練集若缺某類，自動補 0 機率欄
                p_va = m.predict_proba(X_va)
                oof[va_idx] = self._align_proba_columns(p_va, m, n_classes)
            else:
                m = _build_reg(model_name, params)
                m.fit(X_tr, y_tr)
                oof[va_idx, 0] = m.predict(X_va)
            fold_models.append(m)

        # refit on full data 供推論
        if self.task == "classification":
            full = _build_clf(model_name, params)
        else:
            full = _build_reg(model_name, params)
        full.fit(X, y_enc)
        return oof, fold_models, full

    # ------------------------------------------------------------------
    def _oof_knn(self, k, X, y_enc, cv, n_classes):
        """KNN 的 OOF + fold models（KNN 內含 StandardScaler）。"""
        n = len(X)
        oof = np.zeros((n, n_classes), dtype=np.float32)
        fold_models = []
        for tr_idx, va_idx in cv.split(X, y_enc):
            m = _ScaledKNN(task="classification", n_neighbors=k, weights="distance")
            m.fit(X[tr_idx], y_enc[tr_idx])
            p_va = m.predict_proba(X[va_idx])
            oof[va_idx] = self._align_proba_columns_knn(p_va, m, n_classes)
            fold_models.append(m)
        full = _ScaledKNN(task="classification", n_neighbors=k, weights="distance")
        full.fit(X, y_enc)
        return oof, fold_models, full

    # ------------------------------------------------------------------
    def _align_proba_columns(self, p, model, n_classes):
        """若 fold 訓練集少了某類別，predict_proba 欄數會比 n_classes 小，需補 0 欄。"""
        if p.shape[1] == n_classes:
            return p
        full = np.zeros((p.shape[0], n_classes), dtype=np.float32)
        # sklearn 風格的 model.classes_ 是該 fold 出現過的類別
        cls = getattr(model, "classes_", None)
        if cls is None:
            full[:, : p.shape[1]] = p
            return full
        for j, c in enumerate(cls):
            full[:, int(c)] = p[:, j]
        return full

    def _align_proba_columns_knn(self, p, knn_wrapper, n_classes):
        if p.shape[1] == n_classes:
            return p
        full = np.zeros((p.shape[0], n_classes), dtype=np.float32)
        cls = knn_wrapper.model_.classes_
        for j, c in enumerate(cls):
            full[:, int(c)] = p[:, j]
        return full

    # ------------------------------------------------------------------
    def _nas_oof(self, X, y_enc, cv, n_classes, seed=42):
        """NAS 的 OOF：每折獨立訓練 supernet。NAS 不做 fold-bagging（成本太高）。"""
        n = len(X)
        if self.task == "classification":
            oof = np.zeros((n, n_classes), dtype=np.float32)
        else:
            oof = np.zeros((n, 1), dtype=np.float32)

        import torch
        for fold_idx, (tr_idx, va_idx) in enumerate(
            cv.split(X, y_enc if self.task == "classification" else None)
        ):
            print(f"    [NAS OOF seed={seed}] fold {fold_idx + 1}...")
            # 每折 + 每 seed 都用獨立隨機種子
            np.random.seed(seed * 1000 + fold_idx)
            torch.manual_seed(seed * 1000 + fold_idx)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed * 1000 + fold_idx)

            X_tr, X_val = X[tr_idx], X[va_idx]
            y_tr = y_enc[tr_idx]
            nas = NeuralArchitectureSearcher(
                task=self.task,
                n_supernet_epochs=15,
                n_arch_candidates=10,
                n_evolution_rounds=2,
            )
            nas.fit(X_tr, y_tr)
            if self.task == "classification":
                oof[va_idx] = nas.predict_proba(X_val)
            else:
                oof[va_idx, 0] = nas.predict(X_val)
        return oof

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> dict:
        """
        對測試集，每個 base 模型回傳 (n_test, n_classes) 機率陣列。
        對非 NAS 模型使用 fold-bagging（5 個 fold-models 取平均）。
        回傳 dict[tag] -> proba ndarray。
        """
        n_classes = len(self.classes_) if self.task == "classification" else 1
        out = {}
        for tag in self.model_tags_:
            fold_ms = self.fold_models_.get(tag, [])
            full_m = self.full_models_.get(tag)
            if not fold_ms and full_m is None:
                continue

            # NAS：fold_models 可能只有一個 NASWrapper（refit-on-full），不做 fold-bagging
            is_nas = tag.startswith("nas_")
            if is_nas:
                p = self._proba_one(full_m, X, n_classes)
            else:
                # fold-bagging：5 個 fold-models 各自 predict_proba 取平均
                # 加上 full-data model 一起平均（穩定性 +）
                preds = [self._proba_one(m, X, n_classes) for m in fold_ms]
                if full_m is not None:
                    preds.append(self._proba_one(full_m, X, n_classes))
                p = np.mean(np.stack(preds, axis=0), axis=0)
            out[tag] = p
        return out

    def _proba_one(self, model, X, n_classes):
        """統一介面：回傳對齊後的 proba (n_test, n_classes)。回歸則為 (n_test, 1)。"""
        if self.task == "classification":
            p = model.predict_proba(X)
            # 對齊類別欄
            if p.shape[1] != n_classes:
                full = np.zeros((p.shape[0], n_classes), dtype=np.float32)
                cls = getattr(model, "classes_", None)
                if cls is None and hasattr(model, "model_"):
                    cls = model.model_.classes_  # _ScaledKNN
                if cls is not None:
                    for j, c in enumerate(cls):
                        full[:, int(c)] = p[:, j]
                else:
                    full[:, : p.shape[1]] = p
                return full
            return p
        # 回歸
        return model.predict(X).reshape(-1, 1)


# ---------------------------------------------------------------------------
# L2 Multi-Strategy Meta Layer
# ---------------------------------------------------------------------------

class L2EnsembleLayer:
    """
    多策略 L2 meta-learner：
        - lr            ：Logistic Regression 在 OOF 上學線性混合
        - best_single   ：直接挑 OOF 最佳的單一 base（exploit 單一強模）
        - weighted_avg  ：依 OOF micro F1 對所有 base proba 加權算術平均
        - power_blend   ：依 OOF F1 的 k 次方加權幾何平均（強化最佳模型）
        - top3_avg      ：取 OOF 前 3 強的 base 算術平均
    最後 strategy_ 由「每策略的 OOF micro F1」決定。
    """

    def __init__(self, task="classification", power_k: float = 8.0):
        self.task = task
        self.power_k = power_k
        # 訓練後填入：
        self.label_encoder_ = None
        self.classes_ = None
        self.lr_ = None                # LR meta-learner
        self.ridge_ = None             # 回歸用
        self.tags_ = None              # base 模型順序
        self.scores_ = None            # 各 base 的 OOF micro F1
        self.weights_avg_ = None       # weighted_avg 權重
        self.weights_pow_ = None       # power_blend 權重
        self.top3_tags_ = None         # 取前 3 名的 tag list
        self.strategy_ = None          # 最後選用的策略
        self.strategy_scores_ = None   # 每策略的 OOF micro F1（dict）

    # ------------------------------------------------------------------
    def fit(self, l1: L1BaseLayer, y: np.ndarray):
        if self.task != "classification":
            return self._fit_regression(l1, y)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.tags_ = list(l1.model_tags_)
        self.scores_ = np.array([l1.oof_score_per_model_[t] for t in self.tags_], dtype=np.float64)

        # ── 1) LR meta-learner ──
        # 注意：LR 在 OOF 上訓練，若用同一份 OOF 算分會嚴重高估（in-sample bias）。
        # 用 nested 5-fold CV 取得 LR 的「無偏」OOF 預測，再算分數作為策略比較基準。
        meta_train = np.hstack([l1.oof_per_model_[t] for t in self.tags_])
        self.lr_ = LogisticRegression(
            C=1.0, max_iter=2000, random_state=42, multi_class="auto",
        )
        nested_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            lr_oof_pred = cross_val_predict(
                LogisticRegression(C=1.0, max_iter=2000, random_state=42, multi_class="auto"),
                meta_train, y_enc, cv=nested_cv, n_jobs=-1,
            )
            lr_score = f1_score(y_enc, lr_oof_pred, average="micro")
        except Exception as e:
            print(f"  [L2] nested LR CV failed ({e}); fall back to in-sample score (biased)")
            lr_score = f1_score(y_enc, self.lr_.fit(meta_train, y_enc).predict(meta_train), average="micro")
        # 最終 inference 仍用 fit-on-full meta_train 的 LR
        self.lr_.fit(meta_train, y_enc)

        # ── 2) best_single ──
        best_idx = int(np.argmax(self.scores_))
        best_tag = self.tags_[best_idx]
        best_oof = l1.oof_per_model_[best_tag]
        best_pred = best_oof.argmax(axis=1)
        best_score = f1_score(y_enc, best_pred, average="micro")

        # ── 3) weighted_avg：權重 = softmax(score / 0.05) ──
        T = 0.05
        w = np.exp((self.scores_ - self.scores_.max()) / T)
        w = w / w.sum()
        self.weights_avg_ = w
        avg_proba = self._weighted_avg_proba(l1.oof_per_model_, w)
        avg_pred = avg_proba.argmax(axis=1)
        avg_score = f1_score(y_enc, avg_pred, average="micro")

        # ── 4) power_blend：log-space 加權幾何平均（避免乘法浮點下溢） ──
        wp = np.maximum(self.scores_, 0.0) ** self.power_k
        if wp.sum() == 0:
            wp = np.ones_like(wp) / len(wp)
        else:
            wp = wp / wp.sum()
        self.weights_pow_ = wp
        pow_proba = self._geometric_blend(l1.oof_per_model_, wp)
        pow_pred = pow_proba.argmax(axis=1)
        pow_score = f1_score(y_enc, pow_pred, average="micro")

        # ── 5) top3_avg：取 OOF 前 3 強平均 ──
        top_n = min(3, len(self.tags_))
        top_idx = np.argsort(self.scores_)[::-1][:top_n]
        self.top3_tags_ = [self.tags_[i] for i in top_idx]
        top_w = np.ones(top_n) / top_n
        top3_proba = self._weighted_avg_proba(
            {t: l1.oof_per_model_[t] for t in self.top3_tags_}, top_w
        )
        top3_pred = top3_proba.argmax(axis=1)
        top3_score = f1_score(y_enc, top3_pred, average="micro")

        # ── 比較所有策略，選 OOF 最強者 ──
        self.strategy_scores_ = {
            "lr": lr_score,
            "best_single": best_score,
            "weighted_avg": avg_score,
            "power_blend": pow_score,
            "top3_avg": top3_score,
        }
        # 並列時優先順序：power_blend > weighted_avg > top3_avg > lr > best_single
        # （power 通常最穩，best_single 最容易 over-fit 到 OOF）
        priority = ["power_blend", "weighted_avg", "top3_avg", "lr", "best_single"]
        best_strategy = max(
            self.strategy_scores_.keys(),
            key=lambda k: (self.strategy_scores_[k], -priority.index(k)),
        )
        self.strategy_ = best_strategy
        print("  [L2] OOF F1 per strategy:")
        for k in priority:
            mark = "  <-- selected" if k == best_strategy else ""
            print(f"        {k:14s}: {self.strategy_scores_[k]:.4f}{mark}")
        print(f"  [L2] best base: {best_tag}({self.scores_[best_idx]:.4f})")
        return self

    # ------------------------------------------------------------------
    def _fit_regression(self, l1: L1BaseLayer, y: np.ndarray):
        # 回歸：簡單 Ridge 在 OOF 上學線性 combiner（保持與舊版相容）
        self.tags_ = list(l1.model_tags_)
        meta_train = np.hstack([l1.oof_per_model_[t] for t in self.tags_])
        self.ridge_ = Ridge(alpha=1.0)
        self.ridge_.fit(meta_train, y.astype(np.float64))
        self.strategy_ = "ridge"
        return self

    # ------------------------------------------------------------------
    def _weighted_avg_proba(self, oof_dict: dict, weights: np.ndarray) -> np.ndarray:
        keys = list(oof_dict.keys())
        stacked = np.stack([oof_dict[k] for k in keys], axis=0)  # (M, N, C)
        return (stacked * weights.reshape(-1, 1, 1)).sum(axis=0)

    def _geometric_blend(self, oof_dict: dict, weights: np.ndarray) -> np.ndarray:
        """log-space 幾何平均，等效於 ∏ p_i^{w_i} 後再 normalize。"""
        keys = list(oof_dict.keys())
        log_p = np.stack(
            [np.log(np.clip(oof_dict[k], 1e-9, 1.0)) for k in keys], axis=0
        )  # (M, N, C)
        log_blend = (log_p * weights.reshape(-1, 1, 1)).sum(axis=0)  # (N, C)
        # softmax-normalize
        log_blend = log_blend - log_blend.max(axis=1, keepdims=True)
        p = np.exp(log_blend)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    # ------------------------------------------------------------------
    def predict(self, l1: L1BaseLayer, X: np.ndarray) -> np.ndarray:
        # 拿到每個 base 模型在 X 上的 proba dict
        proba_dict = l1.predict(X)

        if self.task != "classification":
            meta_test = np.hstack([proba_dict[t] for t in self.tags_])
            return self.ridge_.predict(meta_test)

        if self.strategy_ == "lr":
            meta_test = np.hstack([proba_dict[t] for t in self.tags_])
            int_pred = self.lr_.predict(meta_test)
        elif self.strategy_ == "best_single":
            best_idx = int(np.argmax(self.scores_))
            best_tag = self.tags_[best_idx]
            int_pred = proba_dict[best_tag].argmax(axis=1)
        elif self.strategy_ == "weighted_avg":
            sub = {t: proba_dict[t] for t in self.tags_}
            int_pred = self._weighted_avg_proba(sub, self.weights_avg_).argmax(axis=1)
        elif self.strategy_ == "power_blend":
            sub = {t: proba_dict[t] for t in self.tags_}
            int_pred = self._geometric_blend(sub, self.weights_pow_).argmax(axis=1)
        elif self.strategy_ == "top3_avg":
            sub = {t: proba_dict[t] for t in self.top3_tags_}
            top_w = np.ones(len(self.top3_tags_)) / len(self.top3_tags_)
            int_pred = self._weighted_avg_proba(sub, top_w).argmax(axis=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_}")

        return self.label_encoder_.inverse_transform(int_pred.astype(int))

    def predict_proba(self, l1: L1BaseLayer, X: np.ndarray) -> np.ndarray:
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        proba_dict = l1.predict(X)
        if self.strategy_ == "lr":
            meta_test = np.hstack([proba_dict[t] for t in self.tags_])
            return self.lr_.predict_proba(meta_test)
        if self.strategy_ == "best_single":
            best_idx = int(np.argmax(self.scores_))
            return proba_dict[self.tags_[best_idx]]
        if self.strategy_ == "weighted_avg":
            return self._weighted_avg_proba(
                {t: proba_dict[t] for t in self.tags_}, self.weights_avg_
            )
        if self.strategy_ == "power_blend":
            return self._geometric_blend(
                {t: proba_dict[t] for t in self.tags_}, self.weights_pow_
            )
        if self.strategy_ == "top3_avg":
            sub = {t: proba_dict[t] for t in self.top3_tags_}
            top_w = np.ones(len(self.top3_tags_)) / len(self.top3_tags_)
            return self._weighted_avg_proba(sub, top_w)
        raise ValueError(f"Unknown strategy: {self.strategy_}")


# ---------------------------------------------------------------------------
# 完整堆疊集成（L1 + L2 對外統一介面）
# ---------------------------------------------------------------------------

class StackingEnsemble:
    """
    L1 Base Layer（多模型 OOF + fold-bagging）
        + L2 多策略 Meta Layer（自動挑 OOF 最佳策略）。
    """

    def __init__(self, task="classification", n_folds=5, top_k_configs=2,
                 use_nas=True, n_nas_seeds=1,
                 use_extratrees=True, use_knn=True):
        self.task = task
        self.n_folds = n_folds
        self.top_k_configs = top_k_configs
        self.use_nas = use_nas
        self.n_nas_seeds = n_nas_seeds
        self.use_extratrees = use_extratrees
        self.use_knn = use_knn
        self.l1_ = L1BaseLayer(
            task=task, n_folds=n_folds, top_k_configs=top_k_configs,
            n_nas_seeds=n_nas_seeds, use_extratrees=use_extratrees, use_knn=use_knn,
        )
        self.l2_ = L2EnsembleLayer(task=task)

    def fit(self, X: np.ndarray, y: np.ndarray, hpo_configs: dict):
        print("  [Ensemble] Training L1 Base Layer...")
        _ = self.l1_.fit_predict(X, y, hpo_configs, use_nas=self.use_nas)
        meta_dim = sum(o.shape[1] for o in self.l1_.oof_per_model_.values())
        print(f"  [Ensemble] Meta-feature shape: ({len(X)}, {meta_dim})")
        print("  [Ensemble] Training L2 Multi-Strategy Meta-Learner...")
        self.l2_.fit(self.l1_, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.l2_.predict(self.l1_, X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.l2_.predict_proba(self.l1_, X)
