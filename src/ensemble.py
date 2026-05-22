"""
模型融合模組。

Ensemble A — Nelder-Mead Weighted Blending：
    以幾何平均融合各模型預測機率，用 scipy.optimize.minimize（Nelder-Mead）
    自動搜尋最佳 log-scale 權重，最大化 OOF Macro F1。

Ensemble B — Meta-Learner Stacking：
    將所有模型的 OOF 預測機率拼接為 meta-features，再用 5-Fold CV 訓練
    Meta-Learner（LGBM 或 LogReg，由評估分數決定）。

所有 Meta-Learner 超參數亦由 Optuna TPE 搜尋，不人為固定。
"""
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from .metrics import calculate_score, get_metric_name
import optuna
from tqdm import tqdm

from .config import SEED

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── 二元分類閾值搜尋 ─────────────────────────────────────────────────────────

def _find_best_threshold(
    proba: np.ndarray, y: np.ndarray, metric: str
) -> tuple:
    """
    二元分類專用：在 OOF 機率上搜尋正類閾值使指標最大。
    回傳 (best_threshold, best_score)。
    若類別數 != 2 則直接回傳 (0.5, nan)。
    """
    if proba.shape[1] != 2:
        return 0.5, float("nan")
    best_t, best_score = 0.5, -np.inf
    for t in np.linspace(0.10, 0.90, 81):
        preds = (proba[:, 1] >= t).astype(int)
        score = calculate_score(y, preds, metric=metric)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score


# ── 幾何平均融合工具 ─────────────────────────────────────────────────────────

def _geometric_blend(preds_list: list, weights: np.ndarray) -> np.ndarray:
    """
    加權幾何平均：result[i,c] = prod_j(pred_j[i,c]^w_j)，再列歸一化。
    preds_list : list of [N, C] arrays
    weights    : [M] 正值陣列，和為 1
    """
    result = np.ones_like(preds_list[0], dtype=np.float64)
    for pred, w in zip(preds_list, weights):
        result *= np.clip(pred, 1e-8, 1.0) ** w
    total = result.sum(axis=1, keepdims=True) + 1e-12
    return (result / total).astype(np.float32)


# ── Ensemble A：Nelder-Mead Weighted Blending ───────────────────────────────

class NelderMeadBlender:
    """
    最佳化各模型的融合權重（log-space softmax 確保和為 1）。
    目標函數：OOF Macro F1（最大化 = minimize negative F1）。
    """

    def __init__(self, n_restarts: int = 3, metric: str = "f1"):
        self.n_restarts = n_restarts
        self.metric = metric
        self.weights_: np.ndarray = None

    def fit(self, oof_list: list, y: np.ndarray) -> "NelderMeadBlender":
        """
        Parameters
        ----------
        oof_list : list of [N, C] OOF 預測機率陣列
        y        : [N] 真實標籤（整數）
        """
        n_models = len(oof_list)

        def score_func(logit_w):
            w = softmax(logit_w)
            blended = _geometric_blend(oof_list, w)
            preds = blended.argmax(axis=1)
            # 最大化指標 = 最小化負指標
            return -calculate_score(y, preds, metric=self.metric)

        best_val, best_x = np.inf, np.zeros(n_models)

        print(f"  [Blend] Nelder-Mead search ({self.n_restarts} restarts, {n_models} models) ...")
        for restart in tqdm(range(self.n_restarts), desc="  Blend restarts", ncols=70):
            x0 = np.random.default_rng(SEED + restart).normal(0, 0.5, n_models)
            res = minimize(score_func, x0, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        self.weights_ = softmax(best_x)
        oof_score = -best_val
        print(f"  [Blend] Best OOF {get_metric_name(self.metric)} = {oof_score:.4f}")
        print(f"  [Blend] Weights: {np.round(self.weights_, 3).tolist()}")

        # 二元分類 + 非 accuracy 指標：在 OOF 機率上搜尋最佳決策閾值
        self._threshold = 0.5
        if self.metric != "accuracy":
            oof_proba = self.predict_proba(oof_list)
            self._threshold, thresh_score = _find_best_threshold(oof_proba, y, self.metric)
            if oof_proba.shape[1] == 2:
                print(f"  [Blend] Threshold={self._threshold:.2f}  "
                      f"OOF {get_metric_name(self.metric)} {oof_score:.4f}→{thresh_score:.4f}")
        return self

    def predict_proba(self, test_list: list) -> np.ndarray:
        return _geometric_blend(test_list, self.weights_)

    def predict(self, test_list: list) -> np.ndarray:
        proba = self.predict_proba(test_list)
        t = getattr(self, "_threshold", 0.5)
        if proba.shape[1] == 2 and t != 0.5:
            return (proba[:, 1] >= t).astype(int)
        return proba.argmax(axis=1)


# ── Ensemble B：Meta-Learner Stacking ────────────────────────────────────────

class MetaLearnerStacker:
    """
    L1 OOF 預測 → meta-features → L2 Meta-Learner（HPO 選擇 LGBM 或 LogReg）。

    若傳入 X_orig（原始特徵），採用 Concatenated Stacking：
    meta-input = [OOF_predictions | original_features]，讓 Meta-Learner 能學習
    「在哪種特徵條件下該信任哪個基模型」的條件上下文。
    """

    def __init__(
        self, n_meta_trials: int = 30, n_folds: int = 5,
        metric: str = "f1", n_samples: int = 10_000,
        k_best: int = 50,
    ):
        self.n_meta_trials = n_meta_trials
        self.n_folds = n_folds
        self.metric = metric
        self.n_samples = n_samples
        self.k_best = k_best
        self.meta_model_ = None
        self.meta_name_: str = None
        self._x_scaler = None
        self._selector = None

    def _build_meta_features(self, oof_list: list, X_orig: np.ndarray = None) -> np.ndarray:
        """把所有模型的 OOF 機率拼接；若有 X_orig 則一併拼入（Concatenated Stacking）。"""
        meta = np.hstack(oof_list)  # [N, M*C]
        if X_orig is not None:
            meta = np.hstack([meta, X_orig])
        return meta  # [N, M*C + F]

    def _hpo_meta(self, X_meta: np.ndarray, y: np.ndarray) -> tuple:
        # X_meta 已包含 OOF + 可選的原始特徵（均由呼叫方縮放）
        """用 Optuna TPE 搜尋 meta-learner 類型 + 超參數，回傳 (best_name, best_params, best_score)。"""
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression

        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)

        # 小樣本強制使用 LogReg：LGBM/XGB 在 <2000 筆 meta-features 上容易過擬合
        if self.n_samples < 2000:
            _meta_candidates = ["logreg"]
            print(f"  [Stack] n_samples={self.n_samples}<2000 → 強制使用 LogReg meta-learner")
        else:
            _meta_candidates = ["lgbm", "logreg", "xgb"]

        def objective(trial):
            meta_type = trial.suggest_categorical("meta_type", _meta_candidates)
            if meta_type == "lgbm":
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                }
                model_fn = lambda p: lgb.LGBMClassifier(
                    **p, random_state=SEED, n_jobs=-1, verbose=-1
                )
            elif meta_type == "xgb":
                import xgboost as _xgb_meta
                params = {
                    "n_estimators": trial.suggest_int("xgb_n_est", 50, 400),
                    "max_depth": trial.suggest_int("xgb_depth", 2, 6),
                    "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.3, log=True),
                    "reg_lambda": trial.suggest_float("xgb_lambda", 1e-4, 10.0, log=True),
                    "subsample": trial.suggest_float("xgb_sub", 0.6, 1.0),
                }
                model_fn = lambda p: _xgb_meta.XGBClassifier(
                    **p, random_state=SEED, verbosity=0, n_jobs=-1,
                    use_label_encoder=False,
                )
            else:
                params = {
                    "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                }
                model_fn = lambda p: LogisticRegression(
                    **p, max_iter=2000, random_state=SEED, n_jobs=1
                )

            scores = []
            for tr_i, val_i in cv.split(X_meta, y):
                m = model_fn(params)
                m.fit(X_meta[tr_i], y[tr_i])
                y_hat = m.predict(X_meta[val_i])
                scores.append(calculate_score(y[val_i], y_hat, metric=self.metric))

            trial.set_user_attr("meta_type", meta_type)
            trial.set_user_attr("params", params)
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with tqdm(total=self.n_meta_trials, desc="  MetaHPO       ", ncols=70, unit="trial") as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                if study.best_trial:
                    _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
            study.optimize(objective, n_trials=self.n_meta_trials, callbacks=[_cb])

        best_t = study.best_trial
        return (
            best_t.user_attrs["meta_type"],
            best_t.user_attrs["params"],
            best_t.value,
        )

    def fit(
        self,
        oof_list: list,
        y: np.ndarray,
        X_orig: np.ndarray = None,
        is_timeseries: bool = False,
    ) -> "MetaLearnerStacker":
        """
        Parameters
        ----------
        oof_list       : list of [N, C] OOF 預測機率陣列
        y              : [N] 真實標籤
        X_orig         : [N, F] 原始特徵（Concatenated Stacking 用）
        is_timeseries  : 若為 True，對訓練樣本套用時間衰減權重
                         （越接近測試集時間點的樣本獲得越高的損失權重）
        """
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # 若傳入原始特徵，先標準化再拼入（避免高維 raw 特徵淹沒 OOF 機率）
        X_orig_scaled = None
        if X_orig is not None:
            self._x_scaler = StandardScaler()
            X_orig_scaled = self._x_scaler.fit_transform(X_orig)
            from sklearn.feature_selection import SelectKBest, f_classif
            actual_k = min(self.k_best, X_orig_scaled.shape[1])
            self._selector = SelectKBest(f_classif, k=actual_k)
            X_orig_scaled = self._selector.fit_transform(X_orig_scaled, y)

        X_meta = self._build_meta_features(oof_list, X_orig_scaled)

        mode = "OOF+RawFeatures" if X_orig is not None else "OOF only"
        print(f"  [Stack] Meta-feature shape: {X_meta.shape}  ({mode})")
        meta_name, meta_params, meta_score = self._hpo_meta(X_meta, y)
        print(f"  [Stack] Best meta-learner = {meta_name}  OOF {get_metric_name(self.metric)} = {meta_score:.4f}")

        if meta_name == "lgbm":
            self.meta_model_ = lgb.LGBMClassifier(
                **meta_params, random_state=SEED, n_jobs=-1, verbose=-1
            )
        elif meta_name == "xgb":
            import xgboost as _xgb_meta
            self.meta_model_ = _xgb_meta.XGBClassifier(
                **meta_params, random_state=SEED, verbosity=0, n_jobs=-1,
                use_label_encoder=False,
            )
        else:
            self.meta_model_ = LogisticRegression(
                **meta_params, max_iter=2000, random_state=SEED, n_jobs=1
            )

        # 時間衰減權重：越靠近測試集時間點（索引越大）的樣本權重越高
        sample_weight = None
        if is_timeseries:
            n = len(y)
            sample_weight = np.linspace(0.3, 1.0, n)
            print(f"  [Stack] 時間衰減權重已啟用（min=0.30, max=1.00，共 {n} 個樣本）")

        if sample_weight is not None:
            self.meta_model_.fit(X_meta, y, sample_weight=sample_weight)
        else:
            self.meta_model_.fit(X_meta, y)

        self.meta_name_ = meta_name

        # 二元分類 + 非 accuracy 指標：在 OOF meta-features 上搜尋最佳決策閾值
        self._threshold = 0.5
        if self.metric != "accuracy":
            oof_proba = self.meta_model_.predict_proba(X_meta)
            self._threshold, thresh_score = _find_best_threshold(oof_proba, y, self.metric)
            if oof_proba.shape[1] == 2:
                print(f"  [Stack] Threshold={self._threshold:.2f}  "
                      f"OOF {get_metric_name(self.metric)} {meta_score:.4f}→{thresh_score:.4f}")
        return self

    def predict_proba(self, test_list: list, X_orig: np.ndarray = None) -> np.ndarray:
        X_orig_scaled = None
        if X_orig is not None and self._x_scaler is not None:
            X_orig_scaled = self._x_scaler.transform(X_orig)
            if self._selector is not None:
                X_orig_scaled = self._selector.transform(X_orig_scaled)
        X_meta = self._build_meta_features(test_list, X_orig_scaled)
        return self.meta_model_.predict_proba(X_meta)

    def predict(self, test_list: list, X_orig: np.ndarray = None) -> np.ndarray:
        X_orig_scaled = None
        if X_orig is not None and self._x_scaler is not None:
            X_orig_scaled = self._x_scaler.transform(X_orig)
            if self._selector is not None:
                X_orig_scaled = self._selector.transform(X_orig_scaled)
        X_meta = self._build_meta_features(test_list, X_orig_scaled)
        proba = self.meta_model_.predict_proba(X_meta)
        t = getattr(self, "_threshold", 0.5)
        if proba.shape[1] == 2 and t != 0.5:
            return (proba[:, 1] >= t).astype(int)
        return proba.argmax(axis=1)
