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
import optuna
from tqdm import tqdm

from .config import SEED, N_SPLITS

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


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

    def __init__(self, n_restarts: int = 3):
        self.n_restarts = n_restarts
        self.weights_: np.ndarray = None

    def fit(self, oof_list: list, y: np.ndarray) -> "NelderMeadBlender":
        """
        Parameters
        ----------
        oof_list : list of [N, C] OOF 預測機率陣列
        y        : [N] 真實標籤（整數）
        """
        n_models = len(oof_list)

        def neg_f1(logit_w):
            w = softmax(logit_w)
            blended = _geometric_blend(oof_list, w)
            preds = blended.argmax(axis=1)
            return -f1_score(y, preds, average="macro", zero_division=0)

        best_val, best_x = np.inf, np.zeros(n_models)

        print(f"  [Blend] Nelder-Mead search ({self.n_restarts} restarts, {n_models} models) ...")
        for restart in tqdm(range(self.n_restarts), desc="  Blend restarts", ncols=70):
            x0 = np.random.default_rng(SEED + restart).normal(0, 0.5, n_models)
            res = minimize(neg_f1, x0, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x

        self.weights_ = softmax(best_x)
        oof_f1 = -best_val
        print(f"  [Blend] Best OOF Macro F1 = {oof_f1:.4f}")
        print(f"  [Blend] Weights: {np.round(self.weights_, 3).tolist()}")
        return self

    def predict_proba(self, test_list: list) -> np.ndarray:
        return _geometric_blend(test_list, self.weights_)

    def predict(self, test_list: list) -> np.ndarray:
        return self.predict_proba(test_list).argmax(axis=1)


# ── Ensemble B：Meta-Learner Stacking ────────────────────────────────────────

class MetaLearnerStacker:
    """
    L1 OOF 預測 → meta-features → L2 Meta-Learner（HPO 選擇 LGBM 或 LogReg）。
    """

    def __init__(self, n_meta_trials: int = 30, n_folds: int = N_SPLITS):
        self.n_meta_trials = n_meta_trials
        self.n_folds = n_folds
        self.meta_model_ = None
        self.meta_name_: str = None

    def _build_meta_features(self, oof_list: list) -> np.ndarray:
        """把所有模型的 OOF 機率橫向拼接成 meta-feature 矩陣。"""
        return np.hstack(oof_list)  # [N, M*C]

    def _hpo_meta(self, X_meta: np.ndarray, y: np.ndarray) -> tuple:
        """用 Optuna TPE 搜尋 meta-learner 類型 + 超參數，回傳 (best_name, best_params, best_score)。"""
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression

        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)

        def objective(trial):
            meta_type = trial.suggest_categorical("meta_type", ["lgbm", "logreg"])
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
            else:
                params = {
                    "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                }
                model_fn = lambda p: LogisticRegression(
                    **p, max_iter=2000, random_state=SEED, n_jobs=-1
                )

            scores = []
            for tr_i, val_i in cv.split(X_meta, y):
                m = model_fn(params)
                m.fit(X_meta[tr_i], y[tr_i])
                y_hat = m.predict(X_meta[val_i])
                scores.append(f1_score(y[val_i], y_hat, average="macro", zero_division=0))

            trial.set_user_attr("meta_type", meta_type)
            trial.set_user_attr("params", params)
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with tqdm(total=self.n_meta_trials, desc="  MetaHPO       ", ncols=70, unit="trial") as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                if study.best_trial:
                    _pbar.set_postfix({"best_f1": f"{study.best_value:.4f}"})
            study.optimize(objective, n_trials=self.n_meta_trials, callbacks=[_cb])

        best_t = study.best_trial
        return (
            best_t.user_attrs["meta_type"],
            best_t.user_attrs["params"],
            best_t.value,
        )

    def fit(self, oof_list: list, y: np.ndarray) -> "MetaLearnerStacker":
        import lightgbm as lgb
        from sklearn.linear_model import LogisticRegression

        X_meta = self._build_meta_features(oof_list)

        print(f"  [Stack] Meta-feature shape: {X_meta.shape}")
        meta_name, meta_params, meta_f1 = self._hpo_meta(X_meta, y)
        print(f"  [Stack] Best meta-learner = {meta_name}  OOF F1 = {meta_f1:.4f}")

        if meta_name == "lgbm":
            self.meta_model_ = lgb.LGBMClassifier(
                **meta_params, random_state=SEED, n_jobs=-1, verbose=-1
            )
        else:
            self.meta_model_ = LogisticRegression(
                **meta_params, max_iter=2000, random_state=SEED, n_jobs=-1
            )

        self.meta_model_.fit(X_meta, y)
        self.meta_name_ = meta_name
        return self

    def predict_proba(self, test_list: list) -> np.ndarray:
        X_meta = self._build_meta_features(test_list)
        return self.meta_model_.predict_proba(X_meta)

    def predict(self, test_list: list) -> np.ndarray:
        X_meta = self._build_meta_features(test_list)
        return self.meta_model_.predict(X_meta)
