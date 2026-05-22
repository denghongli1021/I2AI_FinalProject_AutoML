"""
pipeline_time.py — 時序專用 Pipeline 引擎（分類 + 回歸）

對外提供：
  - TimeBudget          : 直接重用 pipeline.TimeBudget
  - get_cfg_time()      : 時序專用 HPO 設定
  - run_classification(): 時序分類 → 委派給 pipeline.run(is_ts=True)
  - run_regression()    : 時序回歸 → 自帶 HPO/CV/Ensemble（單值輸出）

回歸實作摘要：
  - Tabular: LGBMRegressor/XGBRegressor/CatBoostRegressor/Ridge/RF/ExtraTrees/KNN
  - DL: TSNet / TCN / PatchTST（n_classes=1，MSE loss）
  - CV: TimeSeriesSplit Walk-forward
  - Ensemble A: Nelder-Mead 加權算術平均（最小化 RMSE）
  - Ensemble B: Meta-Learner Stacking（Ridge / LGBMRegressor）
  - 不做 NAS（依使用者指定方案 B），TSNet 使用 _DEFAULT_TSNET_ARCH
"""
from __future__ import annotations

import math
import os
import time
import warnings

import numpy as np
import optuna
import torch
import torch.nn as nn
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from tqdm import tqdm

import lightgbm as lgb
import xgboost as xgb

try:
    from catboost import CatBoostRegressor
    _CATBOOST_OK = True
except ImportError:
    _CATBOOST_OK = False

from src.config import ARTIFACTS_DIR, DEVICE, SEED
from src.data import get_ts_folds
from src.preprocess import (
    FeatureBuilder,
    TS_DL_FEATURE_SETS,
    TS_TABULAR_FEATURE_SETS,
)

# 重用 pipeline.py 的 TimeBudget
from pipeline import TimeBudget

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── 常數 ─────────────────────────────────────────────────────────────────────

_ALL_REG_TABULAR_MODELS = ["lgbm", "xgb", "catboost", "rf", "extra_trees", "ridge", "knn"]

_DEFAULT_TSNET_ARCH = {
    "n_blocks": 3,
    "operations": [0, 2, 3],   # conv_k3, tcn_d2, tcn_d4
    "channels": 64,
    "dropout": 0.1,
}


# ── 配置 ─────────────────────────────────────────────────────────────────────

def get_cfg_time(fast: bool, n_samples: int = 10_000) -> dict:
    """時序專用 HPO 設定。比 tabular get_cfg 略小，因 TS 樣本通常較少。"""
    if fast:
        return {
            "tabular_trials": 5, "tabular_top_k": 1,
            "scout_trials": 3, "scout_val_size": 0.2, "scout_ratio": 2 / 3,
            "tsnet_trials": 3, "tsnet_top_k": 1,
            "dl_trials": 3, "dl_top_k": 1,
            "meta_trials": 5, "blend_restarts": 1,
            "n_repeats": 1, "n_seeds": 1,
            "is_timeseries": True,
        }
    if n_samples < 500:
        return {
            "tabular_trials": 15, "tabular_top_k": 2,
            "scout_trials": 5, "scout_val_size": 0.2, "scout_ratio": 2 / 3,
            "tsnet_trials": 6, "tsnet_top_k": 1,
            "dl_trials": 6, "dl_top_k": 1,
            "meta_trials": 10, "blend_restarts": 2,
            "n_repeats": 1, "n_seeds": 1,
            "is_timeseries": True,
        }
    if n_samples < 50_000:
        return {
            "tabular_trials": 15, "tabular_top_k": 2,
            "scout_trials": 5, "scout_val_size": 0.2, "scout_ratio": 2 / 3,
            "tsnet_trials": 6, "tsnet_top_k": 1,
            "dl_trials": 6, "dl_top_k": 1,
            "meta_trials": 12, "blend_restarts": 2,
            "n_repeats": 1, "n_seeds": 1,
            "is_timeseries": True,
        }
    return {
        "tabular_trials": 8, "tabular_top_k": 1,
        "scout_trials": 4, "scout_val_size": 0.2, "scout_ratio": 2 / 3,
        "tsnet_trials": 4, "tsnet_top_k": 1,
        "dl_trials": 4, "dl_top_k": 1,
        "meta_trials": 8, "blend_restarts": 1,
        "n_repeats": 1, "n_seeds": 1,
        "is_timeseries": True,
    }


# ── 指標 ─────────────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _r2(y_true, y_pred):
    return float(r2_score(y_true, y_pred))


def _mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, y_pred))


_REG_METRICS = {"rmse": _rmse, "r2": _r2, "mae": _mae}


def calc_reg_score(y_true, y_pred, metric: str = "rmse") -> float:
    return _REG_METRICS[metric](np.asarray(y_true).ravel(), np.asarray(y_pred).ravel())


def reg_metric_direction(metric: str) -> str:
    """Optuna direction for regression metrics."""
    return "maximize" if metric == "r2" else "minimize"


# ── 回歸資料載入（y 為 float） ───────────────────────────────────────────────

class _RegDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def _make_reg_loader(X, y=None, batch_size: int = 128, shuffle: bool = False):
    ds = _RegDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ── Tabular 回歸模型工廠 ─────────────────────────────────────────────────────

def build_reg_tabular_model(name: str, params: dict, seed: int = SEED, device: str = "cpu"):
    use_gpu = device == "cuda"
    if name == "lgbm":
        return lgb.LGBMRegressor(**params, random_state=seed, n_jobs=-2, verbose=-1)
    if name == "xgb":
        return xgb.XGBRegressor(
            **params,
            early_stopping_rounds=100,
            random_state=seed,
            verbosity=0,
            **({"device": "cuda"} if use_gpu else {"n_jobs": -2}),
        )
    if name == "catboost":
        if not _CATBOOST_OK:
            raise ImportError("catboost not installed")
        return CatBoostRegressor(
            **params,
            random_seed=seed,
            verbose=0,
            thread_count=max(1, (os.cpu_count() or 2) - 1),
            od_type="Iter",
            od_wait=30,
            used_ram_limit="4gb",
            max_ctr_complexity=2,
            boosting_type="Plain",
        )
    if name == "rf":
        return RandomForestRegressor(**params, random_state=seed, n_jobs=-2)
    if name == "extra_trees":
        return ExtraTreesRegressor(**params, random_state=seed, n_jobs=-2)
    if name == "ridge":
        return Ridge(**params, random_state=seed)
    if name == "knn":
        return KNeighborsRegressor(**params, n_jobs=-2)
    raise ValueError(f"Unknown regression tabular model: {name}")


# ── HPO 搜尋空間 ─────────────────────────────────────────────────────────────

def _reg_tabular_space(name: str, trial: optuna.Trial, fs_list: list) -> dict:
    fs = trial.suggest_categorical("feature_set", fs_list)
    p = {"feature_set": fs}
    if name == "lgbm":
        p.update({
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 80),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        })
    elif name == "xgb":
        p.update({
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        })
    elif name == "catboost":
        p.update({
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 100, 400),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        })
    elif name == "rf":
        p.update({
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
        })
    elif name == "extra_trees":
        p.update({
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
        })
    elif name == "ridge":
        p.update({
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
        })
    elif name == "knn":
        p.update({
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"]),
        })
    return p


def _reg_dl_train_space(trial: optuna.Trial) -> dict:
    """DL 回歸訓練超參數搜尋空間（無 label_smoothing，無 mixup）。"""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "t_max": trial.suggest_int("t_max", 5, 30),
        "n_epochs": trial.suggest_int("n_epochs", 20, 50),
        "patience": trial.suggest_int("patience", 5, 10),
    }


def _tcn_arch_space(trial: optuna.Trial) -> dict:
    return {
        "n_blocks": trial.suggest_int("n_blocks", 2, 6),
        "channels": trial.suggest_categorical("channels", [32, 64, 128, 256]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


def _patchtst_arch_space(trial: optuna.Trial, in_features: int) -> dict:
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    patch_size = trial.suggest_categorical("patch_size", [4, 8, 16, 32])
    return {
        "patch_size": min(patch_size, max(1, in_features)),
        "d_model": d_model,
        "n_heads": n_heads,
        "depth": trial.suggest_int("depth", 1, 6),
        "ff_dim": trial.suggest_categorical("ff_dim", [64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


# ── DL 回歸模型建構與訓練 ───────────────────────────────────────────────────

def _build_reg_dl_model(model_name: str, arch: dict, in_features: int) -> nn.Module:
    """建立 DL 回歸模型：以 n_classes=1 借用既有 TSNet/TCN/PatchTST，輸出形狀 [B,1]。"""
    if model_name == "tsnet":
        from src.nas import TSNet
        return TSNet(
            in_features=in_features,
            channels=arch.get("channels", 64),
            operations=arch.get("operations", [0, 2, 3]),
            n_classes=1,
            dropout=arch.get("dropout", 0.1),
        )
    if model_name == "tcn":
        from src.models.cnn1d import TCN
        return TCN(
            in_features=in_features,
            n_blocks=arch["n_blocks"],
            channels=arch["channels"],
            kernel_size=arch["kernel_size"],
            dropout=arch["dropout"],
            n_classes=1,
        )
    if model_name == "patchtst":
        from src.models.transformer import PatchTST
        return PatchTST(
            in_features=in_features,
            patch_size=arch["patch_size"],
            d_model=arch["d_model"],
            n_heads=arch["n_heads"],
            depth=arch["depth"],
            ff_dim=arch["ff_dim"],
            dropout=arch["dropout"],
            n_classes=1,
        )
    raise ValueError(f"Unknown reg DL model: {model_name}")


def train_reg_dl_single_fold(
    model_name: str,
    arch: dict,
    train_p: dict,
    X_tr, y_tr, X_val, y_val,
    device: str,
    metric: str = "rmse",
) -> float:
    torch.manual_seed(SEED)
    in_features = X_tr.shape[1]
    model = _build_reg_dl_model(model_name, arch, in_features).to(device)

    lr = train_p["lr"]
    wd = train_p["weight_decay"]
    bs = int(train_p["batch_size"])
    t_max = int(train_p["t_max"])
    n_epochs = int(train_p["n_epochs"])
    patience = int(train_p["patience"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    criterion = nn.MSELoss()

    loader = _make_reg_loader(X_tr, y_tr, batch_size=bs, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    direction = reg_metric_direction(metric)
    best = -np.inf if direction == "maximize" else np.inf
    cnt = 0
    for _ in range(n_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            yp = model(X_val_t).squeeze(-1).cpu().numpy()
        score = calc_reg_score(y_val, yp, metric=metric)

        improved = (score > best) if direction == "maximize" else (score < best)
        if improved:
            best = score
            cnt = 0
        else:
            cnt += 1
            if cnt >= patience:
                break

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return best


# ── Tabular 回歸 HPO ─────────────────────────────────────────────────────────

class TabularRegHPO:
    """時序回歸 Tabular HPO（Scout + Full）。CV 採 TimeSeriesSplit。"""

    def __init__(
        self,
        model_names: list = None,
        n_trials: int = 15,
        n_folds: int = 5,
        top_k: int = 2,
        per_model_trials: dict = None,
        per_model_timeout: dict = None,
        device: str = None,
        metric: str = "rmse",
    ):
        self.model_names = model_names or _ALL_REG_TABULAR_MODELS
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.top_k = top_k
        self.per_model_trials = per_model_trials or {}
        self.per_model_timeout = per_model_timeout or {}
        self.device = device or DEVICE
        self.metric = metric

    # ---- Scout ----
    def scout(self, X, y, scout_trials: int = 5, scout_cv_folds: int = 3, global_cfg: dict = None):
        global_cfg = global_cfg or {}
        folds = get_ts_folds(len(X), n_splits=scout_cv_folds)

        # 預先計算所有 feature_set × fold 的特徵快取
        fs_set = set(TS_TABULAR_FEATURE_SETS)
        cache = {}
        for fs in sorted(fs_set):
            for fi, (tr, vl) in enumerate(folds):
                fb = FeatureBuilder(feature_set=fs, global_cfg=global_cfg)
                cache[(fs, fi)] = (fb.fit_transform(X[tr]), fb.transform(X[vl]))

        scores = {}
        best_params = {}
        direction = reg_metric_direction(self.metric)

        for name in self.model_names:
            fs_candidates = TS_TABULAR_FEATURE_SETS

            def objective(trial, _name=name, _fs=fs_candidates):
                merged = _reg_tabular_space(_name, trial, _fs)
                fs = merged.pop("feature_set")
                fold_scores = []
                for fi, (tr_idx, val_idx) in enumerate(folds):
                    X_t, X_v = cache[(fs, fi)]
                    m = build_reg_tabular_model(_name, merged, device=self.device)
                    if _name == "lgbm":
                        m.fit(X_t, y[tr_idx],
                              eval_set=[(X_v, y[val_idx])],
                              callbacks=[lgb.early_stopping(30, verbose=False),
                                         lgb.log_evaluation(-1)])
                    elif hasattr(m, "early_stopping_rounds") and getattr(m, "early_stopping_rounds", None):
                        m.fit(X_t, y[tr_idx], eval_set=[(X_v, y[val_idx])], verbose=False)
                    else:
                        m.fit(X_t, y[tr_idx])
                    yh = m.predict(X_v)
                    fold_scores.append(calc_reg_score(y[val_idx], yh, metric=self.metric))
                trial.set_user_attr("feature_set", fs)
                return float(np.mean(fold_scores))

            sampler = optuna.samplers.RandomSampler(seed=SEED)
            study = optuna.create_study(direction=direction, sampler=sampler)
            with tqdm(total=scout_trials, desc=f"Scout {name.upper():8s}", unit="trial", ncols=80) as pbar:
                def _cb(study, trial, _pbar=pbar):
                    _pbar.update(1)
                    try:
                        _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                    except ValueError:
                        pass
                scout_timeout = self.per_model_timeout.get(name)
                study.optimize(objective, n_trials=scout_trials, timeout=scout_timeout,
                               callbacks=[_cb], catch=(Exception,))

            try:
                best_tr = study.best_trial
                best_val = study.best_value
            except ValueError:
                best_tr = None
                best_val = float("nan")
            scores[name] = best_val
            if best_tr is not None:
                p = dict(best_tr.params)
                if "feature_set" not in p:
                    fs_val = best_tr.user_attrs.get("feature_set")
                    if fs_val:
                        p["feature_set"] = fs_val
                best_params[name] = p
            print(f"  [Scout] {name.upper():12s} {scout_cv_folds}-Fold CV {self.metric.upper()} = {best_val:.4f}")
        return scores, best_params

    # ---- Full HPO ----
    def run(self, X, y, global_cfg: dict = None, warm_start: dict = None, locked_feature_sets: dict = None):
        global_cfg = global_cfg or {}
        warm_start = warm_start or {}
        locked_feature_sets = locked_feature_sets or {}

        folds = get_ts_folds(len(X), n_splits=self.n_folds)

        # 預算所有 feature_set × fold
        fs_set = set()
        for n in self.model_names:
            if int(self.per_model_trials.get(n, self.n_trials)) <= 0:
                continue
            lk = locked_feature_sets.get(n)
            if lk:
                fs_set.add(lk)
            else:
                fs_set.update(TS_TABULAR_FEATURE_SETS)
        print(f"  [HPO] Pre-computing {len(fs_set)} feature set(s) × {len(folds)} folds ...")
        cache = {}
        for fs in sorted(fs_set):
            for fi, (tr, vl) in enumerate(folds):
                fb = FeatureBuilder(feature_set=fs, global_cfg=global_cfg)
                cache[(fs, fi)] = (fb.fit_transform(X[tr]), fb.transform(X[vl]))

        direction = reg_metric_direction(self.metric)
        all_configs = []
        for name in self.model_names:
            n_tr = int(self.per_model_trials.get(name, self.n_trials))
            if n_tr <= 0:
                continue
            fs_candidates = TS_TABULAR_FEATURE_SETS
            _locked = locked_feature_sets.get(name)

            def objective(trial, _name=name, _fs=fs_candidates, _locked=_locked):
                if _locked:
                    merged = _reg_tabular_space(_name, trial, [_locked])
                else:
                    merged = _reg_tabular_space(_name, trial, _fs)
                fs = merged.pop("feature_set")
                fold_scores = []
                for fi, (tr_idx, val_idx) in enumerate(folds):
                    X_t, X_v = cache[(fs, fi)]
                    m = build_reg_tabular_model(_name, merged, device=self.device)
                    if _name == "lgbm":
                        m.fit(X_t, y[tr_idx],
                              eval_set=[(X_v, y[val_idx])],
                              callbacks=[lgb.early_stopping(50, verbose=False),
                                         lgb.log_evaluation(-1)])
                    elif hasattr(m, "early_stopping_rounds") and getattr(m, "early_stopping_rounds", None):
                        m.fit(X_t, y[tr_idx], eval_set=[(X_v, y[val_idx])], verbose=False)
                    else:
                        m.fit(X_t, y[tr_idx])
                    yh = m.predict(X_v)
                    fold_scores.append(calc_reg_score(y[val_idx], yh, metric=self.metric))
                trial.set_user_attr("feature_set", fs)
                trial.set_user_attr("model_params", merged)
                return float(np.mean(fold_scores))

            sampler = optuna.samplers.TPESampler(seed=SEED)
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
            study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

            if name in warm_start:
                study.enqueue_trial(warm_start[name])

            timeout = self.per_model_timeout.get(name)
            with tqdm(total=n_tr, desc=f"HPO {name.upper():10s}", unit="trial", ncols=80) as pbar:
                def _cb(study, trial, _pbar=pbar):
                    _pbar.update(1)
                    try:
                        _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                    except ValueError:
                        pass
                study.optimize(objective, n_trials=n_tr, timeout=timeout,
                               callbacks=[_cb], catch=(Exception,))

            recs = []
            for t in study.trials:
                if t.value is not None and not (isinstance(t.value, float) and math.isnan(t.value)):
                    recs.append({
                        "model_name": name,
                        "feature_set": t.user_attrs.get("feature_set", "ts_tabular"),
                        "params": t.user_attrs.get("model_params", {}),
                        "score": t.value,
                    })
            recs.sort(key=lambda x: x["score"], reverse=(direction == "maximize"))
            top = recs[:self.top_k]
            all_configs.extend(top)
            if top:
                best = top[0]["score"]
                print(f"  [HPO] {name.upper()} best {self.metric.upper()} = {best:.4f}  "
                      f"feature_set = {top[0]['feature_set']}")
            else:
                print(f"  [HPO] {name.upper()} 無有效 trial")
        return all_configs


# ── DL 回歸 HPO（TCN / PatchTST / TSNet-train） ─────────────────────────────

class DLRegHPO:
    """時序回歸 DL 模型 HPO。fold-0 評估，回傳 top_k 設定。"""

    def __init__(self, model_name: str, arch_fixed: dict = None,
                 n_trials: int = 6, top_k: int = 1, device: str = None,
                 metric: str = "rmse"):
        assert model_name in ("tcn", "patchtst", "tsnet"), f"Unsupported reg DL model: {model_name}"
        self.model_name = model_name
        self.arch_fixed = arch_fixed   # 若提供則不搜架構（TSNet 用）
        self.n_trials = n_trials
        self.top_k = top_k
        self.device = device or DEVICE
        self.metric = metric

    def run(self, X, y, global_cfg: dict = None) -> list:
        global_cfg = global_cfg or {}
        folds = get_ts_folds(len(X), n_splits=5)
        tr_idx, val_idx = folds[0]
        direction = reg_metric_direction(self.metric)

        def objective(trial):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            fs = trial.suggest_categorical("feature_set", TS_DL_FEATURE_SETS)
            fb = FeatureBuilder(feature_set=fs, global_cfg=global_cfg)
            X_t = fb.fit_transform(X[tr_idx])
            X_v = fb.transform(X[val_idx])
            cur_in = X_t.shape[1]

            train_p = _reg_dl_train_space(trial)

            if self.arch_fixed is not None:
                arch_p = self.arch_fixed
            elif self.model_name == "tcn":
                arch_p = _tcn_arch_space(trial)
            else:  # patchtst
                arch_p = _patchtst_arch_space(trial, cur_in)

            try:
                score = train_reg_dl_single_fold(
                    self.model_name, arch_p, train_p,
                    X_t, y[tr_idx], X_v, y[val_idx],
                    device=self.device, metric=self.metric,
                )
            except (RuntimeError, Exception) as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    score = train_reg_dl_single_fold(
                        self.model_name, arch_p, train_p,
                        X_t, y[tr_idx], X_v, y[val_idx],
                        device="cpu", metric=self.metric,
                    )
                else:
                    raise

            trial.set_user_attr("feature_set", fs)
            trial.set_user_attr("arch_params", arch_p)
            trial.set_user_attr("train_params", train_p)
            return score

        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction=direction, sampler=sampler)
        label = self.model_name.upper()
        with tqdm(total=self.n_trials, desc=f"HPO {label:11s}", unit="trial", ncols=80) as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                try:
                    _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                except ValueError:
                    pass
            study.optimize(objective, n_trials=self.n_trials, callbacks=[_cb], catch=(Exception,))

        recs = []
        for t in study.trials:
            if t.value is not None and not (isinstance(t.value, float) and math.isnan(t.value)):
                recs.append({
                    "model_name": self.model_name,
                    "feature_set": t.user_attrs.get("feature_set", "raw"),
                    "arch_params": t.user_attrs.get("arch_params", {}),
                    "train_params": t.user_attrs.get("train_params", {}),
                    "score": t.value,
                })
        recs.sort(key=lambda x: x["score"], reverse=(direction == "maximize"))
        top = recs[:self.top_k]
        if top:
            print(f"  [HPO] {label} best {self.metric.upper()} = {top[0]['score']:.4f}  "
                  f"feature_set = {top[0]['feature_set']}")
        else:
            print(f"  [HPO] {label} 無有效 trial")
        return top


# ── 5-Fold CV ────────────────────────────────────────────────────────────────

def run_reg_tabular_cv(config, X, y, X_test, device=None, tag=None, save_artifacts=True,
                       global_cfg=None, metric="rmse"):
    device = device or DEVICE
    global_cfg = global_cfg or {}
    tag = tag or f"{config['model_name']}_{config['feature_set']}"
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)
    oof_counts = np.zeros(n, dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    folds = get_ts_folds(n, n_splits=5)
    fold_pbar = tqdm(enumerate(folds), total=len(folds),
                     desc=f"  CV {tag[:22]:22s}", ncols=90, leave=False)
    for fi, (tr_idx, val_idx) in fold_pbar:
        fb = FeatureBuilder(feature_set=config["feature_set"], global_cfg=global_cfg)
        X_t = fb.fit_transform(X[tr_idx])
        X_v = fb.transform(X[val_idx])
        X_te = fb.transform(X_test)

        m = build_reg_tabular_model(config["model_name"], config["params"], device=device)
        if config["model_name"] == "lgbm":
            m.fit(X_t, y[tr_idx], eval_set=[(X_v, y[val_idx])],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        elif hasattr(m, "early_stopping_rounds") and getattr(m, "early_stopping_rounds", None):
            m.fit(X_t, y[tr_idx], eval_set=[(X_v, y[val_idx])], verbose=False)
        else:
            m.fit(X_t, y[tr_idx])

        oof[val_idx] += m.predict(X_v)
        oof_counts[val_idx] += 1
        test_preds += m.predict(X_te) / len(folds)
        fold_pbar.set_postfix({f"fold_{metric}": f"{calc_reg_score(y[val_idx], m.predict(X_v), metric=metric):.4f}"})

    mask = oof_counts > 0
    oof[mask] /= oof_counts[mask]
    oof_score = calc_reg_score(y[mask], oof[mask], metric=metric)
    print(f"  [CV] {tag:30s} OOF {metric.upper()} = {oof_score:.4f}")

    if save_artifacts:
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_oof.npy"), oof)
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_test.npy"), test_preds)
    return oof, test_preds, oof_counts


def run_reg_dl_cv(config, X, y, X_test, device=None, tag=None, save_artifacts=True,
                  global_cfg=None, metric="rmse"):
    device = device or DEVICE
    global_cfg = global_cfg or {}
    tag = tag or f"{config['model_name']}_{config['feature_set']}"
    n = len(y)
    oof = np.zeros(n, dtype=np.float32)
    oof_counts = np.zeros(n, dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)

    train_p = config.get("train_params") or {}
    arch_p = config.get("arch_params") or {}
    bs = int(train_p.get("batch_size", 128))
    t_max = int(train_p.get("t_max", 10))
    n_epochs = int(train_p.get("n_epochs", 30))
    patience = int(train_p.get("patience", 7))

    direction = reg_metric_direction(metric)
    folds = get_ts_folds(n, n_splits=5)
    for fi, (tr_idx, val_idx) in enumerate(
        tqdm(folds, desc=f"  CV {tag[:22]:22s}", ncols=90, leave=False)
    ):
        torch.manual_seed(SEED + fi)
        fb = FeatureBuilder(feature_set=config["feature_set"], global_cfg=global_cfg)
        X_t = fb.fit_transform(X[tr_idx])
        X_v = fb.transform(X[val_idx])
        X_te = fb.transform(X_test)

        in_features = X_t.shape[1]
        model = _build_reg_dl_model(config["model_name"], arch_p, in_features).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_p.get("lr", 1e-3),
            weight_decay=train_p.get("weight_decay", 1e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        criterion = nn.MSELoss()
        loader = _make_reg_loader(X_t, y[tr_idx], batch_size=bs, shuffle=True)
        X_v_t = torch.tensor(X_v, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

        best = -np.inf if direction == "maximize" else np.inf
        cnt = 0
        best_state = None
        for _ in range(n_epochs):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze(-1)
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            model.eval()
            with torch.no_grad():
                yp = model(X_v_t).squeeze(-1).cpu().numpy()
            score = calc_reg_score(y[val_idx], yp, metric=metric)
            improved = (score > best) if direction == "maximize" else (score < best)
            if improved:
                best = score
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                cnt = 0
            else:
                cnt += 1
                if cnt >= patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        with torch.no_grad():
            oof[val_idx] += model(X_v_t).squeeze(-1).cpu().numpy()
            oof_counts[val_idx] += 1
            test_preds += model(X_te_t).squeeze(-1).cpu().numpy() / len(folds)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    mask = oof_counts > 0
    oof[mask] /= oof_counts[mask]
    oof_score = calc_reg_score(y[mask], oof[mask], metric=metric)
    print(f"  [CV] {tag:30s} OOF {metric.upper()} = {oof_score:.4f}")

    if save_artifacts:
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_oof.npy"), oof)
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_test.npy"), test_preds)
    return oof, test_preds, oof_counts


def run_reg_cv(config, X, y, X_test, device=None, tag=None, save_artifacts=True,
               global_cfg=None, metric="rmse"):
    dl = {"tsnet", "tcn", "patchtst"}
    if config["model_name"] in dl:
        return run_reg_dl_cv(config, X, y, X_test, device, tag, save_artifacts, global_cfg, metric)
    return run_reg_tabular_cv(config, X, y, X_test, device, tag, save_artifacts, global_cfg, metric)


# ── Ensemble A: 加權算術平均 Blender（最小化 RMSE / 最大化 R²） ────────────

class NelderMeadRegBlender:
    def __init__(self, n_restarts: int = 2, metric: str = "rmse"):
        self.n_restarts = n_restarts
        self.metric = metric
        self.weights_: np.ndarray = None

    def fit(self, oof_list: list, y: np.ndarray, oof_masks: list = None):
        """
        oof_list   : list of [N] OOF 回歸預測（TimeSeriesSplit 下前部會缺值，由 mask 排除）
        oof_masks  : list of [N] 0/1，標示該模型在哪些位置有預測；None 時自動以前 N//5 樣本截去
        """
        n_models = len(oof_list)
        N = len(y)
        oof_arr = np.stack([o.ravel() for o in oof_list], axis=1)  # [N, M]
        # 共同遮罩：所有模型皆有預測的位置
        if oof_masks is not None:
            masks = np.stack([m.ravel() > 0 for m in oof_masks], axis=1)
            common = masks.all(axis=1)
        else:
            common = np.ones(N, dtype=bool)
            common[: N // 5] = False  # TimeSeriesSplit fold-0 之前的部分無 OOF
        y_eff = y[common]
        oof_eff = oof_arr[common]
        direction = reg_metric_direction(self.metric)
        sign = -1.0 if direction == "maximize" else 1.0  # minimize 統一視角

        def score_func(logit_w):
            w = softmax(logit_w)
            blended = oof_eff @ w
            return sign * calc_reg_score(y_eff, blended, metric=self.metric) * (-1 if direction == "maximize" else 1)
            # 等價：minimize 時直接回傳 score；maximize 時回傳 -score
            # 上式為了清楚分支寫法

        # 簡化：直接寫
        def loss_fn(logit_w):
            w = softmax(logit_w)
            blended = oof_eff @ w
            s = calc_reg_score(y_eff, blended, metric=self.metric)
            return -s if direction == "maximize" else s

        print(f"  [Blend] Nelder-Mead search ({self.n_restarts} restarts, {n_models} models, metric={self.metric}) ...")
        best_val, best_x = np.inf, np.zeros(n_models)
        for r in tqdm(range(self.n_restarts), desc="  Blend restarts", ncols=70):
            x0 = np.random.default_rng(SEED + r).normal(0, 0.5, n_models)
            res = minimize(loss_fn, x0, method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-5, "fatol": 1e-5})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        self.weights_ = softmax(best_x)
        oof_score = -best_val if direction == "maximize" else best_val
        print(f"  [Blend] Best OOF {self.metric.upper()} = {oof_score:.4f}")
        print(f"  [Blend] Weights: {np.round(self.weights_, 3).tolist()}")
        return self

    def predict(self, test_list: list) -> np.ndarray:
        arr = np.stack([t.ravel() for t in test_list], axis=1)
        return arr @ self.weights_


# ── Ensemble B: Meta-Learner Stacking（回歸） ───────────────────────────────

class MetaLearnerRegStacker:
    def __init__(self, n_meta_trials: int = 10, n_folds: int = 5,
                 metric: str = "rmse", n_samples: int = 1000):
        self.n_meta_trials = n_meta_trials
        self.n_folds = n_folds
        self.metric = metric
        self.n_samples = n_samples
        self.meta_model_ = None
        self.meta_name_: str = None
        self._x_scaler = None

    def _build_meta(self, oof_list, X_orig=None):
        meta = np.stack([o.ravel() for o in oof_list], axis=1)  # [N, M]
        if X_orig is not None:
            meta = np.hstack([meta, X_orig])
        return meta

    def fit(self, oof_list, y, X_orig=None, oof_masks=None):
        # 找共同有預測的位置
        N = len(y)
        if oof_masks is not None:
            masks = np.stack([m.ravel() > 0 for m in oof_masks], axis=1)
            common = masks.all(axis=1)
        else:
            common = np.ones(N, dtype=bool)
            common[: N // 5] = False

        X_orig_scaled = None
        if X_orig is not None:
            self._x_scaler = StandardScaler()
            X_orig_scaled = self._x_scaler.fit_transform(X_orig)

        X_meta = self._build_meta(oof_list, X_orig_scaled)
        X_meta_eff = X_meta[common]
        y_eff = y[common]

        print(f"  [Stack] Meta-feature shape: {X_meta_eff.shape}  (effective)")

        direction = reg_metric_direction(self.metric)
        cv = KFold(n_splits=min(self.n_folds, max(2, len(y_eff) // 20)), shuffle=False)
        # 注意：時序資料原則不要 shuffle，但 meta features 來自 OOF（已折疊過時序資訊），這裡保守不 shuffle

        # 小樣本強制 Ridge
        if len(y_eff) < 2000:
            candidates = ["ridge"]
        else:
            candidates = ["ridge", "lgbm"]

        def objective(trial):
            meta_type = trial.suggest_categorical("meta_type", candidates)
            if meta_type == "lgbm":
                params = {
                    "num_leaves": trial.suggest_int("num_leaves", 8, 64),
                    "learning_rate": trial.suggest_float("lr", 0.01, 0.3, log=True),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                }
                model_fn = lambda p: lgb.LGBMRegressor(**p, random_state=SEED, n_jobs=-1, verbose=-1)
            else:
                params = {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}
                model_fn = lambda p: Ridge(**p, random_state=SEED)

            scores = []
            for tr_i, val_i in cv.split(X_meta_eff):
                m = model_fn(params)
                m.fit(X_meta_eff[tr_i], y_eff[tr_i])
                yh = m.predict(X_meta_eff[val_i])
                scores.append(calc_reg_score(y_eff[val_i], yh, metric=self.metric))
            trial.set_user_attr("meta_type", meta_type)
            trial.set_user_attr("params", params)
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction=direction, sampler=sampler)
        with tqdm(total=self.n_meta_trials, desc="  MetaHPO       ", ncols=70, unit="trial") as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                try:
                    _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                except ValueError:
                    pass
            study.optimize(objective, n_trials=self.n_meta_trials, callbacks=[_cb], catch=(Exception,))

        try:
            best_t = study.best_trial
        except ValueError:
            # 全部失敗時 fall back 到 Ridge(alpha=1.0)
            self.meta_model_ = Ridge(alpha=1.0, random_state=SEED)
            self.meta_model_.fit(X_meta_eff, y_eff)
            self.meta_name_ = "ridge_fallback"
            print("  [Stack] Meta-HPO 全失敗，使用 Ridge(alpha=1.0) fallback")
            return self

        name = best_t.user_attrs["meta_type"]
        params = best_t.user_attrs["params"]
        score = best_t.value
        print(f"  [Stack] Best meta-learner = {name}  OOF {self.metric.upper()} = {score:.4f}")
        if name == "lgbm":
            self.meta_model_ = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbose=-1)
        else:
            self.meta_model_ = Ridge(**params, random_state=SEED)
        self.meta_model_.fit(X_meta_eff, y_eff)
        self.meta_name_ = name
        return self

    def predict(self, test_list, X_orig=None):
        X_orig_scaled = None
        if X_orig is not None and self._x_scaler is not None:
            X_orig_scaled = self._x_scaler.transform(X_orig)
        X_meta = self._build_meta(test_list, X_orig_scaled)
        return self.meta_model_.predict(X_meta)


# ── 結果容器 ──────────────────────────────────────────────────────────────────

class PipelineRegResult:
    def __init__(self, test_blend, test_stack, all_oof, all_test, model_tags, blender, stacker):
        self.test_blend = test_blend
        self.test_stack = test_stack
        self.all_oof = all_oof
        self.all_test = all_test
        self.model_tags = model_tags
        self.blender = blender
        self.stacker = stacker


# ── 主引擎：回歸 ─────────────────────────────────────────────────────────────

def run_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    cfg: dict,
    budget: TimeBudget,
    *,
    skip_tabular: bool = False,
    skip_dl: bool = False,
    artifacts_dir: str = ARTIFACTS_DIR,
    metric: str = "rmse",
) -> PipelineRegResult:
    """時序回歸完整 pipeline。"""
    os.makedirs(artifacts_dir, exist_ok=True)
    cfg = dict(cfg)
    cfg["is_timeseries"] = True

    y_train = np.asarray(y_train, dtype=np.float32).ravel()
    direction = reg_metric_direction(metric)

    # RobustScaler uses median/IQR instead of mean/std, so extreme outliers in training
    # (e.g. SodiumConcentration has 5 values >1000 vs median~35) do not distort the scale.
    _y_scaler = RobustScaler()
    y_scaled = _y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    _center = _y_scaler.center_[0]
    _scale  = _y_scaler.scale_[0]
    print(f"  [TargetScale] median={_center:.4g}  IQR={_scale:.4g}"
          f"  range=[{y_train.min():.4g}, {y_train.max():.4g}]")

    all_oof = []
    all_test = []
    all_masks = []
    model_tags = []
    tabular_configs = []
    dl_configs = []
    scout_best_params: dict = {}
    scout_scores: dict = {}

    # ── [2] Tabular HPO ─────────────────────────────────────────────────────
    if not skip_tabular:
        print(f"\n[2a] Tabular Scout ({cfg['scout_trials']} trials/model, 3-Fold TS CV) ...")
        print(f"  [Budget] {budget.status_str()}")
        # CatBoost per-phase timeout（寬裕設定，防單一模型佔用過久）
        _catboost_timeout = {"catboost": 600}   # Scout: 10 min
        scout = TabularRegHPO(
            model_names=_ALL_REG_TABULAR_MODELS,
            n_trials=cfg["scout_trials"],
            top_k=1,
            metric=metric,
            per_model_timeout=_catboost_timeout,
        )
        scout_scores, scout_best_params = scout.scout(X_train, y_scaled,
                                                      scout_trials=cfg["scout_trials"],
                                                      global_cfg=cfg)
        # 篩選：依方向排序，保留前 2/3 且不偏離 best 太多
        def _sort_key(kv):
            v = kv[1]
            if isinstance(v, float) and math.isnan(v):
                return -np.inf if direction == "maximize" else np.inf
            return v if direction == "maximize" else -v

        ranked = sorted(scout_scores.items(), key=_sort_key, reverse=True)
        n_keep = math.ceil(len(ranked) * cfg["scout_ratio"])
        best_score = ranked[0][1] if ranked else float("nan")
        # 閾值定義（差距容忍 15%，回歸誤差更敏感）
        tol = 0.15
        if direction == "maximize":
            threshold = best_score * (1.0 - tol) if best_score >= 0 else best_score * (1.0 + tol)
            selected = [n for n, s in ranked[:n_keep] if not math.isnan(s) and s >= threshold]
        else:
            threshold = best_score * (1.0 + tol) if best_score >= 0 else best_score * (1.0 - tol)
            selected = [n for n, s in ranked[:n_keep] if not math.isnan(s) and s <= threshold]
        dropped = [n for n, _ in ranked if n not in selected]
        print(f"  [Scout] 排名: " + "  ".join(f"{n}={s:.4f}" for n, s in ranked))
        print(f"  [Scout] 保留 {len(selected)}/{len(ranked)}: {selected}  (淘汰: {dropped})")

        if not selected:
            print("  [Scout] 無模型通過篩選，退化使用全部模型")
            selected = [n for n, _ in ranked if not math.isnan(_)]

        # Full HPO
        if budget.should_skip(0.20):
            print("\n[2b] 時間預算緊迫，跳過 Tabular Full HPO")
            tabular_configs = []
        else:
            print(f"\n[2b] Tabular Full HPO ({cfg['tabular_trials']} trials/model, 5-Fold TS CV) ...")
            locked_fs = {
                name: params["feature_set"]
                for name, params in scout_best_params.items()
                if name in selected and "feature_set" in params
            }
            if locked_fs:
                print("  [FS Lock] " + "  ".join(f"{n}={fs}" for n, fs in locked_fs.items()))
            hpo = TabularRegHPO(
                model_names=selected,
                n_trials=cfg["tabular_trials"],
                top_k=cfg["tabular_top_k"],
                metric=metric,
                per_model_timeout={"catboost": 1200},  # Full HPO: 20 min
            )
            tabular_configs = hpo.run(X_train, y_scaled, global_cfg=cfg,
                                      warm_start=scout_best_params,
                                      locked_feature_sets=locked_fs)
    else:
        print("\n[2] 跳過 Tabular HPO（skip_tabular=True）")

    # ── [3-6] DL ─────────────────────────────────────────────────────────────
    if not skip_dl:
        # [4] TSNet 訓練 HPO（架構固定為 _DEFAULT_TSNET_ARCH）
        if budget.should_skip(0.20):
            print("\n[4] 時間預算緊迫，跳過 TSNet HPO")
            tsnet_configs = []
        else:
            print(f"\n[4] TSNet 訓練 HPO ({cfg['tsnet_trials']} trials, arch 固定) ...")
            tsnet_hpo = DLRegHPO(model_name="tsnet", arch_fixed=_DEFAULT_TSNET_ARCH,
                                 n_trials=cfg["tsnet_trials"], top_k=cfg["tsnet_top_k"],
                                 metric=metric)
            tsnet_configs = tsnet_hpo.run(X_train, y_scaled, global_cfg=cfg)

        # [5] TCN HPO
        if budget.should_skip(0.20):
            print("\n[5] 時間預算緊迫，跳過 TCN HPO")
            tcn_configs = []
        else:
            print(f"\n[5] TCN HPO ({cfg['dl_trials']} trials) ...")
            tcn_hpo = DLRegHPO(model_name="tcn", n_trials=cfg["dl_trials"],
                               top_k=cfg["dl_top_k"], metric=metric)
            tcn_configs = tcn_hpo.run(X_train, y_scaled, global_cfg=cfg)

        # [6] PatchTST HPO
        if budget.should_skip(0.20):
            print("\n[6] 時間預算緊迫，跳過 PatchTST HPO")
            patchtst_configs = []
        else:
            print(f"\n[6] PatchTST HPO ({cfg['dl_trials']} trials) ...")
            patchtst_hpo = DLRegHPO(model_name="patchtst", n_trials=cfg["dl_trials"],
                                    top_k=cfg["dl_top_k"], metric=metric)
            patchtst_configs = patchtst_hpo.run(X_train, y_scaled, global_cfg=cfg)

        dl_configs = tsnet_configs + tcn_configs + patchtst_configs
    else:
        print("\n[3-6] 跳過 DL（skip_dl=True）")

    # ── [7] 5-Fold CV ────────────────────────────────────────────────────────
    all_configs = tabular_configs + dl_configs
    if not all_configs and scout_best_params:
        # Full HPO 被 budget 截斷 → 退而使用 Scout 最佳參數直接做 CV
        print("  [Fallback] Full HPO 無結果，改用 Scout 最佳參數直接進行 CV")
        for model_name, params in scout_best_params.items():
            p = dict(params)
            fs = p.pop("feature_set", TS_TABULAR_FEATURE_SETS[0])
            all_configs.append({
                "model_name": model_name,
                "feature_set": fs,
                "params": p,
                "score": scout_scores.get(model_name, float("nan")),
            })
    if not all_configs:
        raise RuntimeError("沒有任何 model config 可供 CV！")

    print(f"\n[7] 5-Fold TimeSeries CV — {len(all_configs)} 個模型 config ...")
    for i, config in enumerate(all_configs):
        # _yrs suffix: robust-scaled predictions (RobustScaler); invalidates old StandardScaler cache
        tag = f"reg_{config['model_name']}_{config['feature_set']}_c{i}_yrs".replace("/", "_")
        oof_path = os.path.join(artifacts_dir, f"{tag}_oof.npy")
        tst_path = os.path.join(artifacts_dir, f"{tag}_test.npy")
        mask_path = os.path.join(artifacts_dir, f"{tag}_mask.npy")
        if os.path.exists(oof_path) and os.path.exists(tst_path) and os.path.exists(mask_path):
            print(f"  [CV] 載入快取 {tag}")
            oof = np.load(oof_path)
            test_pred = np.load(tst_path)
            counts = np.load(mask_path)
        else:
            oof, test_pred, counts = run_reg_cv(
                config, X_train, y_scaled, X_test,
                device=DEVICE, tag=tag, global_cfg=cfg, metric=metric,
            )
            np.save(mask_path, counts)
        all_oof.append(oof)
        all_test.append(test_pred)
        all_masks.append(counts)
        model_tags.append(tag)

    # ── Quality filter: drop models with catastrophically bad OOF R² ────────
    # R² is scale-invariant, so we can compute it directly on y_scaled predictions.
    _MIN_OOF_R2 = -2.0
    _keep = []
    for _i, (_oof_i, _mask_i) in enumerate(zip(all_oof, all_masks)):
        _common = _mask_i > 0
        _r2_i = r2_score(y_scaled[_common], _oof_i[_common]) if _common.sum() >= 2 else 0.0
        if _r2_i < _MIN_OOF_R2:
            print(f"  [QFilter] Dropping {model_tags[_i]} — OOF R²={_r2_i:.4f} (< {_MIN_OOF_R2})")
            _keep.append(False)
        else:
            _keep.append(True)
    if not all(_keep):
        all_oof    = [o for o, k in zip(all_oof,    _keep) if k]
        all_test   = [t for t, k in zip(all_test,   _keep) if k]
        all_masks  = [m for m, k in zip(all_masks,  _keep) if k]
        model_tags = [t for t, k in zip(model_tags, _keep) if k]
    if not all_oof:
        raise RuntimeError("所有 model config 被 quality filter 排除！")

    # ── [8] Ensemble A ───────────────────────────────────────────────────────
    print("\n[8] Ensemble A — Nelder-Mead Weighted Blending (RMSE-min) ...")
    blender = NelderMeadRegBlender(n_restarts=cfg["blend_restarts"], metric=metric)
    blender.fit(all_oof, y_scaled, oof_masks=all_masks)
    # Inverse-transform from scaled space back to original target scale
    test_blend = _y_scaler.inverse_transform(
        blender.predict(all_test).reshape(-1, 1)
    ).ravel()

    # ── [9] Ensemble B ───────────────────────────────────────────────────────
    print("\n[9] Ensemble B — Meta-Learner Stacking ...")
    stacker = MetaLearnerRegStacker(n_meta_trials=cfg["meta_trials"], metric=metric,
                                    n_samples=len(y_train))
    stacker.fit(all_oof, y_scaled, X_orig=X_train, oof_masks=all_masks)
    test_stack = _y_scaler.inverse_transform(
        stacker.predict(all_test, X_orig=X_test).reshape(-1, 1)
    ).ravel()

    # Inverse-transform OOF to original scale for result consistency
    all_oof = [_y_scaler.inverse_transform(o.reshape(-1, 1)).ravel() for o in all_oof]

    return PipelineRegResult(
        test_blend=test_blend,
        test_stack=test_stack,
        all_oof=all_oof,
        all_test=all_test,
        model_tags=model_tags,
        blender=blender,
        stacker=stacker,
    )


# ── 分類入口（委派給 pipeline.run） ──────────────────────────────────────────

def run_classification(*args, **kwargs):
    """時序分類直接委派給既有 pipeline.run。"""
    import pipeline as _pl
    kwargs["is_ts"] = True
    return _pl.run(*args, **kwargs)
