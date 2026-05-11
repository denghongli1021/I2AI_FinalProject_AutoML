"""
超參數優化模組（HPO）。

- 傳統 Tabular 模型：TPE + 5-Fold CV，feature_set 也列入搜尋空間
- DL 模型（CNN1D / Transformer）：TPE 同時搜尋架構參數 + 訓練參數 + feature_set
- MLP：架構由 NAS 決定後，TPE 僅搜尋訓練參數 + feature_set
- 進度條：tqdm
- 所有數值型搜尋邊界在此定義；train.py / models/ 完全不人為固定任何超參數
"""
import warnings
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import SEED, N_SPLITS
from .preprocess import (
    FeatureBuilder,
    TABULAR_FEATURE_SETS,
    LINEAR_FEATURE_SETS,
    MLP_FEATURE_SETS,
    CNN_FEATURE_SETS,
    TRANSFORMER_FEATURE_SETS,
)
from .models.tabular import build_tabular_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── 各模型的超參數搜尋空間 ─────────────────────────────────────────────────────

def _tabular_space(name: str, trial: optuna.Trial, feat_sets: list) -> dict:
    """傳統 Tabular 模型搜尋空間（所有邊界不人為固定於 train.py）。"""
    feature_set = trial.suggest_categorical("feature_set", feat_sets)
    params = {"feature_set": feature_set}

    if name == "lgbm":
        params.update({
            "num_leaves": trial.suggest_int("num_leaves", 16, 512, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        })
    elif name == "xgb":
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        })
    elif name == "catboost":
        params.update({
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 100, 300),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        })
    elif name == "rf":
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
        })
    elif name == "logreg":
        params.update({
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["lbfgs", "saga"]
            ),
            "penalty": "l2",
        })
        feat_sets = LINEAR_FEATURE_SETS
        params["feature_set"] = trial.suggest_categorical("feature_set", feat_sets)
    elif name == "svm":
        params.update({
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 10.0, log=True),
            "kernel": "rbf",
        })
        feat_sets = LINEAR_FEATURE_SETS
        params["feature_set"] = trial.suggest_categorical("feature_set", feat_sets)

    return params


def _dl_train_space(trial: optuna.Trial) -> dict:
    """DL 訓練超參數搜尋空間（所有值由 HPO 決定）。"""
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.5),
        "mixup_prob": trial.suggest_float("mixup_prob", 0.0, 1.0),
        "t_max": trial.suggest_int("t_max", 5, 50),
        "n_epochs": trial.suggest_int("n_epochs", 20, 100),
        "patience": trial.suggest_int("patience", 5, 20),
    }


def _cnn_arch_space(trial: optuna.Trial) -> dict:
    """CNN1D 架構搜尋空間。"""
    return {
        "n_blocks": trial.suggest_int("n_blocks", 1, 5),
        "channels": trial.suggest_categorical("channels", [32, 64, 128, 256]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7, 9]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


def _transformer_arch_space(trial: optuna.Trial, in_features: int) -> dict:
    """SignalTransformer 架構搜尋空間。"""
    # d_model 必須能被 n_heads 整除
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    n_heads_choices = [h for h in [2, 4, 8] if d_model % h == 0]
    n_heads = trial.suggest_categorical("n_heads", n_heads_choices)
    # patch_size 不超過 in_features
    ps_choices = [p for p in [8, 16, 32, 64] if p <= in_features]
    if not ps_choices:
        ps_choices = [in_features]
    return {
        "patch_size": trial.suggest_categorical("patch_size", ps_choices),
        "d_model": d_model,
        "n_heads": n_heads,
        "depth": trial.suggest_int("depth", 1, 8),
        "ff_dim": trial.suggest_categorical("ff_dim", [64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


# ── Tabular HPO ──────────────────────────────────────────────────────────────

class TabularHPO:
    """
    對指定的傳統 Tabular 模型列表執行 TPE HPO。
    objective 為 5-Fold CV Macro F1（OOF）。
    """

    def __init__(
        self,
        model_names: list = None,
        n_trials: int = 50,
        n_folds: int = N_SPLITS,
        top_k: int = 3,
        per_model_trials: dict = None,
        device: str = None,
    ):
        self.model_names = model_names or ["lgbm", "xgb", "catboost", "rf", "logreg", "svm"]
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.top_k = top_k
        self.per_model_trials = per_model_trials or {}
        from .config import DEVICE as _DEVICE
        self.device = device or _DEVICE

    def run(self, X: np.ndarray, y: np.ndarray) -> list:
        """回傳 list of config dict，每個 dict 含 model_name / feature_set / params / score。"""
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)
        all_configs = []

        for name in self.model_names:
            n_trials = int(self.per_model_trials.get(name, self.n_trials))
            if n_trials <= 0:
                continue

            fs_candidates = (
                LINEAR_FEATURE_SETS if name in ("logreg", "svm") else TABULAR_FEATURE_SETS
            )
            trial_records = []

            def objective(trial, _name=name, _fs=fs_candidates, _device=self.device):
                merged = _tabular_space(_name, trial, _fs)
                fs = merged.pop("feature_set")
                model_params = merged

                scores = []
                for tr_idx, val_idx in cv.split(X, y):
                    fb = FeatureBuilder(feature_set=fs)
                    X_tr = fb.fit_transform(X[tr_idx])
                    X_val = fb.transform(X[val_idx])
                    m = build_tabular_model(_name, model_params, device=_device)
                    m.fit(X_tr, y[tr_idx])
                    y_hat = m.predict(X_val)
                    scores.append(f1_score(y[val_idx], y_hat, average="macro", zero_division=0))

                score = float(np.mean(scores))
                trial.set_user_attr("feature_set", fs)
                trial.set_user_attr("model_params", model_params)
                return score

            sampler = optuna.samplers.TPESampler(seed=SEED)
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, pruner=pruner
            )

            with tqdm(total=n_trials, desc=f"HPO {name.upper():8s}", unit="trial", ncols=80) as pbar:
                def _cb(study, trial, _pbar=pbar):
                    _pbar.update(1)
                    if study.best_trial:
                        _pbar.set_postfix({"best_f1": f"{study.best_value:.4f}"})

                study.optimize(objective, n_trials=n_trials, callbacks=[_cb])

            for t in study.trials:
                if t.value is not None:
                    trial_records.append({
                        "model_name": name,
                        "feature_set": t.user_attrs.get("feature_set", "raw"),
                        "params": t.user_attrs.get("model_params", {}),
                        "score": t.value,
                    })

            trial_records.sort(key=lambda x: x["score"], reverse=True)
            top = trial_records[: self.top_k]
            all_configs.extend(top)
            best = top[0]["score"] if top else float("nan")
            print(f"  [HPO] {name.upper()} best Macro F1 = {best:.4f}  "
                  f"feature_set = {top[0]['feature_set'] if top else '-'}")

        return all_configs


# ── DL HPO（CNN1D & Transformer）───────────────────────────────────────────────

class DLHPO:
    """
    對 CNN1D 或 Transformer 同時搜尋架構參數 + 訓練參數 + feature_set。
    為加速評估，使用單一 fold（fold 0）做快速篩選。
    """

    def __init__(
        self,
        model_name: str,
        n_trials: int = 30,
        top_k: int = 2,
        n_classes: int = None,
        device: str = None,
    ):
        assert model_name in ("cnn1d", "transformer"), f"Unsupported DL model: {model_name}"
        self.model_name = model_name
        self.n_trials = n_trials
        self.top_k = top_k
        self.n_classes = n_classes
        self.device = device

    def run(self, X: np.ndarray, y: np.ndarray) -> list:
        from .config import DEVICE
        from .train import train_dl_single_fold

        device = self.device or DEVICE
        n_classes = self.n_classes or len(np.unique(y))
        in_features = X.shape[1]

        fs_candidates = (
            CNN_FEATURE_SETS if self.model_name == "cnn1d" else TRANSFORMER_FEATURE_SETS
        )

        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_splits = list(cv.split(X, y))
        tr_idx, val_idx = fold_splits[0]  # 快速評估用 fold-0

        trial_records = []

        def objective(trial):
            fs = trial.suggest_categorical("feature_set", fs_candidates)
            fb = FeatureBuilder(feature_set=fs)
            X_tr = fb.fit_transform(X[tr_idx])
            X_val = fb.transform(X[val_idx])
            cur_in = X_tr.shape[1]

            train_p = _dl_train_space(trial)

            if self.model_name == "cnn1d":
                arch_p = _cnn_arch_space(trial)
            else:
                arch_p = _transformer_arch_space(trial, cur_in)

            score = train_dl_single_fold(
                model_name=self.model_name,
                arch_params=arch_p,
                train_params=train_p,
                X_tr=X_tr,
                y_tr=y[tr_idx],
                X_val=X_val,
                y_val=y[val_idx],
                n_classes=n_classes,
                device=device,
            )
            trial.set_user_attr("feature_set", fs)
            trial.set_user_attr("arch_params", arch_p)
            trial.set_user_attr("train_params", train_p)
            return score

        sampler = optuna.samplers.TPESampler(seed=SEED)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )
        name_label = self.model_name.upper()

        with tqdm(total=self.n_trials, desc=f"HPO {name_label:11s}", unit="trial", ncols=80) as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                if study.best_trial:
                    _pbar.set_postfix({"best_f1": f"{study.best_value:.4f}"})

            study.optimize(objective, n_trials=self.n_trials, callbacks=[_cb])

        for t in study.trials:
            if t.value is not None:
                trial_records.append({
                    "model_name": self.model_name,
                    "feature_set": t.user_attrs.get("feature_set", "raw"),
                    "arch_params": t.user_attrs.get("arch_params", {}),
                    "train_params": t.user_attrs.get("train_params", {}),
                    "score": t.value,
                })

        trial_records.sort(key=lambda x: x["score"], reverse=True)
        top = trial_records[: self.top_k]
        best = top[0]["score"] if top else float("nan")
        print(f"  [HPO] {name_label} best Macro F1 = {best:.4f}  "
              f"feature_set = {top[0]['feature_set'] if top else '-'}")
        return top


# ── MLP 訓練參數 HPO（架構由 NAS 傳入）─────────────────────────────────────────

class MLPTrainHPO:
    """在 NAS 搜尋出最佳架構後，對訓練超參數 + feature_set 做 TPE 搜尋。"""

    def __init__(self, arch_params: dict, n_trials: int = 30, top_k: int = 2, device: str = None):
        self.arch_params = arch_params
        self.n_trials = n_trials
        self.top_k = top_k
        self.device = device

    def run(self, X: np.ndarray, y: np.ndarray, n_classes: int) -> list:
        from .config import DEVICE
        from .train import train_dl_single_fold

        device = self.device or DEVICE
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_splits = list(cv.split(X, y))
        tr_idx, val_idx = fold_splits[0]

        trial_records = []

        def objective(trial):
            fs = trial.suggest_categorical("feature_set", MLP_FEATURE_SETS)
            fb = FeatureBuilder(feature_set=fs)
            X_tr = fb.fit_transform(X[tr_idx])
            X_val = fb.transform(X[val_idx])
            train_p = _dl_train_space(trial)

            score = train_dl_single_fold(
                model_name="mlp",
                arch_params=self.arch_params,
                train_params=train_p,
                X_tr=X_tr,
                y_tr=y[tr_idx],
                X_val=X_val,
                y_val=y[val_idx],
                n_classes=n_classes,
                device=device,
            )
            trial.set_user_attr("feature_set", fs)
            trial.set_user_attr("train_params", train_p)
            return score

        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with tqdm(total=self.n_trials, desc="HPO MLP-train  ", unit="trial", ncols=80) as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                if study.best_trial:
                    _pbar.set_postfix({"best_f1": f"{study.best_value:.4f}"})

            study.optimize(objective, n_trials=self.n_trials, callbacks=[_cb])

        for t in study.trials:
            if t.value is not None:
                trial_records.append({
                    "model_name": "mlp",
                    "feature_set": t.user_attrs.get("feature_set", "raw"),
                    "arch_params": self.arch_params,
                    "train_params": t.user_attrs.get("train_params", {}),
                    "score": t.value,
                })

        trial_records.sort(key=lambda x: x["score"], reverse=True)
        top = trial_records[: self.top_k]
        best = top[0]["score"] if top else float("nan")
        print(f"  [HPO] MLP-train best Macro F1 = {best:.4f}  "
              f"feature_set = {top[0]['feature_set'] if top else '-'}")
        return top
