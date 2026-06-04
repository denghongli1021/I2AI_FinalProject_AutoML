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
from sklearn.model_selection import StratifiedKFold, train_test_split
from .metrics import calculate_score, get_metric_name
from tqdm import tqdm

from .config import SEED
from .data import get_ts_folds
from .preprocess import (
    FeatureBuilder,
    TABULAR_FEATURE_SETS,
    CATBOOST_FEATURE_SETS,
    LINEAR_FEATURE_SETS,
    MLP_FEATURE_SETS,
    CNN_FEATURE_SETS,
    TRANSFORMER_FEATURE_SETS,
    TS_TABULAR_FEATURE_SETS,
    TS_DL_FEATURE_SETS,
)
from .models.tabular import build_tabular_model

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── Scout Phase 經驗預設值（各模型合理起點，確保第一個 trial 不隨機浪費）──────────
_SCOUT_DEFAULTS: dict = {
    "lgbm": {
        "feature_set": "raw",
        "num_leaves": 64,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "xgb": {
        "feature_set": "raw",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    },
    "catboost": {
        "feature_set": "raw",
        "depth": 6,
        "learning_rate": 0.05,
        "iterations": 200,
        "l2_leaf_reg": 3.0,
        "bagging_temperature": 0.5,
    },
    "rf": {
        "feature_set": "raw",
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
    },
    "logreg": {
        "feature_set": "pca64",
        "C": 1.0,
        "solver": "lbfgs",
    },
    "svm": {
        "feature_set": "pca64",
        "C": 1.0,
        "gamma": 0.01,
    },
    "extra_trees": {
        "feature_set": "raw",
        "n_estimators": 200,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
    },
    "knn": {
        "feature_set": "pca64",
        "n_neighbors": 7,
        "weights": "distance",
        "metric": "euclidean",
    },
}


# ── 各模型的超參數搜尋空間 ─────────────────────────────────────────────────────

def _tabular_space(name: str, trial: optuna.Trial, feat_sets: list,
                   global_cfg: dict = None) -> dict:
    """傳統 Tabular 模型搜尋空間（所有邊界不人為固定於 train.py）。"""
    global_cfg = global_cfg or {}
    is_fast = global_cfg.get("is_fast", False)
    feature_set = trial.suggest_categorical("feature_set", feat_sets)
    params = {"feature_set": feature_set}

    if name == "lgbm":
        n_est_max = 500 if is_fast else 2000
        params.update({
            "num_leaves": trial.suggest_int("num_leaves", 16, 512, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, n_est_max),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        })
    elif name == "xgb":
        n_est_max = 500 if is_fast else 2000
        params.update({
            "max_depth": trial.suggest_int("max_depth", 3, 10 if is_fast else 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, n_est_max),
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
            "iterations": trial.suggest_int("iterations", 100, 500),
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
    elif name == "svm":
        params.update({
            "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-5, 10.0, log=True),
            "kernel": "rbf",
        })
    elif name == "extra_trees":
        # ExtraTrees：決策邊界與 RF/Boosting 不同，提供正交視角
        params.update({
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]
            ),
        })
    elif name == "knn":
        # KNN：非參數化，與樹模型決策邊界完全不同
        params.update({
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 30),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "chebyshev"]),
        })

    return params


def _dl_train_space(trial: optuna.Trial) -> dict:
    """DL 訓練超參數搜尋空間（CNN / MLP 用）。"""
    n_epochs = trial.suggest_int("n_epochs", 20, 50)
    # t_max ∈ [n_epochs//2, n_epochs]，確保 CosineAnnealingLR 最多 2 個週期
    t_max = trial.suggest_int("t_max", max(5, n_epochs // 2), n_epochs)
    return {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.5),
        "mixup_prob": trial.suggest_float("mixup_prob", 0.0, 1.0),
        "t_max": t_max,
        "n_epochs": n_epochs,
        "patience": trial.suggest_int("patience", 5, 10),
    }


def _transformer_train_space(trial: optuna.Trial) -> dict:
    """Transformer / PatchTST 專用訓練超參數（更多 epoch、更低 lr）。"""
    n_epochs = trial.suggest_int("n_epochs", 40, 120)
    # t_max ∈ [n_epochs//2, n_epochs]，確保 CosineAnnealingLR 最多 2 個週期
    t_max = trial.suggest_int("t_max", max(15, n_epochs // 2), n_epochs)
    return {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.15),
        "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.4),
        "mixup_prob": trial.suggest_float("mixup_prob", 0.0, 0.8),
        "t_max": t_max,
        "n_epochs": n_epochs,
        "patience": trial.suggest_int("patience", 8, 15),
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
    # d_model 從 128 起，確保對大特徵維度有足夠容量；n_heads 移除 2（太小）
    d_model    = trial.suggest_categorical("d_model", [128, 256, 512])
    n_heads    = trial.suggest_categorical("n_heads", [4, 8])
    patch_size = trial.suggest_categorical("patch_size", [8, 16, 32, 64])
    # norm_first: True=Pre-Norm（深層/大資料穩定），False=Post-Norm（淺層精度高），HPO 自動選擇
    norm_first = trial.suggest_categorical("norm_first", [True, False])
    return {
        "patch_size": min(patch_size, max(1, in_features)),
        "d_model": d_model,
        "n_heads": n_heads,
        "depth": trial.suggest_int("depth", 2, 6),
        "ff_dim": trial.suggest_categorical("ff_dim", [256, 512, 1024, 2048]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3),
        "norm_first": norm_first,
    }


def _resnet_arch_space(trial: optuna.Trial) -> dict:
    """ResNet1D_18 架構搜尋空間（固定深度，只搜 channels 與 dropout）。"""
    return {
        "channels": trial.suggest_categorical("channels", [32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
    }


def _tcn_arch_space(trial: optuna.Trial) -> dict:
    """TCN 架構搜尋空間（因果擴張卷積，不含未來資訊）。"""
    return {
        "n_blocks": trial.suggest_int("n_blocks", 2, 6),
        "channels": trial.suggest_categorical("channels", [32, 64, 128, 256]),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
    }


def _patchtst_arch_space(trial: optuna.Trial, in_features: int) -> dict:
    """PatchTST 架構搜尋空間（Mean-Pooling，無 CLS Token 過擬合問題）。"""
    d_model    = trial.suggest_categorical("d_model", [64, 128, 256])
    n_heads    = trial.suggest_categorical("n_heads", [2, 4, 8])
    patch_size = trial.suggest_categorical("patch_size", [4, 8, 16, 32])
    return {
        "patch_size": min(patch_size, max(1, in_features)),
        "d_model": d_model,
        "n_heads": n_heads,
        "depth": trial.suggest_int("depth", 1, 6),
        "ff_dim": trial.suggest_categorical("ff_dim", [64, 128, 256, 512]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
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
        n_folds: int = 5,
        top_k: int = 3,
        per_model_trials: dict = None,
        per_model_timeout: dict = None,
        device: str = None,
        metric: str = "f1",
    ):
        self.model_names = model_names or ["lgbm", "xgb", "catboost", "rf", "logreg", "svm"]
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.top_k = top_k
        self.per_model_trials = per_model_trials or {}
        self.per_model_timeout = per_model_timeout or {}
        self.metric = metric
        from .config import DEVICE as _DEVICE
        self.device = device or _DEVICE

    def run(self, X: np.ndarray, y: np.ndarray, global_cfg: dict = None,
            warm_start: dict = None, locked_feature_sets: dict = None) -> list:
        """
        回傳 list of config dict，每個 dict 含 model_name / feature_set / params / score。

        warm_start : {model_name: params_dict}
            由 scout() 回傳的 best_params，注入為 Trial 0，讓 TPE 從強基線出發。
        locked_feature_sets : {model_name: feature_set_str}
            固定每個模型使用 Scout 找到的最佳 feature_set，不再重新搜尋。
            可讓有限 trial 數集中在超參數空間，顯著提升 HPO 效率。
        """
        global_cfg = global_cfg or {}
        warm_start = warm_start or {}
        locked_feature_sets = locked_feature_sets or {}
        n_repeats = global_cfg.get("n_repeats", 1)
        is_ts = global_cfg.get("is_timeseries", False)

        from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
        if is_ts:
            folds = get_ts_folds(len(X), n_splits=self.n_folds)
        elif n_repeats > 1:
            cv = RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=n_repeats, random_state=SEED)
            folds = list(cv.split(X, y))
        else:
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=SEED)
            folds = list(cv.split(X, y))

        # HPO 評估折數：預設與 n_folds 相同，可由 global_cfg["hpo_n_folds"] 縮短
        # （Final CV 仍使用完整 n_folds，此設定只影響 HPO objective 的評估速度）
        hpo_n_folds = min(len(folds), global_cfg.get("hpo_n_folds", len(folds)))
        hpo_folds = folds[:hpo_n_folds]

        # === 預先計算所有 feature_set × fold 組合，消除 objective 內的重複 FeatureBuilder ===
        _all_fs: set = set()
        for _n in self.model_names:
            _nt = int(self.per_model_trials.get(_n, self.n_trials))
            if _nt <= 0:
                continue
            _lk = locked_feature_sets.get(_n)
            if _lk:
                _all_fs.add(_lk)
            elif is_ts:
                _all_fs.update(TS_TABULAR_FEATURE_SETS)
            elif _n in ("logreg", "svm", "knn"):
                _all_fs.update(LINEAR_FEATURE_SETS)
            elif _n == "catboost":
                _all_fs.update(CATBOOST_FEATURE_SETS)
            else:
                _all_fs.update(TABULAR_FEATURE_SETS)
        print(f"  [HPO] Pre-computing {len(_all_fs)} feature set(s) × {hpo_n_folds} folds ...")
        _feat_cache: dict = {}
        for _fs in sorted(_all_fs):
            for _fi, (_tr, _vl) in enumerate(hpo_folds):
                _fb = FeatureBuilder(feature_set=_fs, global_cfg=global_cfg)
                _feat_cache[(_fs, _fi)] = (_fb.fit_transform(X[_tr]), _fb.transform(X[_vl]))

        all_configs = []

        for name in self.model_names:
            n_trials = int(self.per_model_trials.get(name, self.n_trials))
            if n_trials <= 0:
                continue

            if is_ts:
                fs_candidates = TS_TABULAR_FEATURE_SETS
            elif name in ("logreg", "svm", "knn"):
                fs_candidates = LINEAR_FEATURE_SETS
            elif name == "catboost":
                fs_candidates = CATBOOST_FEATURE_SETS
            else:
                fs_candidates = TABULAR_FEATURE_SETS
            trial_records = []
            # 若 Scout 已找到最佳 feature_set，固定使用（不再佔用 trial 搜尋維度）
            _locked_fs = locked_feature_sets.get(name)

            _cw = "balanced" if self.metric != "accuracy" else None
            _min_train = min(len(tr) for tr, _ in folds)

            def objective(trial, _name=name, _fs=fs_candidates, _device=self.device,
                          _locked=_locked_fs, _cw=_cw, _gcfg=global_cfg,
                          _min_train=_min_train):
                if _locked:
                    # 固定 feature_set，讓 TPE 專注在超參數空間
                    merged = _tabular_space(_name, trial, [_locked], global_cfg=_gcfg)
                else:
                    merged = _tabular_space(_name, trial, _fs, global_cfg=_gcfg)
                fs = merged.pop("feature_set")
                model_params = merged
                # KNN 防呆：n_neighbors 不得超過最小 fold 訓練樣本數
                if _name == "knn" and "n_neighbors" in model_params:
                    model_params["n_neighbors"] = max(1, min(model_params["n_neighbors"], _min_train - 1))

                scores = []
                for _fi, (tr_idx, val_idx) in enumerate(hpo_folds):
                    X_tr, X_val = _feat_cache[(fs, _fi)]
                    m = build_tabular_model(_name, model_params, device=_device, class_weight=_cw)
                    _unseen = set(np.unique(y[val_idx])) - set(np.unique(y[tr_idx]))
                    if _name == "lgbm":
                        import lightgbm as _lgb
                        if _unseen:
                            m.fit(X_tr, y[tr_idx], callbacks=[_lgb.log_evaluation(-1)])
                        else:
                            m.fit(X_tr, y[tr_idx],
                                  eval_set=[(X_val, y[val_idx])],
                                  callbacks=[_lgb.early_stopping(100, verbose=False),
                                             _lgb.log_evaluation(-1)])
                    elif hasattr(m, "early_stopping_rounds") and m.early_stopping_rounds:
                        if _unseen:
                            m.fit(X_tr, y[tr_idx])
                        else:
                            m.fit(X_tr, y[tr_idx], eval_set=[(X_val, y[val_idx])], verbose=False)
                    else:
                        m.fit(X_tr, y[tr_idx])
                    y_hat = m.predict(X_val)
                    scores.append(calculate_score(y[val_idx], y_hat, metric=self.metric))

                score = float(np.mean(scores))
                trial.set_user_attr("feature_set", fs)
                trial.set_user_attr("model_params", model_params)
                return score

            sampler = optuna.samplers.TPESampler(seed=SEED)
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            study = optuna.create_study(
                direction="maximize", sampler=sampler, pruner=pruner
            )

            # 暖啟動：注入 scout 找到的最佳參數作為 Trial 0，TPE 立即從強基線出發
            if name in warm_start:
                study.enqueue_trial(warm_start[name])

            model_timeout = self.per_model_timeout.get(name)
            with tqdm(total=n_trials, desc=f"HPO {name.upper():8s}", unit="trial", ncols=80) as pbar:
                def _cb(study, trial, _pbar=pbar):
                    _pbar.update(1)
                    try:
                        _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                    except ValueError:
                        # 尚無成功完成的 trial（例如全部 OOM/失敗）
                        pass

                study.optimize(objective, n_trials=n_trials, timeout=model_timeout,
                               callbacks=[_cb], catch=(Exception,))

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
            print(f"  [HPO] {name.upper()} best {get_metric_name(self.metric)} = {best:.4f}  "
                  f"feature_set = {top[0]['feature_set'] if top else '-'}")

        return all_configs

    def scout(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scout_trials: int = 5,
        val_size: float = 0.2,
        global_cfg: dict = None,
        scout_cv_folds: int = 3,
    ) -> tuple:
        """
        Phase 1 快速篩選：3-Fold mini-CV + 輕量化 HPO。

        使用 3-Fold CV（而非單次 holdout）確保 Scout 分數與 Full HPO 5-Fold 分數
        在同一度量體系下，讓淘汰閾值有意義。小型 CV 雖然慢約 3×，但有效消除
        lucky split 偏差，避免誤留弱模型或誤淘汰強模型。

        策略：
          - Trial 0：注入 _SCOUT_DEFAULTS（經驗預設值），保證不浪費在隨機差點
          - Trial 1+：RandomSampler 隨機探索
        回傳 ({model_name: best_f1}, {model_name: best_params})。
        """
        global_cfg = global_cfg or {}
        is_ts = global_cfg.get("is_timeseries", False)

        # 建立 Scout 用的 CV folds（分數體系與 Full HPO 一致）
        if is_ts:
            scout_folds = get_ts_folds(len(X), n_splits=scout_cv_folds)
        else:
            cv = StratifiedKFold(n_splits=scout_cv_folds, shuffle=True, random_state=SEED)
            scout_folds = list(cv.split(X, y))

        # === 預先計算 Scout 所有 feature_set × fold 組合 ===
        _scout_fs: set = set()
        for _n in self.model_names:
            if is_ts:
                _scout_fs.update(TS_TABULAR_FEATURE_SETS)
            elif _n in ("logreg", "svm", "knn"):
                _scout_fs.update(LINEAR_FEATURE_SETS)
            elif _n == "catboost":
                _scout_fs.update(CATBOOST_FEATURE_SETS)
            else:
                _scout_fs.update(TABULAR_FEATURE_SETS)
        _scout_cache: dict = {}
        for _fs in sorted(_scout_fs):
            for _fi, (_tr, _vl) in enumerate(scout_folds):
                _fb = FeatureBuilder(feature_set=_fs, global_cfg=global_cfg)
                _scout_cache[(_fs, _fi)] = (_fb.fit_transform(X[_tr]), _fb.transform(X[_vl]))

        scores = {}
        best_params = {}

        for name in self.model_names:
            if is_ts:
                fs_candidates = TS_TABULAR_FEATURE_SETS
            elif name in ("logreg", "svm", "knn"):
                fs_candidates = LINEAR_FEATURE_SETS
            elif name == "catboost":
                fs_candidates = CATBOOST_FEATURE_SETS
            else:
                fs_candidates = TABULAR_FEATURE_SETS

            _scout_cw = "balanced" if self.metric != "accuracy" else None

            def objective(trial, _name=name, _fs=fs_candidates, _scout_cw=_scout_cw,
                         _gcfg=global_cfg):
                merged = _tabular_space(_name, trial, _fs, global_cfg=_gcfg)
                fs = merged.pop("feature_set")
                fold_scores = []
                for _fi, (tr_idx, val_idx) in enumerate(scout_folds):
                    X_t, X_v = _scout_cache[(fs, _fi)]
                    m = build_tabular_model(_name, merged, device=self.device, class_weight=_scout_cw)
                    _unseen = set(np.unique(y[val_idx])) - set(np.unique(y[tr_idx]))
                    if _name == "lgbm":
                        import lightgbm as _lgb
                        if _unseen:
                            m.fit(X_t, y[tr_idx], callbacks=[_lgb.log_evaluation(-1)])
                        else:
                            m.fit(X_t, y[tr_idx],
                                  eval_set=[(X_v, y[val_idx])],
                                  callbacks=[_lgb.early_stopping(30, verbose=False),
                                             _lgb.log_evaluation(-1)])
                    elif hasattr(m, "early_stopping_rounds") and m.early_stopping_rounds:
                        if _unseen:
                            m.fit(X_t, y[tr_idx])
                        else:
                            m.fit(X_t, y[tr_idx], eval_set=[(X_v, y[val_idx])], verbose=False)
                    else:
                        m.fit(X_t, y[tr_idx])
                    y_hat = m.predict(X_v)
                    fold_scores.append(calculate_score(y[val_idx], y_hat, metric=self.metric))
                score = float(np.mean(fold_scores))
                trial.set_user_attr("feature_set", fs)
                return score

            # RandomSampler：少量 trial 下比 TPE 更穩定，不需要熱身期
            sampler = optuna.samplers.RandomSampler(seed=SEED)
            study = optuna.create_study(direction="maximize", sampler=sampler)

            # 注入經驗預設值作為 Trial 0，確保至少評估一組合理起點
            # 若預設的 feature_set 不在當前候選集（如 TS 模式只有 "raw"），改用候選集第一個
            if name in _SCOUT_DEFAULTS:
                default = dict(_SCOUT_DEFAULTS[name])
                if default.get("feature_set") not in fs_candidates:
                    default["feature_set"] = fs_candidates[0]
                study.enqueue_trial(default)

            with tqdm(total=scout_trials, desc=f"Scout {name.upper():8s}", unit="trial", ncols=80) as pbar:
                def _cb(study, trial, _pbar=pbar):
                    _pbar.update(1)
                    try:
                        _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                    except ValueError:
                        pass
                study.optimize(objective, n_trials=scout_trials, callbacks=[_cb],
                               catch=(Exception,))

            try:
                best_trial = study.best_trial
                best_val = study.best_value
            except ValueError:
                best_trial = None
                best_val = float("nan")
            scores[name] = best_val
            if best_trial is not None:
                p = dict(best_trial.params)
                # feature_set 可能不在 params（被 merged.pop 取走），從 user_attrs 補回
                if "feature_set" not in p:
                    fs_val = best_trial.user_attrs.get("feature_set")
                    if fs_val:
                        p["feature_set"] = fs_val
                best_params[name] = p
            print(f"  [Scout] {name.upper():8s} {scout_cv_folds}-Fold CV {get_metric_name(self.metric)} = {best_val:.4f}")

        return scores, best_params


# ── DL HPO（CNN1D & Transformer）───────────────────────────────────────────────

class DLHPO:
    """
    對 CNN1D 或 Transformer 同時搜尋架構參數 + 訓練參數 + feature_set。
    使用前 n_hpo_folds 個 fold 平均評分，減少 selection bias（預設 2 fold）。
    """

    def __init__(
        self,
        model_name: str,
        n_trials: int = 30,
        top_k: int = 2,
        n_classes: int = None,
        device: str = None,
        metric: str = "f1",
    ):
        _valid = {"cnn1d", "resnet1d", "transformer", "tcn", "patchtst"}
        assert model_name in _valid, f"Unsupported DL model: {model_name}"
        self.model_name = model_name
        self.n_trials = n_trials
        self.top_k = top_k
        self.n_classes = n_classes
        self.device = device
        self.metric = metric

    def run(self, X: np.ndarray, y: np.ndarray, global_cfg: dict = None) -> list:
        global_cfg = global_cfg or {}
        is_ts = global_cfg.get("is_timeseries", False)

        from .config import DEVICE
        from .train import train_dl_single_fold

        device = self.device or DEVICE
        n_classes = self.n_classes or len(np.unique(y))
        in_features = X.shape[1]

        # 時序模式下 DL 直接用 raw/signal（lag/rolling 已預先計算到 X）
        if is_ts:
            fs_candidates = TS_DL_FEATURE_SETS
        elif self.model_name in ("cnn1d", "resnet1d", "tcn"):
            fs_candidates = CNN_FEATURE_SETS
        else:
            fs_candidates = TRANSFORMER_FEATURE_SETS

        # HPO 評估：使用前 n_hpo_folds 個 fold 平均分數，降低 selection bias
        if is_ts:
            fold_splits = get_ts_folds(len(X), n_splits=5)
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            fold_splits = list(cv.split(X, y))
        n_hpo_folds = 2
        hpo_folds = fold_splits[:n_hpo_folds]

        trial_records = []

        def objective(trial):
            fs = trial.suggest_categorical("feature_set", fs_candidates)
            if self.model_name in ("transformer", "patchtst"):
                train_p = _transformer_train_space(trial)
            else:
                train_p = _dl_train_space(trial)

            scores = []
            arch_p = None
            for i, (tr_idx, val_idx) in enumerate(hpo_folds):
                fb = FeatureBuilder(feature_set=fs, global_cfg=global_cfg)
                X_tr = fb.fit_transform(X[tr_idx])
                X_val = fb.transform(X[val_idx])

                # arch space 只在第一個 fold 確定（in_features 跨 fold 穩定）
                if i == 0:
                    cur_in = X_tr.shape[1]
                    if self.model_name == "cnn1d":
                        arch_p = _cnn_arch_space(trial)
                    elif self.model_name == "resnet1d":
                        arch_p = _resnet_arch_space(trial)
                    elif self.model_name == "tcn":
                        arch_p = _tcn_arch_space(trial)
                    elif self.model_name == "patchtst":
                        arch_p = _patchtst_arch_space(trial, cur_in)
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
                    global_cfg=global_cfg,
                )
                scores.append(score)

            mean_score = float(np.mean(scores))
            trial.set_user_attr("feature_set", fs)
            trial.set_user_attr("arch_params", arch_p)
            trial.set_user_attr("train_params", train_p)
            return mean_score

        sampler = optuna.samplers.TPESampler(seed=SEED)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )
        name_label = self.model_name.upper()

        with tqdm(total=self.n_trials, desc=f"HPO {name_label:11s}", unit="trial", ncols=80) as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                try:
                    _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                except ValueError:
                    pass

            study.optimize(objective, n_trials=self.n_trials, callbacks=[_cb],
                           catch=(Exception,))

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
        print(f"  [HPO] {name_label} best {get_metric_name(self.metric)} = {best:.4f}  "
              f"feature_set = {top[0]['feature_set'] if top else '-'}")
        return top


# ── MLP 訓練參數 HPO（架構由 NAS 傳入）─────────────────────────────────────────

class MLPTrainHPO:
    """在 NAS 搜尋出最佳架構後，對訓練超參數 + feature_set 做 TPE 搜尋。"""

    def __init__(self, arch_params: dict, n_trials: int = 30, top_k: int = 2, device: str = None, metric: str = "f1"):
        self.arch_params = arch_params
        self.n_trials = n_trials
        self.top_k = top_k
        self.device = device
        self.metric = metric

    def run(self, X: np.ndarray, y: np.ndarray, n_classes: int, global_cfg: dict = None) -> list:
        global_cfg = global_cfg or {}
        from .config import DEVICE
        from .train import train_dl_single_fold

        device = self.device or DEVICE
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_splits = list(cv.split(X, y))
        tr_idx, val_idx = fold_splits[0]

        trial_records = []

        def objective(trial):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fs = trial.suggest_categorical("feature_set", MLP_FEATURE_SETS)
            fb = FeatureBuilder(feature_set=fs, global_cfg=global_cfg)
            X_tr = fb.fit_transform(X[tr_idx])
            X_val = fb.transform(X[val_idx])
            train_p = _dl_train_space(trial)

            _device = device
            try:
                score = train_dl_single_fold(
                    model_name="mlp",
                    arch_params=self.arch_params,
                    train_params=train_p,
                    X_tr=X_tr,
                    y_tr=y[tr_idx],
                    X_val=X_val,
                    y_val=y[val_idx],
                    n_classes=n_classes,
                    device=_device,
                    global_cfg=global_cfg,
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    score = train_dl_single_fold(
                        model_name="mlp",
                        arch_params=self.arch_params,
                        train_params=train_p,
                        X_tr=X_tr,
                        y_tr=y[tr_idx],
                        X_val=X_val,
                        y_val=y[val_idx],
                        n_classes=n_classes,
                        device="cpu",
                        global_cfg=global_cfg,
                    )
                else:
                    raise
            trial.set_user_attr("feature_set", fs)
            trial.set_user_attr("train_params", train_p)
            return score

        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with tqdm(total=self.n_trials, desc="HPO MLP-train  ", unit="trial", ncols=80) as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                try:
                    if study.best_trial:
                        _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                except ValueError:
                    pass

            study.optimize(objective, n_trials=self.n_trials, callbacks=[_cb],
                           catch=(Exception,))

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
        print(f"  [HPO] MLP-train best {get_metric_name(self.metric)} = {best:.4f}  "
              f"feature_set = {top[0]['feature_set'] if top else '-'}")
        return top


# ── TSNet 訓練參數 HPO（架構由 TSNASSearcher 傳入）────────────────────────────

class TSNetTrainHPO:
    """
    針對 TSNet 搜尋訓練超參數，輸出 model_name="tsnet" 的 config。

    使用 get_ts_folds（Walk-forward）切割，feature_set 從 TS_DL_FEATURE_SETS 搜尋。
    """

    def __init__(
        self,
        arch_params: dict,
        n_trials: int = 30,
        top_k: int = 2,
        device: str = None,
        metric: str = "f1",
    ):
        self.arch_params = arch_params
        self.n_trials = n_trials
        self.top_k = top_k
        self.device = device
        self.metric = metric

    def run(self, X: np.ndarray, y: np.ndarray, n_classes: int, global_cfg: dict = None) -> list:
        """
        回傳 list of config dict，每個 dict 含：
            model_name="tsnet", feature_set, arch_params, train_params, score
        """
        global_cfg = global_cfg or {}
        from .config import DEVICE
        from .train import train_dl_single_fold

        device = self.device or DEVICE

        # 時序切割：使用 Walk-forward 的第一個 fold 做快速評估
        fold_splits = get_ts_folds(len(X), n_splits=5)
        tr_idx, val_idx = fold_splits[0]

        trial_records = []

        def objective(trial):
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            fs = trial.suggest_categorical("feature_set", TS_DL_FEATURE_SETS)
            fb = FeatureBuilder(feature_set=fs, global_cfg=global_cfg)
            X_tr = fb.fit_transform(X[tr_idx])
            X_val = fb.transform(X[val_idx])
            train_p = _dl_train_space(trial)

            _device = device
            try:
                score = train_dl_single_fold(
                    model_name="tsnet",
                    arch_params=self.arch_params,
                    train_params=train_p,
                    X_tr=X_tr,
                    y_tr=y[tr_idx],
                    X_val=X_val,
                    y_val=y[val_idx],
                    n_classes=n_classes,
                    device=_device,
                    global_cfg=global_cfg,
                )
            except (RuntimeError, Exception) as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    import torch as _torch
                    _torch.cuda.empty_cache()
                    score = train_dl_single_fold(
                        model_name="tsnet",
                        arch_params=self.arch_params,
                        train_params=train_p,
                        X_tr=X_tr,
                        y_tr=y[tr_idx],
                        X_val=X_val,
                        y_val=y[val_idx],
                        n_classes=n_classes,
                        device="cpu",
                        global_cfg=global_cfg,
                    )
                else:
                    raise

            trial.set_user_attr("feature_set", fs)
            trial.set_user_attr("train_params", train_p)
            return score

        sampler = optuna.samplers.TPESampler(seed=SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        with tqdm(total=self.n_trials, desc="HPO TSNet-train ", unit="trial", ncols=80) as pbar:
            def _cb(study, trial, _pbar=pbar):
                _pbar.update(1)
                try:
                    if study.best_trial:
                        _pbar.set_postfix({f"best_{self.metric}": f"{study.best_value:.4f}"})
                except ValueError:
                    pass

            study.optimize(objective, n_trials=self.n_trials, callbacks=[_cb],
                           catch=(Exception,))

        for t in study.trials:
            if t.value is not None:
                trial_records.append({
                    "model_name": "tsnet",
                    "feature_set": t.user_attrs.get("feature_set", "raw"),
                    "arch_params": self.arch_params,
                    "train_params": t.user_attrs.get("train_params", {}),
                    "score": t.value,
                })

        trial_records.sort(key=lambda x: x["score"], reverse=True)
        top = trial_records[:self.top_k]
        best = top[0]["score"] if top else float("nan")
        print(f"  [HPO] TSNet-train best {get_metric_name(self.metric)} = {best:.4f}  "
              f"feature_set = {top[0]['feature_set'] if top else '-'}")
        return top
