"""
超參數最佳化（Hyperparameter Optimization）模組
使用 Optuna TPE 取樣器搭配 MedianPruner，配合 5-fold 交叉驗證。
針對每個模型型別回傳前 5 組最佳設定。
"""
import warnings
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# 把 Optuna 預設的 INFO 訊息關掉，避免大量列印干擾
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ── 模型工廠：根據參數產生對應的 sklearn-API 模型 ──────────────────────────────

def _make_xgb_clf(params):
    """建立 XGBoost 分類器（**params 直接展開，搜尋空間擴充時不需改 factory）。"""
    return xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _make_xgb_reg(params):
    return xgb.XGBRegressor(
        **params,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _make_lgb_clf(params):
    return lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)


def _make_lgb_reg(params):
    return lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)


def _make_rf_clf(params):
    """建立 Random Forest 分類器。"""
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1,
    )


def _make_rf_reg(params):
    """建立 Random Forest 回歸器。"""
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        max_features=params["max_features"],
        random_state=42,
        n_jobs=-1,
    )


# ── 各模型型別的搜尋空間定義（trial 物件 → 參數字典）─────────────────────────────
# 參數範圍依 GBDT / RF 常用區間設計，learning_rate 與正則項採對數採樣
SEARCH_SPACES = {
    "xgb": lambda trial: {
        # 拉寬：高維資料常需更多樹與更深 / 更強正則
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
    },
    "lgb": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 5e-3, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
    },
    "rf": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    },
}

# 模型工廠對照表：MODEL_FACTORIES[name][task] -> 對應的建構函式
MODEL_FACTORIES = {
    "xgb": {"classification": _make_xgb_clf, "regression": _make_xgb_reg},
    "lgb": {"classification": _make_lgb_clf, "regression": _make_lgb_reg},
    "rf":  {"classification": _make_rf_clf,  "regression": _make_rf_reg},
}


class TPEOptimizer:
    """
    對每個模型型別執行 Optuna TPE 搜尋（搭配 MedianPruner 提早終止表現差的試驗）。
    回傳：每個模型的前 k 組 (params, score)。
    """

    def __init__(self, task="classification", n_trials=25, n_folds=5, top_k=5,
                 model_names=None, per_model_trials=None):
        """
        Args:
            n_trials: 預設每模型試驗次數（被 per_model_trials 覆寫）
            model_names: 要搜尋的模型清單，預設 ["xgb", "lgb", "rf"]
            per_model_trials: dict[model_name -> n_trials]，可單獨覆寫某模型試驗次數
        """
        self.task = task
        self.n_trials = n_trials
        self.n_folds = n_folds
        self.top_k = top_k
        self.model_names = model_names if model_names is not None else ["xgb", "lgb", "rf"]
        self.per_model_trials = per_model_trials or {}
        self.results_ = {}

    def optimize(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        對所有模型型別跑 HPO，回傳 dict[model_name] -> [(score, params), ...]。
        """
        # 分類用 accuracy、回歸用 neg_RMSE（Optuna 統一最大化）
        metric = "accuracy" if self.task == "classification" else "neg_root_mean_squared_error"
        cv = (
            StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            if self.task == "classification"
            else KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        )

        all_top_configs = {}

        for model_name in self.model_names:
            n_trials = int(self.per_model_trials.get(model_name, self.n_trials))
            if n_trials <= 0:
                continue
            print(f"  [HPO] Optimizing {model_name.upper()} ({n_trials} trials)...")
            trial_records = []

            def objective(trial):
                # 由搜尋空間取一組參數，建立模型，跑 CV，回傳平均分數給 Optuna
                params = SEARCH_SPACES[model_name](trial)
                factory = MODEL_FACTORIES[model_name][self.task]
                model = factory(params)
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                score = scores.mean()
                # 把參數記錄在 trial 屬性，後續才能取出對應的 params
                trial.set_user_attr("params", params)
                return score

            # MedianPruner：當前 trial 的中間結果若低於歷史中位數則提早砍掉
            pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(
                direction="maximize",
                pruner=pruner,
                sampler=sampler,
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            # 收集所有有效（非 pruned）試驗的成績
            for t in study.trials:
                if t.value is not None:
                    trial_records.append((t.value, t.user_attrs.get("params", {})))

            # 依分數由大到小排序，取前 top_k 組
            trial_records.sort(key=lambda x: x[0], reverse=True)
            top_configs = trial_records[: self.top_k]
            all_top_configs[model_name] = top_configs
            best_score = top_configs[0][0] if top_configs else float("nan")
            print(f"  [HPO] {model_name.upper()} best CV score: {best_score:.4f}")

        self.results_ = all_top_configs
        return all_top_configs
