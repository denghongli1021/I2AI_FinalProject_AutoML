"""
Tabular 模型工廠（LGBM / XGB / CatBoost / RF / LogReg / SVM）。
所有超參數由外部傳入（來自 HPO），此模組不預設任何數值。
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except ImportError:
    _CATBOOST_OK = False


def build_tabular_model(name: str, params: dict, seed: int = 42, device: str = "cpu"):
    """
    依 name 與 params 建立 sklearn-API 分類器。
    params 直接來自 HPO，不在此固定任何值。
    device="cuda" 時對支援的模型啟用 GPU 加速。
    """
    use_gpu = device == "cuda"

    if name == "lgbm":
        # LightGBM GPU 對寬特徵資料（1000+）比 CPU 慢，固定用 CPU
        return lgb.LGBMClassifier(
            **params,
            random_state=seed,
            n_jobs=-2,
            verbose=-1,
        )
    if name == "xgb":
        return xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=seed,
            verbosity=0,
            **({"device": "cuda"} if use_gpu else {"n_jobs": -2}),
        )
    if name == "catboost":
        if not _CATBOOST_OK:
            raise ImportError("catboost not installed")
        # CatBoost GPU 對寬特徵資料（1000+）顯存需求過大，固定用 CPU
        return CatBoostClassifier(
            **params,
            random_seed=seed,
            verbose=0,
            thread_count=max(1, (os.cpu_count() or 2) - 1),
        )
    if name == "rf":
        return RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=-2,
        )
    if name == "logreg":
        return LogisticRegression(
            **params,
            random_state=seed,
            max_iter=2000,
            n_jobs=-2,
        )
    if name == "svm":
        return SVC(**params, probability=True, random_state=seed)
    raise ValueError(f"Unknown tabular model: {name}")


TABULAR_MODEL_NAMES = ["lgbm", "xgb", "catboost", "rf", "logreg", "svm"]
