"""
Tabular 模型工廠（LGBM / XGB / CatBoost / RF / LogReg / SVM）。
所有超參數由外部傳入（來自 HPO），此模組不預設任何數值。
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    _CATBOOST_OK = True
except ImportError:
    _CATBOOST_OK = False


def build_tabular_model(
    name: str, params: dict, seed: int = 42, device: str = "cpu",
    class_weight=None,
):
    """
    依 name 與 params 建立 sklearn-API 分類器。
    params 直接來自 HPO，不在此固定任何值。
    device="cuda" 時對支援的模型啟用 GPU 加速。
    class_weight='balanced' 預設對不平衡類別自動加權，可傳 None 關閉。
    """
    use_gpu = device == "cuda"

    if name == "lgbm":
        # LightGBM GPU 對寬特徵資料（1000+）比 CPU 慢，固定用 CPU
        return lgb.LGBMClassifier(
            **params,
            random_state=seed,
            n_jobs=-2,
            verbose=-1,
            class_weight=class_weight,
        )
    if name == "xgb":
        return xgb.XGBClassifier(
            **params,
            use_label_encoder=False,
            early_stopping_rounds=100,
            random_state=seed,
            verbosity=0,
            class_weight=class_weight,
            **({"device": "cuda"} if use_gpu else {"n_jobs": -2}),
        )
    if name == "catboost":
        if not _CATBOOST_OK:
            raise ImportError("catboost not installed")
        # CatBoost GPU 對寬特徵資料（1000+）顯存需求過大，固定用 CPU
        # od_type="Iter" 為 train-loss 早停，不需 eval_set，避免跑滿 iterations 而超時
        # used_ram_limit / max_ctr_complexity / boosting_type=Plain：寬特徵下避免 bad_alloc
        cat_kwargs = {}
        if class_weight == "balanced":
            cat_kwargs["auto_class_weights"] = "Balanced"
        return CatBoostClassifier(
            **params,
            random_seed=seed,
            verbose=0,
            thread_count=max(1, (os.cpu_count() or 2) - 1),
            od_type="Iter",
            od_wait=30,
            used_ram_limit="4gb",
            max_ctr_complexity=2,
            boosting_type="Plain",
            **cat_kwargs,
        )
    if name == "rf":
        return RandomForestClassifier(
            **params,
            random_state=seed,
            n_jobs=-2,
            class_weight=class_weight,
        )
    if name == "logreg":
        return LogisticRegression(
            **params,
            random_state=seed,
            max_iter=2000,
            n_jobs=1,  # avoid BrokenProcessPool on Windows when PyTorch CUDA DLLs are loaded
            class_weight=class_weight,
        )
    if name == "svm":
        return SVC(**params, probability=True, random_state=seed, class_weight=class_weight)
    if name == "extra_trees":
        return ExtraTreesClassifier(
            **params,
            random_state=seed,
            n_jobs=-2,
            class_weight=class_weight,
        )
    if name == "knn":
        # KNN 不支援 random_state 與 class_weight；需要 pca/svd 降維特徵才高效
        return KNeighborsClassifier(**params, n_jobs=-2)
    raise ValueError(f"Unknown tabular model: {name}")


TABULAR_MODEL_NAMES = ["lgbm", "xgb", "catboost", "rf", "logreg", "svm", "extra_trees", "knn"]
