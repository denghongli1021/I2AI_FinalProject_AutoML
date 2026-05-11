"""
AutoML 主流程：串接「特徵工程 → 超參數搜尋 → 集成學習」三大模組。

整體流程（fit）：
    1. FeatureExtractionPipeline：MI 特徵選擇 / 多項式交互 / 群組聚合 / 時序統計 / 標準化 / SHAP 剪枝
    2. TPEOptimizer：對 XGB / LGB / RF 各自跑 Optuna TPE，取每個模型 top-k 組超參數
    3. StackingEnsemble：L1 OOF 預測 + L2 meta-learner 堆疊
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .feature_extraction import FeatureExtractionPipeline
from .hpo import TPEOptimizer
from .ensemble import StackingEnsemble

# 屏蔽訓練過程中外部套件（sklearn / xgboost / lightgbm 等）噴出的雜訊警告
warnings.filterwarnings("ignore")


class AutoMLPipeline:
    """
    端到端 AutoML 主流程：
        1. 特徵工程（MI / Poly / GroupAgg / TS / Scaling / SHAP pruning）
        2. 超參數最佳化（Optuna TPE，每個模型保留前 5 組）
        3. 神經架構搜尋 + 堆疊集成（L1 OOF + L2 meta-learner）
    """

    def __init__(
        self,
        task: str = "classification",
        is_timeseries: bool = False,
        n_hpo_trials: int = 25,
        n_folds: int = 5,
        top_k_hpo: int = 5,
        use_nas: bool = True,
        mi_k: int = 50,
        poly_max_cols: int = 15,
        use_shap_pruning: bool = True,
        # ── 新增 (集成相關) ──
        top_k_ensemble: int = 3,           # L1 每個樹模型取 HPO 前幾組進入 stacking
        n_nas_seeds: int = 1,              # NAS 多 seed 集成數
        use_extratrees: bool = True,       # L1 加入 ExtraTrees
        use_knn: bool = True,              # L1 加入 KNN
        hpo_models=None,                   # HPO 跑哪些模型，預設 ["xgb","lgb","rf"]
        per_model_trials=None,             # dict 覆寫單模型試驗次數
    ):
        self.task = task
        self.is_timeseries = is_timeseries
        self.n_hpo_trials = n_hpo_trials
        self.n_folds = n_folds
        self.top_k_hpo = top_k_hpo
        self.use_nas = use_nas
        self.mi_k = mi_k
        self.poly_max_cols = poly_max_cols
        self.use_shap_pruning = use_shap_pruning
        self.top_k_ensemble = top_k_ensemble
        self.n_nas_seeds = n_nas_seeds
        self.use_extratrees = use_extratrees
        self.use_knn = use_knn
        self.hpo_models = hpo_models
        self.per_model_trials = per_model_trials

        # --- 訓練後才會被填入的內部物件 ---
        self.fe_pipeline_ = None     # 特徵工程 pipeline（fit 後保存以供 transform 使用）
        self.hpo_configs_ = None     # HPO 結果：dict[model_name] -> [(score, params), ...]
        self.ensemble_ = None        # L1 + L2 堆疊集成器
        self.label_encoder_ = None   # 分類任務專用的 y 編碼器

    # ------------------------------------------------------------------
    def _prepare_y(self, y: pd.Series):
        """將標籤轉為模型可訓練的數值表示（分類任務做 LabelEncoder，回歸轉 float64）。"""
        if self.task == "classification":
            self.label_encoder_ = LabelEncoder()
            return self.label_encoder_.fit_transform(y.astype(str))
        return y.values.astype(np.float64)

    def _decode_y(self, y_enc: np.ndarray):
        """將模型輸出（已編碼的整數類別）還原回原始標籤。"""
        if self.task == "classification" and self.label_encoder_:
            return self.label_encoder_.inverse_transform(y_enc.astype(int))
        return y_enc

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AutoMLPipeline":
        """主訓練流程：依序執行特徵工程 → HPO → 堆疊集成訓練。"""

        # === 步驟 1：特徵工程 ===
        print("\n[Pipeline] === Feature Extraction ===")
        self.fe_pipeline_ = FeatureExtractionPipeline(
            task=self.task,
            is_timeseries=self.is_timeseries,
            mi_k=self.mi_k,
            poly_max_cols=self.poly_max_cols,
            use_shap_pruning=self.use_shap_pruning,
        )
        X_feat = self.fe_pipeline_.fit_transform(X, y)
        y_enc = self._prepare_y(y)

        # 統一轉成 float32 並把 NaN/Inf 換成 0，確保下游模型不會因為非法值報錯
        X_arr = X_feat.values.astype(np.float32)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # === 步驟 2：超參數最佳化 ===
        print("\n[Pipeline] === Hyperparameter Optimization ===")
        optimizer = TPEOptimizer(
            task=self.task,
            n_trials=self.n_hpo_trials,
            n_folds=self.n_folds,
            top_k=self.top_k_hpo,
            model_names=self.hpo_models,
            per_model_trials=self.per_model_trials,
        )
        self.hpo_configs_ = optimizer.optimize(X_arr, y_enc)

        # === 步驟 3：架構搜尋 + 堆疊集成 ===
        print("\n[Pipeline] === Architecture Search & Ensemble Training ===")
        self.ensemble_ = StackingEnsemble(
            task=self.task,
            n_folds=self.n_folds,
            top_k_configs=self.top_k_ensemble,
            use_nas=self.use_nas,
            n_nas_seeds=self.n_nas_seeds,
            use_extratrees=self.use_extratrees,
            use_knn=self.use_knn,
        )
        self.ensemble_.fit(X_arr, y_enc, self.hpo_configs_)

        return self

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """對外暴露的特徵轉換工具：將原始欄位轉為訓練時相同的特徵空間。"""
        X_feat = self.fe_pipeline_.transform(X)
        arr = X_feat.values.astype(np.float32)
        # 與 fit 階段相同：消除 NaN / Inf
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """端到端預測：自動完成特徵轉換 → 集成預測 → 標籤反編碼。"""
        X_arr = self.transform_features(X)
        y_enc = self.ensemble_.predict(X_arr)
        return self._decode_y(y_enc)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """分類任務專用：回傳各類別的機率（shape = [n_samples, n_classes]）。"""
        if self.task != "classification":
            raise ValueError("predict_proba only for classification")
        X_arr = self.transform_features(X)
        return self.ensemble_.predict_proba(X_arr)
