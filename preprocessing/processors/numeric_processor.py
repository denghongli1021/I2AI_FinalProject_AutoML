# preprocessing/processors/numeric_processor.py
import warnings
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

# ⚠️ 必須先 import 這個實驗性標籤，才能使用 Scikit-learn 的 IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class SafePowerTransformer(BaseEstimator, TransformerMixin):
    """PowerTransformer（Yeo-Johnson）的安全包裝。
    逐欄 fit；若某欄 scipy 最佳化失敗（找不到 valid bracket、常數欄等），
    自動退回 StandardScaler，確保整體管線不崩潰。
    """

    def __init__(self, method: str = "yeo-johnson"):
        self.method = method

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_cols = X.shape[1]
        self._transformers = []
        for i in range(n_cols):
            col = X[:, i].reshape(-1, 1)
            pt = PowerTransformer(method=self.method, copy=True)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")   # 把 warning 轉 exception，一起捕捉
                    pt.fit(col)
                self._transformers.append(pt)
            except Exception:
                # 退回 StandardScaler：對常數欄、極端分布欄安全
                fb = StandardScaler()
                fb.fit(col)
                self._transformers.append(fb)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for i, tf in enumerate(self._transformers):
            col = X[:, i].reshape(-1, 1)
            try:
                out[:, i] = tf.transform(col).ravel()
            except Exception:
                # transform 也可能失敗（推論期分布偏移），直接用 StandardScaler fallback
                fb = StandardScaler()
                out[:, i] = fb.fit_transform(col).ravel()
        return out

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

def build_numeric_pipeline(
    impute_strategy: str = "median", 
    scaler_type: str = "standard",
    handle_outliers: bool = False
) -> Pipeline:
    """
    建構數值型特徵的處理管線。
    
    參數:
    - impute_strategy: 'median', 'mean', 'knn', 'mice', 'none'
    - scaler_type: 'standard', 'robust', 'none'
    - handle_outliers: 是否啟動 PowerTransformer 來校正偏態與極端值
    """
    steps = []

    # ==========================================
    # 第一站：缺失值填補 (Imputation)
    # ==========================================
    if impute_strategy == "median":
        steps.append(('imputer', SimpleImputer(strategy='median')))
    elif impute_strategy == "mean":
        steps.append(('imputer', SimpleImputer(strategy='mean')))
    elif impute_strategy == "knn":
        steps.append(('imputer', KNNImputer(n_neighbors=5)))
    elif impute_strategy == "mice":
        # MICE 會使用 Ridge Regression 反覆預測缺失值，max_iter 控制收斂次數
        steps.append(('imputer', IterativeImputer(max_iter=10, random_state=42)))
    elif impute_strategy != "none":
        # 預設防呆機制
        steps.append(('imputer', SimpleImputer(strategy='median')))

    # ==========================================
    # 第二站：偏態與離群值校正 (Outlier Handling)
    # ==========================================
    if handle_outliers:
        # Yeo-Johnson 可以處理包含負數的資料，將其轉換為接近常態分佈
        # 使用 SafePowerTransformer：逐欄處理，優化失敗的欄位自動退回 StandardScaler
        steps.append(('power_transform', SafePowerTransformer(method='yeo-johnson')))

    # ==========================================
    # 第三站：尺度縮放 (Scaling)
    # ==========================================
    if scaler_type == "standard":
        steps.append(('scaler', StandardScaler()))
    elif scaler_type == "robust":
        # 使用四分位距縮放，對極端值不敏感
        steps.append(('scaler', RobustScaler()))
    
    if len(steps) == 0:
        return 'passthrough'
        
    return Pipeline(steps=steps)