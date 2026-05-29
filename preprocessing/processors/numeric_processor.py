# preprocessing/processors/numeric_processor.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

# ⚠️ 必須先 import 這個實驗性標籤，才能使用 Scikit-learn 的 IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

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
        steps.append(('power_transform', PowerTransformer(method='yeo-johnson')))

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