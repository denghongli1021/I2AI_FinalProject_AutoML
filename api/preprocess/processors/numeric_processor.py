# preprocessing/processors/numeric_processor.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer

# ⚠️ 必須先 import 這個實驗性標籤，才能使用 Scikit-learn 的 IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class _SafeFinitizer(BaseEstimator, TransformerMixin):
    """
    把 inf / 超過 float32 上限的值清乾淨,給下游 sklearn step 一個保證 finite 的輸入。

    為什麼需要:PowerTransformer(yeo-johnson) 套到 test 資料上,當 test 有超出 train
    分布範圍的極端值時,公式 ((x+1)^lambda - 1) / lambda 會溢位成 ±inf,然後 PT 內建的
    StandardScaler (或外接的下游 transformer) 一律會 check_array(force_all_finite=True)
    拋 "Input X contains infinity or a value too large for dtype('float32')"。

    順序:
      ① inf → NaN
      ② clip 殘餘超大有限值到 float32 安全範圍
      ③ NaN → 0 (PT 後面是 StandardScaler,NaN 也會被拒;設 0 = 中性值)
    """
    def fit(self, X, y=None):
        # 記下輸入維度,讓 get_feature_names_out 在 input_features=None 時也能 fallback
        try:
            self.n_features_in_ = int(np.asarray(X).shape[1])
        except Exception:
            self.n_features_in_ = None
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        F32_MAX = float(np.finfo(np.float32).max)
        X_arr = np.where(np.isinf(X_arr), np.nan, X_arr)
        X_arr = np.clip(X_arr, -F32_MAX, F32_MAX)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=F32_MAX, neginf=-F32_MAX)
        return X_arr

    def get_feature_names_out(self, input_features=None):
        """
        sklearn Pipeline 串接時會呼叫,SafeFinitizer 不改變欄位數 → 原樣回傳。
        若上游沒給 input_features,用 fit 階段記下的 n_features_in_ 生 f0/f1/... fallback。
        """
        if input_features is not None:
            return np.asarray(input_features)
        n = getattr(self, "n_features_in_", None)
        if n is None:
            return np.asarray([])
        return np.asarray([f"f{i}" for i in range(n)])


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
        # 【inf 防禦】關掉 PT 內建 standardize,改外接 StandardScaler。
        # PT 內建會在 yeo-johnson 之後馬上跑 StandardScaler.check_array,
        # 而 yeo-johnson 對 test 極端值會產生 inf → 內建 scaler 直接拋。
        # 拆成「PT(無 standardize) → SafeFinitizer → 外部 StandardScaler」,
        # SafeFinitizer 中間接住 inf 不讓下游炸。
        steps.append(('power_transform', PowerTransformer(method='yeo-johnson', standardize=False)))
        steps.append(('finitize_after_pt', _SafeFinitizer()))

    # ==========================================
    # 第三站：尺度縮放 (Scaling)
    # ==========================================
    if scaler_type == "standard":
        steps.append(('scaler', StandardScaler()))
    elif scaler_type == "robust":
        # 使用四分位距縮放，對極端值不敏感
        steps.append(('scaler', RobustScaler()))

    # 若有 PT 但沒有外部 scaler,補一個 StandardScaler 補上原本 PT 內建的功能,
    # 避免行為變動。
    if handle_outliers and scaler_type == "none":
        steps.append(('scaler_after_pt', StandardScaler()))

    if len(steps) == 0:
        return 'passthrough'

    return Pipeline(steps=steps)