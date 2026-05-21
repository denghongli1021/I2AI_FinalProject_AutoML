# preprocessing/processors/time_processor.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    [時間特徵萃取器]
    自動將傳入的 Datetime 欄位拆解成：年、月、日、星期幾、是否為週末。
    若包含具體時間，還會進一步萃取：小時、分鐘。
    🆕 包含「週期性編碼 (Cyclical Encoding)」，保留時間的循環特性。
    """
    def __init__(self, extract_time: bool = True):
        self.extract_time = extract_time

    def fit(self, X, y=None):
        # 萃取特徵不需要學習任何參數，直接回傳自己
        return self

    def transform(self, X):
        # Scikit-learn 會傳入 numpy array，我們先轉回 DataFrame 方便操作
        df = pd.DataFrame(X)
        extracted_features = []

        for col in df.columns:
            # 🚀 AutoML 智慧時間解析機制
            try:
                # 策略 1：盲猜最符合國際標準、速度最快的 ISO8601 格式
                dt_series = pd.to_datetime(df[col], format='ISO8601', errors='coerce')
            except ValueError:
                # 策略 2：如果資料格式太髒或不統一，啟用 'mixed' 模式
                # 這會明確告訴 Pandas：「我知道格式很亂，請你處理，不要再噴警告了」
                dt_series = pd.to_datetime(df[col], format='mixed', errors='coerce')

            # 1. 基礎日期特徵萃取
            features = pd.DataFrame({
                f'year': dt_series.dt.year,
                f'month': dt_series.dt.month,
                f'day': dt_series.dt.day,
                f'dayofweek': dt_series.dt.dayofweek, # 0=星期一, 6=星期日
                # 💡 領域知識注入：是否為週末 (這對商業預測、資安異常登入都超級重要)
                f'is_weekend': dt_series.dt.dayofweek.isin([5, 6]).astype(float),
                
                # 🆕 3. 週期性特徵轉換 (Cyclical Encoding)
                # 月份的週期是 12
                f'month_sin': np.sin(2 * np.pi * dt_series.dt.month / 12.0),
                f'month_cos': np.cos(2 * np.pi * dt_series.dt.month / 12.0),
                # 星期幾的週期是 7
                f'dayofweek_sin': np.sin(2 * np.pi * dt_series.dt.dayofweek / 7.0),
                f'dayofweek_cos': np.cos(2 * np.pi * dt_series.dt.dayofweek / 7.0)
            })

            # 2. 進階時間特徵萃取 (如果資料精確到秒)
            if self.extract_time:
                features[f'hour'] = dt_series.dt.hour
                features[f'minute'] = dt_series.dt.minute
                
                # 🆕 小時的週期是 24
                features[f'hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24.0)
                features[f'hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24.0)

            extracted_features.append(features)

        # 將所有萃取出來的欄位水平合併，並轉回 numpy array
        return pd.concat(extracted_features, axis=1).values
    
    # 💡 新增這個方法：告訴系統我們切出了哪些時間欄位
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = ["datetime"]
        
        out_features = []
        for col in input_features:
            base = [
                f"{col}_year", f"{col}_month", f"{col}_day", 
                f"{col}_dayofweek", f"{col}_is_weekend",
                f"{col}_month_sin", f"{col}_month_cos",       # 🆕 新增對齊
                f"{col}_dayofweek_sin", f"{col}_dayofweek_cos"  # 🆕 新增對齊
            ]
            if self.extract_time:
                base.extend([
                    f"{col}_hour", f"{col}_minute",
                    f"{col}_hour_sin", f"{col}_hour_cos"      # 🆕 新增對齊
                ])
            out_features.extend(base)
        return np.array(out_features)


def build_time_pipeline(
    extract_time: bool = True,
    impute_strategy: str = "median"
) -> Pipeline:
    """
    建構日期時間 (Datetime) 的處理管線。
    
    流程：
    1. DatetimeFeatureExtractor：將字串拆解為 年/月/日/時/分 等多維數值。
    2. SimpleImputer：處理萃取後可能產生的 NaN (原先為亂碼或空值)。
    3. StandardScaler：對年份、月份等進行縮放，幫助神經網路與線性模型收斂。
    """
    return Pipeline([
        # 第一站：時間解剖刀
        ("extractor", DatetimeFeatureExtractor(extract_time=extract_time)),
        
        # 第二站：填補空缺 (注意：我們是萃取完再補中位數，這比補字串合理多了)
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        
        # 第三站：尺度縮放 (將 2026, 12 等數字拉回常態分佈範圍)
        ("scaler", StandardScaler())
    ])