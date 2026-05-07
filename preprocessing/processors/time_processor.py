# preprocessing/processors/time_processor.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class DatetimeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # 初始化一個空清單，用來存放轉換後的特徵名稱
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        # 決定轉換後會產生哪些欄位名稱
        # X 可能是 DataFrame 或 NumPy Array，我們統一轉為 DataFrame 處理
        X_df = pd.DataFrame(X)
        self.feature_names_out_ = []
        for col in X_df.columns:
            self.feature_names_out_.extend([
                f'{col}_year', f'{col}_month', f'{col}_day', 
                f'{col}_dayofweek', f'{col}_is_weekend'
            ])
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        out_df = pd.DataFrame()
        
        for col in X_df.columns:
            dt_series = pd.to_datetime(X_df[col], errors='coerce')
            
            # 實作轉換邏輯
            out_df[f'{col}_year'] = dt_series.dt.year
            out_df[f'{col}_month'] = dt_series.dt.month
            out_df[f'{col}_day'] = dt_series.dt.day
            out_df[f'{col}_dayofweek'] = dt_series.dt.dayofweek
            out_df[f'{col}_is_weekend'] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)
            
        return out_df.fillna(-1).values # 回傳 values 確保格式統一

    # ⭐ 關鍵修復：定義這個方法，讓 Scikit-learn 找得到欄位名稱
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

def build_time_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ('extractor', DatetimeExtractor())
    ])