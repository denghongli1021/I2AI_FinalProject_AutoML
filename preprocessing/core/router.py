# preprocessing/core/router.py
import pandas as pd
import numpy as np

class AutoRouter:
    """
    AutoML 系統的自動分類器。
    負責判定 DataFrame 中的每個欄位屬於哪種資料型態。
    """
    def __init__(self, categorical_threshold=10, text_length_threshold=15):
        # 如果獨立種類少於 50，視為類別特徵
        self.cat_threshold = categorical_threshold
        # 如果平均字串長度大於 20，視為純文字特徵
        self.text_threshold = text_length_threshold
        
        self.feature_groups = {
            "numeric": [],
            "categorical": [],
            "text": [],
            "datetime": []
        }

    def fit_predict(self, df: pd.DataFrame) -> dict:
        """
        掃描 DataFrame，回傳分類結果字典。
        """
        # 為了避免改動到原始資料，我們唯讀掃描
        for col in df.columns:
            dtype = df[col].dtype

            # 1. 時間序列 (Datetime)
            if pd.api.types.is_datetime64_any_dtype(dtype):
                self.feature_groups["datetime"].append(col)
                continue

            # 2. 數值型態 (Numeric)
            # 包括 int64, float64 等
            if pd.api.types.is_numeric_dtype(dtype):
                self.feature_groups["numeric"].append(col)
                continue

            # 3. 類別 vs 純文字 (Categorical vs Text)
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                unique_count = df[col].nunique()

                # 如果種類很少，絕對是類別 (例如：性別、城市)
                if unique_count <= self.cat_threshold:
                    self.feature_groups["categorical"].append(col)
                else:
                    # 檢查字串長度來判定是否為一段話 (Text)
                    # 隨機抽樣 100 筆來算長度比較有效率，避免資料太大卡住
                    valid_samples = df[col].dropna().astype(str)
                    sample_size = min(100, len(valid_samples))
                    
                    if sample_size == 0:
                        self.feature_groups["categorical"].append(col) # 全空的話暫時當類別
                        continue # 全是空值，可以直接忽略
                        
                    sample_texts = valid_samples.sample(n=sample_size, random_state=42)
                    avg_length = sample_texts.apply(len).mean()
                    
                    if avg_length > self.text_threshold:
                        self.feature_groups["text"].append(col) # 是純文字 (例如：HTTP Payload, 留言)
                    else:
                        # 種類多但長度短，通常是「高基數類別」(High Cardinality，如郵遞區號)
                        self.feature_groups["categorical"].append(col)

        print("[Auto-Router] 分類完成:", {k: len(v) for k, v in self.feature_groups.items()})
        return self.feature_groups