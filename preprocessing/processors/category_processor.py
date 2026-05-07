# preprocessing/processors/category_processor.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def build_category_pipeline() -> Pipeline:
    """
    建立類別型特徵的處理管線。
    步驟 1: 填補缺失值 (使用眾數)
    步驟 2: One-Hot Encoding (獨熱編碼)
    """
    category_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' 是防止未來預測時遇到沒看過的類別而當機
        # sparse_output=False 確保輸出是密集的 DataFrame，方便後續 XGBoost 食用
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return category_pipeline

# 未來進階版預留：
# 如果發現某些欄位種類太多 (High Cardinality)，
# 可以把 OneHotEncoder 換成 TargetEncoder 或 HashEncoder 避免記憶體爆炸。