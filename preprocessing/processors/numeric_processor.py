# preprocessing/processors/numeric_processor.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def build_numeric_pipeline() -> Pipeline:
    """
    建立數值型特徵的處理管線。
    步驟 1: 填補缺失值 (使用中位數，抵抗離群值干擾)
    步驟 2: 標準化 (Z-score scaling，讓均值為 0，變異數為 1)
    """
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    return numeric_pipeline

# 未來進階版預留：
# 當你們進入第二週，可以把 SimpleImputer 換成 IterativeImputer (MICE)
# 或者加入 PowerTransformer 來處理極度偏態的分佈。