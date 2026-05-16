# preprocessing/processors/category_processor.py
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def build_category_pipeline(
    impute_strategy: str = "most_frequent",
    fill_value: str = "missing",
    handle_unknown: str = "ignore"
) -> Pipeline:
    """
    建構低基數類別特徵 (Low-Cardinality Categorical) 的處理管線。
    
    流程：
    1. SimpleImputer：填補缺失值 (預設為眾數，也可選填固定字串)。
    2. OneHotEncoder：獨熱編碼，將類別展開為二元欄位 (0 或 1)。
    
    參數:
    - impute_strategy: 'most_frequent' (補眾數) 或 'constant' (補固定字串)
    - fill_value: 當 strategy 為 'constant' 時要補的字串，預設為 'missing'
    - handle_unknown: 遇到測試集中沒看過的新類別時的處理方式 ('ignore' 或 'error')
    """
    steps = []
    
    # ==========================================
    # 第一站：缺失值填補 (Imputation)
    # ==========================================
    if impute_strategy == "constant":
        # 有時候「不填寫」本身就是一種強烈的情感/特徵，所以保留為獨立類別 'missing'
        steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=fill_value)))
    else:
        # 預設：哪種人最多，就當作那種人 (例如：大家都填 'Taipei'，空值就補 'Taipei')
        steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        
    # ==========================================
    # 第二站：獨熱編碼 (One-Hot Encoding)
    # ==========================================
    steps.append(('encoder', OneHotEncoder(
        # 💡 MLOps 終極防線：遇到沒看過的類別，直接忽略變全 0，千萬不能報錯！
        handle_unknown=handle_unknown,
        # 因為 Router 已經確保這裡是低基數，不會維度爆炸，所以直接輸出 Dense 矩陣，下游更好處理
        sparse_output=False
    )))
    
    return Pipeline(steps)