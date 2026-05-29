# preprocessing/processors/category_processor.py
<<<<<<< HEAD
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

=======
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e

def build_category_pipeline(
    impute_strategy: str = "most_frequent",
    fill_value: str = "missing",
<<<<<<< HEAD
    handle_unknown: str = "infrequent_if_exist", # 🚀 配合原廠低頻處理
    min_freq: float = 0.01,
) -> Pipeline:
    """
    建構低基數類別特徵（DL 軌道）的處理管線。

    流程：
    1. SimpleImputer：填補缺失值。
    2. OneHotEncoder (內建低頻聚合)：
       - min_frequency=0.01: 自動將出現頻率低於 1% 的類別合併成一個 "infrequent" 欄位。
       - max_categories=50: 強制封頂，就算頻率都 > 1%，最多也只保留前 50 大類別，防爆維度。
    """
    if impute_strategy == "constant":
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy='most_frequent')

    return Pipeline([
        ('imputer', imputer),
        # 🚀 終極型態的 OneHotEncoder：自帶頻率過濾與數量封頂！
        ('encoder', OneHotEncoder(
            handle_unknown=handle_unknown, 
            sparse_output=False,
            min_frequency=min_freq,    # 完全取代你的 RareCategoryGrouper
            max_categories=50          # 雙重保險：強制維度不超過 50
        )),
    ])


def build_tree_category_pipeline(min_freq: float = 0.01) -> Pipeline:
    """
    建構低基數類別特徵（Tree 軌道）的處理管線。
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # OrdinalEncoder 在新版 sklearn 也支援 min_frequency 與 max_categories
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1,
            min_frequency=min_freq, # 讓決策樹也能享受低頻合併的好處 (減少無效的分支)
        )),
    ])
=======
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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
