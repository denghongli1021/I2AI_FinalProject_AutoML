# preprocessing/processors/category_processor.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def build_category_pipeline(
    impute_strategy: str = "most_frequent",
    fill_value: str = "missing",
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
