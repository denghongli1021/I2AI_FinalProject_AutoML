# preprocessing/processors/category_processor.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    將出現頻率低於 min_freq 的低頻類別合併為 '__other__'。

    防止 OHE 為罕見類別（如 native-country 中只出現 1 次的國家）產生幾乎全為零的
    稀疏欄位，這類欄位對模型無意義且增加 overfitting 風險。
    """

    def __init__(self, min_freq: float = 0.01, fill_value: str = "__other__"):
        self.min_freq = min_freq
        self.fill_value = fill_value

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=object)
        self.keep_values_: dict = {}
        for i in range(X_arr.shape[1]):
            col = pd.Series(X_arr[:, i].ravel())
            counts = col.value_counts(normalize=True)
            self.keep_values_[i] = set(counts[counts >= self.min_freq].index)
        return self

    def transform(self, X, y=None):
        X_arr = np.asarray(X, dtype=object).copy()
        for i in range(X_arr.shape[1]):
            col = pd.Series(X_arr[:, i].ravel())
            rare_mask = ~col.isin(self.keep_values_[i])
            if rare_mask.any():
                X_arr[rare_mask, i] = self.fill_value
        return X_arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        return np.asarray([f"x{i}" for i in range(len(self.keep_values_))])


def build_category_pipeline(
    impute_strategy: str = "most_frequent",
    fill_value: str = "missing",
    handle_unknown: str = "ignore",
    min_freq: float = 0.01,
) -> Pipeline:
    """
    建構低基數類別特徵（DL 軌道）的處理管線。

    流程：
    1. SimpleImputer：填補缺失值（預設眾數）。
    2. RareCategoryGrouper：將低頻類別（< min_freq）合併為 '__other__'。
    3. OneHotEncoder：獨熱編碼，輸出 dense 矩陣。
    """
    if impute_strategy == "constant":
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
    else:
        imputer = SimpleImputer(strategy='most_frequent')

    return Pipeline([
        ('imputer', imputer),
        ('rare_grouper', RareCategoryGrouper(min_freq=min_freq)),
        ('encoder', OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)),
    ])


def build_tree_category_pipeline(min_freq: float = 0.01) -> Pipeline:
    """
    建構低基數類別特徵（Tree 軌道）的處理管線。

    使用 OrdinalEncoder 而非 OHE：
    - 保持單欄（1 per feature），避免維度膨脹
    - LightGBM / XGBoost 對整數編碼的類別支援較好，不需要 binary one-hot
    - unknown_value=-1：推論時遇到未見過的類別自動給 -1，不崩潰
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('rare_grouper', RareCategoryGrouper(min_freq=min_freq)),
        ('encoder', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])