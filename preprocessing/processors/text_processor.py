# preprocessing/processors/text_processor.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

class SafeTruncatedSVD(BaseEstimator, TransformerMixin):
    """
    [安全版 SVD] 
    自動偵測輸入的特徵維度，防止 SVD 要求降維的目標維度大於實際單字數量。
    """
    def __init__(self, n_components=50, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        # 注意：這裡不需要宣告 self.svd = None 了，全交給 fit 產生

    def fit(self, X, y=None):
        actual_features = X.shape[1]
        safe_components = min(self.n_components, actual_features - 1)
        safe_components = max(1, safe_components)
        
        # 💡 建立並擬合帶有底線的 self.svd_
        self.svd_ = TruncatedSVD(n_components=safe_components, random_state=self.random_state)
        self.svd_.fit(X)
        return self

    def transform(self, X):
        # 💡 必須呼叫帶有底線的 self.svd_
        return self.svd_.transform(X)
        
    def get_feature_names_out(self, input_features=None):
        # 💡 必須呼叫帶有底線的 self.svd_
        actual_components = self.svd_.components_.shape[0]
        return np.array([f"svd_{i}" for i in range(actual_components)])


def build_text_pipeline(
    max_features: int = 1000, 
    svd_components: int = 50,
) -> Pipeline:
    """
    建構自由文字 (NLP) 的處理管線 (AutoML 萬用版)。
    """
    return Pipeline([
        # 第一站：文字轉為高維度稀疏矩陣
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            # 💡 防禦極端狀況二：不預設為 english，讓系統能吃多國語言
            stop_words=None,
            # 💡 容錯機制：改用 char_wb (字元級別)，能無視語言隔閡處理中文與亂碼
            # ngram_range=(2, 4) 代表擷取 2~4 個字元的組合
            analyzer='char_wb',
            ngram_range=(2, 4),
            # 確保不會因為全是極罕見字而崩潰
            min_df=1
        )),
        
        # 第二站：動態防禦版 SVD 降維
        # 💡 防禦極端狀況一：遇到詞彙量極少的欄位，不會再崩潰
        ("svd", SafeTruncatedSVD(
            n_components=svd_components,
            random_state=42
        ))
    ])