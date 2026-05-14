# preprocessing/processors/text_processor.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class _FlattenToText(BaseEstimator, TransformerMixin):
    """把 ColumnTransformer 傳進來的 2D 陣列 (n_samples, n_text_cols) 壓成
    1D 字串陣列,讓 TfidfVectorizer 能吃。

    為什麼需要這個:ColumnTransformer 給文字欄位是 2D 的,但 TfidfVectorizer
    只接受「一維字串可迭代物」。若有多個文字欄,這裡把每列各欄用空格串成一段。
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=object)
        if arr.ndim == 1:
            return np.array([str(v) for v in arr], dtype=object)
        # 2D → 每列把各欄字串用空格接起來
        return np.array([" ".join(str(v) for v in row) for row in arr], dtype=object)

    def get_feature_names_out(self, input_features=None):
        # 下一步 TfidfVectorizer 會自己產生詞彙欄位名,這裡只需給個佔位名
        # 讓 Pipeline.get_feature_names_out 的鏈式呼叫不會中斷
        return np.array(["text_combined"])


class _SafeSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD 的安全版:n_components 自動夾在 (n_features - 1) 以內。

    為什麼需要這個:TfidfVectorizer 在小資料集可能只產生個位數的詞彙特徵,
    若 n_components >= n_features,TruncatedSVD 會直接報錯。
    """
    def __init__(self, n_components: int = 50, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        n_feat = X.shape[1] if hasattr(X, "shape") and len(X.shape) > 1 else 1
        k = min(self.n_components, max(1, n_feat - 1))
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.svd_.fit(X)
        return self

    def transform(self, X):
        return self.svd_.transform(X)

    def get_feature_names_out(self, input_features=None):
        n = getattr(getattr(self, "svd_", None), "n_components", self.n_components)
        return np.array([f"svd_{i}" for i in range(n)])


def build_text_pipeline() -> Pipeline:
    """
    建立純文字(NLP)特徵的處理管線。
    步驟 1: 填補缺失值 (補上 'missing_text',避免向量化時報錯)
    步驟 2: 壓成 1D 字串 (TfidfVectorizer 只吃一維)
    步驟 3: TF-IDF 向量化 (限制最多 1000 個字詞)
    步驟 4: SVD 降維 (降至最多 50 維,小資料集會自動縮減維度)
    """
    text_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing_text')),
        ('flatten', _FlattenToText()),
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        ('svd', _SafeSVD(n_components=50, random_state=42)),
    ])

    return text_pipeline
