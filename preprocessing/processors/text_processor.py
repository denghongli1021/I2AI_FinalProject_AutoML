# preprocessing/processors/text_processor.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer

def build_text_pipeline() -> Pipeline:
    """
    建立純文字(NLP)特徵的處理管線。
    步驟 1: 填補缺失值 (補上空字串，避免向量化時報錯)
    步驟 2: TF-IDF 向量化 (將文字轉為詞頻-逆文件頻率矩陣，限制最多 1000 個字詞)
    步驟 3: SVD 降維 (將高維度稀疏矩陣降至 50 維，節省記憶體並去雜訊)
    """
    text_pipeline = Pipeline(steps=[
        # 遇到空值補上空字串 'missing_text'
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing_text')),
        # 將字串轉為 TF-IDF 特徵，限制特徵上限避免記憶體爆炸
        ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
        # 降維，將 1000 維濃縮成 50 維的核心特徵
        ('svd', TruncatedSVD(n_components=50, random_state=42))
    ])
    
    return text_pipeline