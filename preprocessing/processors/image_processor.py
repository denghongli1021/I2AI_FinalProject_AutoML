# preprocessing/processors/image_processor.py
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 需要安裝 Pillow: pip install Pillow
try:
    from PIL import Image, ImageStat
except ImportError:
    raise ImportError("請安裝 Pillow 套件以啟用圖片處理功能：pip install Pillow")

class ImageFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    [圖片特徵萃取器]
    輸入：包含圖片檔案路徑 (File Paths) 的 1D 陣列。
    輸出：每張圖片的統計特徵 (平均 RGB、對比度、亮度等) 構成的 2D 矩陣。
    
    防護機制：遇到破圖、路徑不存在或空值時，安全回傳 NaN，交由下游 Imputer 填補。
    """
    def __init__(self):
        # 這裡設定我們預期萃取出的特徵數量 (R, G, B 均值 + R, G, B 標準差 + 亮度 = 7 維)
        self.n_features = 7

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 確保 X 是一維的字串路徑陣列
        X_paths = np.asarray(X).ravel()
        extracted_features = []

        for path in X_paths:
            # 1. 空值或無效路徑防護
            if pd.isna(path) or not isinstance(path, str) or not os.path.exists(path):
                extracted_features.append(np.full(self.n_features, np.nan))
                continue

            # 2. 嘗試讀取並萃取圖片特徵
            try:
                with Image.open(path) as img:
                    # 強制轉為 RGB 模式 (避免灰階圖或 RGBA 透明圖造成維度錯誤)
                    img = img.convert('RGB')
                    
                    # 取得圖片統計資訊
                    stat = ImageStat.Stat(img)
                    
                    # 特徵 1~3: 平均 R, G, B (色彩傾向)
                    mean_r, mean_g, mean_b = stat.mean
                    
                    # 特徵 4~6: R, G, B 標準差 (代表對比度、紋理豐富度)
                    std_r, std_g, std_b = stat.stddev
                    
                    # 特徵 7: 視覺亮度 (使用感知亮度公式)
                    # Luminance = 0.299*R + 0.587*G + 0.114*B
                    luminance = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b
                    
                    extracted_features.append([
                        mean_r, mean_g, mean_b, 
                        std_r, std_g, std_b, 
                        luminance
                    ])
                    
            except Exception as e:
                # 破圖或無法解析的檔案防護：回傳 NaN，絕不讓管線崩潰
                # 開發階段可以用 print(f"讀取失敗: {path}, 原因: {e}") 來 debug
                extracted_features.append(np.full(self.n_features, np.nan))

        return np.array(extracted_features)

def build_image_pipeline(impute_strategy: str = "median") -> Pipeline:
    """
    建構圖片特徵 (Image) 的處理管線。
    
    流程：
    1. ImageFeatureExtractor：讀取路徑，萃取 7 維數值特徵。
    2. SimpleImputer：將破圖或缺失的圖片特徵補上中位數。
    3. StandardScaler：縮放色彩與亮度數值，對齊下游模型需求。
    """
    return Pipeline([
        # 第一站：視覺特徵萃取
        ("extractor", ImageFeatureExtractor()),
        
        # 第二站：填補空缺 (破圖的圖片會在這裡被補上整體資料集的平均色彩與亮度)
        ("imputer", SimpleImputer(strategy=impute_strategy)),
        
        # 第三站：尺度縮放
        ("scaler", StandardScaler())
    ])