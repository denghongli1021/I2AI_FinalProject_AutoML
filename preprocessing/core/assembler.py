# preprocessing/core/assembler.py
from sklearn.compose import ColumnTransformer

from ..processors.numeric_processor import build_numeric_pipeline
from ..processors.category_processor import build_category_pipeline
from ..processors.text_processor import build_text_pipeline
from ..processors.time_processor import build_time_pipeline

class PipelineAssembler:
    """
    特徵組裝廠。
    根據 Router 的分類結果，將對應的 sklearn Pipeline 綁定到特定欄位上。
    """
    def __init__(self, feature_groups: dict):
        self.feature_groups = feature_groups

    def build(self) -> ColumnTransformer:
        transformers = []

        # 1. 綁定數值處理管線
        if self.feature_groups.get("numeric"):
            transformers.append((
                "num_pipeline", 
                build_numeric_pipeline(), 
                self.feature_groups["numeric"]
            ))

        # 2. 綁定類別處理管線
        if self.feature_groups.get("categorical"):
            transformers.append((
                "cat_pipeline", 
                build_category_pipeline(), 
                self.feature_groups["categorical"]
            ))

        # 3. 綁定純文字 NLP 處理管線
        if self.feature_groups.get("text"):
            transformers.append((
                "text_pipeline", 
                build_text_pipeline(), 
                self.feature_groups["text"]
            ))
        
        # 4. 綁定時間序列處理管線
        if self.feature_groups.get("datetime"):
            transformers.append((
                "time_pipeline", 
                build_time_pipeline(), 
                self.feature_groups["datetime"]
            ))

        # 5. 組裝成最終的 scikit-learn 預處理器
        # remainder='drop' 表示如果遇到我們還不支援的型態，就先安全地丟棄
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop', 
            sparse_threshold=0.3 # 幫助控制 TF-IDF 產生的稀疏矩陣記憶體
        )

        return preprocessor