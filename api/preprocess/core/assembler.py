# preprocessing/core/assembler.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder

from ..processors.numeric_processor import build_numeric_pipeline
from ..processors.category_processor import build_category_pipeline
from ..processors.text_processor import build_text_pipeline
from ..processors.time_processor import build_time_pipeline


# ──────────────────────────────────────────────────────────────────
# 私有輔助函式
# ──────────────────────────────────────────────────────────────────

def _flatten_and_fill_na(X) -> np.ndarray:
    """
    展平任意維度的字串輸入為 1D 陣列，並將 NaN 填補為 'missing_text'。

    為什麼需要這個步驟？
    text_processor.py 的管線順序是：
        SimpleImputer → TfidfVectorizer → TruncatedSVD

    問題：
    - ColumnTransformer 傳給 Pipeline 的是 2D array (n_samples, 1)
    - SimpleImputer 輸出仍然是 2D (n_samples, 1)
    - TfidfVectorizer 需要 1D 字串陣列（每個元素是一篇文件）
    - 傳入 2D 時 TF-IDF 把「列」視為特徵而非文件 → ValueError 崩潰

    解法：在 Assembler 層加前置展平步驟，繞過 SimpleImputer，
    自行處理 NaN 後傳 1D 給 TF-IDF。
    """
    flat = pd.Series(np.asarray(X, dtype=object).ravel())
    return flat.fillna("missing_text").astype(str).values


def _build_text_column_pipeline() -> Pipeline:
    """
    建立單一文字欄位的處理管線（含維度相容性修正）。

    流程：
        展平 + NaN 填補（1D）→ TfidfVectorizer（詞頻矩陣）→ TruncatedSVD（降維 50 維）

    為什麼跳過 text_processor.py 裡的 SimpleImputer？
    SimpleImputer 輸出 2D，TF-IDF 需要 1D，兩者維度不相容。
    _flatten_and_fill_na 統一替代 SimpleImputer 的 NaN 填補功能。
    """
    inner = build_text_pipeline()
    tfidf = inner.named_steps["tfidf"]
    svd   = inner.named_steps["svd"]
    return Pipeline([
        # 💡 新增 feature_names_out="one-to-one"
        ("flatten", FunctionTransformer(_flatten_and_fill_na, validate=False, feature_names_out="one-to-one")),
        ("tfidf",   tfidf),
        ("svd",     svd),
    ])


def _build_high_cardinality_pipeline() -> Pipeline:
    """
    建立高基數類別欄位的安全編碼管線。

    為什麼用 OrdinalEncoder 而非 OneHotEncoder？
    高基數類別（如 500 種郵遞區號）若用 OHE，一欄會展開成 500 欄，
    1000 種產品代碼 → 1000 欄 → 記憶體爆炸（Memory OutOfBounds）。

    OrdinalEncoder 將每個類別映射為一個整數，欄位數保持不變（仍然 1 欄）。
    handle_unknown='use_encoded_value' + unknown_value=-1：
        當測試集遇到訓練集未見過的新類別時，
        用 -1 代替而不崩潰，確保推論時的健壯性。

    為什麼不用 Target Encoding？
    Target Encoding 需要在 fit 時用到目標欄 y，若直接在 Pipeline 中使用，
    訓練集會被「洩漏」自身的標籤資訊，違反防洩漏原則。
    OrdinalEncoder 不需要 y，安全且簡單。
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-1,
        )),
    ])


# ──────────────────────────────────────────────────────────────────
# PipelineAssembler 主類別
# ──────────────────────────────────────────────────────────────────

class PipelineAssembler:
    """
    特徵管線組裝廠（Pipeline Assembler）。

    根據 AutoRouter 的分類結果，將對應的 sklearn Pipeline 綁定到特定欄位，
    組裝成一個 ColumnTransformer，可直接 fit_transform / transform。

    支援的群組：
    - numeric         → build_numeric_pipeline()（median 填補 + StandardScaler）
    - categorical     → build_category_pipeline()（眾數填補 + OneHotEncoder）
    - high_cardinality→ OrdinalEncoder（眾數填補 + Ordinal 編碼，防維度爆炸）
    - text            → 每欄獨立：展平 + TF-IDF + SVD（修正維度問題）
    - datetime        → build_time_pipeline()（提取年/月/日/星期等）
    - dropped         → 不建任何 Transformer，由 remainder='drop' 自動忽略
    """

    def __init__(self, feature_groups: dict):
        """
        Parameters
        ----------
        feature_groups : dict
            來自 AutoRouter.fit_predict() 的分類結果。
        """
        if not isinstance(feature_groups, dict):
            raise TypeError(
                f"feature_groups 必須是 dict，收到的是 {type(feature_groups).__name__}"
            )
        self.feature_groups = feature_groups
        self._transformers: list = []

    def build(self) -> ColumnTransformer:
        """
        組裝並回傳 ColumnTransformer。

        Returns
        -------
        sklearn.compose.ColumnTransformer
            可直接呼叫 fit_transform(X_train) 和 transform(X_test)。

        Raises
        ------
        ValueError
            若所有群組均為空，無法建立任何管線。
        """
        transformers = []

        # ── 1. 數值特徵 ───────────────────────────────────────────
        numeric_cols = self.feature_groups.get("numeric", [])
        if numeric_cols:
            transformers.append((
                "num_pipeline", build_numeric_pipeline(), numeric_cols
            ))
            print(f"  [Assembler] 數值管線       ← {len(numeric_cols):2d} 欄: {numeric_cols}")

        # ── 2. 低基數類別特徵（OHE）─────────────────────────────────
        categorical_cols = self.feature_groups.get("categorical", [])
        if categorical_cols:
            transformers.append((
                "cat_pipeline", build_category_pipeline(), categorical_cols
            ))
            print(f"  [Assembler] 類別管線(OHE)  ← {len(categorical_cols):2d} 欄: {categorical_cols}")

        # ── 3. 高基數類別特徵（OrdinalEncoder，防 OHE 維度爆炸）────
        high_card_cols = self.feature_groups.get("high_cardinality", [])
        if high_card_cols:
            transformers.append((
                "high_card_pipeline",
                _build_high_cardinality_pipeline(),
                high_card_cols,
            ))
            print(f"  [Assembler] 高基數管線     ← {len(high_card_cols):2d} 欄: {high_card_cols}")

        # ── 4. 文字特徵（每欄獨立，修正 TF-IDF 維度問題）──────────
        text_cols = self.feature_groups.get("text", [])
        for col in text_cols:
            transformers.append((
                f"text_{col}",               # 名稱唯一，ColumnTransformer 要求
                _build_text_column_pipeline(),
                [col],                        # 傳 list → 2D (n,1) → flatten 展平
            ))
        if text_cols:
            print(f"  [Assembler] 文字管線       ← {len(text_cols):2d} 欄: {text_cols}（每欄獨立）")

        # ── 5. 日期時間特徵 ──────────────────────────────────────────
        datetime_cols = self.feature_groups.get("datetime", [])
        if datetime_cols:
            transformers.append((
                "time_pipeline", build_time_pipeline(), datetime_cols
            ))
            print(f"  [Assembler] 日期管線       ← {len(datetime_cols):2d} 欄: {datetime_cols}")

        # ── 空管線早期錯誤 ────────────────────────────────────────
        if not transformers:
            raise ValueError(
                "所有欄位均已被剔除，無法建立任何預處理管線。\n"
                "可能原因：\n"
                "  1. 輸入資料全部是常數欄或 ID 欄\n"
                "  2. 大量欄位缺失率 > 60% 被整欄刪除\n"
                "  3. schema_override 設定過於激進\n"
                f"  目前 feature_groups = {self.feature_groups}"
            )

        self._transformers = transformers

        preprocessor = ColumnTransformer(
            transformers=transformers,
            # remainder='drop'：
            #   'dropped' 群組的欄位不在任何 transformer 的清單中，
            #   ColumnTransformer 自動忽略它們，不進模型。
            remainder="drop",
            # sparse_threshold=0.3：
            #   TF-IDF 輸出是稀疏矩陣。若整體輸出 70% 以上是零，
            #   保留 scipy sparse 格式節省記憶體。
            sparse_threshold=0.3,
        )
        return preprocessor

    def describe(self) -> None:
        """印出已組裝的管線摘要（需先呼叫 build()）"""
        if not self._transformers:
            print("[Assembler] 尚未 build()，請先呼叫 build()。")
            return
        print("\n[Assembler] ══════════ 管線組裝摘要 ══════════")
        for name, _pipe, cols in self._transformers:
            col_list = [cols] if isinstance(cols, str) else list(cols)
            print(f"  {name:<30} → {len(col_list):2d} 個欄位")
        print("[Assembler] ════════════════════════════════\n")

    def get_feature_groups(self) -> dict:
        """回傳原始 feature_groups（方便外部查詢）"""
        return self.feature_groups
