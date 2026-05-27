# preprocessing/core/assembler.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion          # 新增 FeatureUnion（功能 1）
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, StandardScaler  # 新增 StandardScaler（功能 1）

# 💡 新增功能 1：迭代填補（MICE）+ 缺失值指示器
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import IterativeImputer, MissingIndicator
# 💡 新增功能 5：Mutual Information 特徵選擇
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression,
)

from ..processors.numeric_processor import build_numeric_pipeline
from ..processors.category_processor import build_category_pipeline
from ..processors.text_processor import build_text_pipeline
from ..processors.time_processor import build_time_pipeline

# 🆕 引入特徵軍火庫 (請確認路徑與檔名是否正確)
from ..processors.feature_generator import (
    GroupedAggregater,
    NonLinearScaler,
    PolynomialInteracter,
)


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


# 💡 新增功能 1：含缺失值指示器（+ 可選 MICE）的數值管線
def _build_numeric_pipeline_with_indicators(use_mice: bool = False) -> Pipeline:
    """
    建立包含「缺失值指示器」的數值處理管線。

    為什麼需要 Missing Indicator？
    某些缺失本身就是信號（MNAR，非隨機缺失）。
    例如高收入者故意不填薪資 → 「薪資欄是否缺失」本身就是有用特徵。
    此管線用 FeatureUnion 同時輸出：
      ① 填補後的數值（imputed values）
      ② 每個有缺失的欄位對應的 0/1 指示器（was_missing）

    use_mice=True 時使用 IterativeImputer（MICE）：
      用其他欄位的值迭代估計缺失值，比中位數填補更準確，
      但計算成本較高，適合欄位數 < 50 的場景。
    use_mice=False 時使用 SimpleImputer(median)（預設，速度快）。
    """
    imputer = (
        IterativeImputer(random_state=42, max_iter=10)
        if use_mice
        else SimpleImputer(strategy="median")
    )
    return Pipeline([
        ("features", FeatureUnion([
            ("imputed", Pipeline([
                ("imp", imputer),
                ("scl", StandardScaler()),
            ])),
            ("indicators", MissingIndicator(
                features="missing-only",   # 只對真的有缺失的欄位產生指示器
                error_on_new=False,        # 推論時遇到新缺失欄位不崩潰
            )),
        ])),
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

    def __init__(self, feature_groups: dict,
                 use_mice: bool = False,
                 add_missing_indicators: bool = False):
        """
        Parameters
        ----------
        feature_groups : dict
            來自 AutoRouter.fit_predict() 的分類結果。
        use_mice : bool
            數值欄填補是否改用 IterativeImputer（MICE）；False 用 median。（功能 1）
        add_missing_indicators : bool
            是否在數值管線加上缺失值指示器（was_missing 欄）。（功能 1）
        """
        if not isinstance(feature_groups, dict):
            raise TypeError(
                f"feature_groups 必須是 dict，收到的是 {type(feature_groups).__name__}"
            )
        self.feature_groups = feature_groups
        self.use_mice = use_mice                                      # 新增：MICE 開關（功能 1）
        self.add_missing_indicators = add_missing_indicators  # 新增：缺失指示器開關（功能 1）
        self._transformers: list = []
        # build() 過程中記錄「實際自動觸發了哪些進階特徵工程」,給前端 UI 透明顯示。
        # 每筆: {"type": "grouped_agg"|"polynomial"|"nonlinear_scale", "label", "detail", "columns"}
        self.applied_feature_steps: list = []

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
        self.applied_feature_steps = []   # 每次 build 重置

        # ── 1. 數值特徵 ───────────────────────────────────────────
        numeric_cols = self.feature_groups.get("numeric", [])
        if numeric_cols:
            if self.add_missing_indicators:
                # 新增（功能 1）：含缺失指示器的管線（MICE 或 median 填補 + was_missing 欄）
                num_pipe = _build_numeric_pipeline_with_indicators(use_mice=self.use_mice)
            else:
                # 維持原本行為（不改動現有邏輯）
                num_pipe = build_numeric_pipeline()
            transformers.append(("num_pipeline", num_pipe, numeric_cols))
            print(f"  [Assembler] 數值管線       ← {len(numeric_cols):2d} 欄: {numeric_cols[:5]}...")

        # ── 2. 低基數類別特徵（OHE）─────────────────────────────────
        categorical_cols = self.feature_groups.get("categorical", [])
        if categorical_cols:
            transformers.append((
                "cat_pipeline", build_category_pipeline(), categorical_cols
            ))
            print(f"  [Assembler] 類別管線(OHE)  ← {len(categorical_cols):2d} 欄: {categorical_cols[:5]}...")

        # ── 3. 高基數類別特徵（OrdinalEncoder，防 OHE 維度爆炸）────
        high_card_cols = self.feature_groups.get("high_cardinality", [])
        if high_card_cols:
            transformers.append((
                "high_card_pipeline",
                _build_high_cardinality_pipeline(),
                high_card_cols,
            ))
            print(f"  [Assembler] 高基數管線     ← {len(high_card_cols):2d} 欄: {high_card_cols[:5]}...")

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

        # ── 🆕 6. 核心特徵煉金術 (Advanced Feature Generation) ───────
        # 💡 AutoML 智慧識別機制：自動偵測適合進行特徵工程的黃金欄位
        
        # (A) 自動智慧聚合 (Grouped Aggregation)
        # 自動尋找資料集中適合當作 ID 的類別欄位 (例如包含 'card', 'addr', 'ProductCD' 或 email 的欄位)
        potential_ids = [
            c for c in (categorical_cols + high_card_cols)
            if any(k in c.lower() for k in ["card", "addr", "prod", "email"])
        ]
        # 自動尋找核心數值欄位 (如交易金額 TransactionAmt 或 距離 dist)
        potential_nums = [
            c for c in numeric_cols
            if any(k in c.lower() for k in ["amt", "amount", "dist", "c1", "c2"])
        ]

        if potential_ids and potential_nums:
            transformers.append((
                "feat_grouped_agg",
                GroupedAggregater(
                    group_cols=potential_ids[:3],
                    num_cols=potential_nums[:2],
                    aggs=["mean", "std"],
                ),
                list(set(potential_ids[:3] + potential_nums[:2])),
            ))
            print(f"  [Assembler] 🛡️ 注入自動群組聚合統計 分支！(ID: {potential_ids[:3]} | 數值: {potential_nums[:2]})")
            self.applied_feature_steps.append({
                "type": "grouped_agg",
                "label": "分組聚合統計",
                "detail": f"依 {potential_ids[:3]} 分組,算 {potential_nums[:2]} 的 mean/std",
                "columns": list(set(potential_ids[:3] + potential_nums[:2])),
            })

        # (B) 自動多項式交互特徵 (Polynomial Interactions)
        # 挑選前 3 個最重要的數值特徵進行兩兩交叉乘除，避免維度過度爆炸
        if len(potential_nums) >= 2:
            poly_targets = potential_nums[:3]
            transformers.append((
                "feat_polynomial",
                PolynomialInteracter(target_cols=poly_targets, allow_division=True),
                poly_targets,
            ))
            print(f"  [Assembler] ⚔️ 注入多項式交叉乘除 分支！(目標特徵: {poly_targets})")
            self.applied_feature_steps.append({
                "type": "polynomial",
                "label": "多項式交互特徵",
                "detail": f"對 {poly_targets} 兩兩做乘 / 除,造交互項",
                "columns": list(poly_targets),
            })

        # (C) 自動非線性縮放 (Non-linear Transformations)
        # 專門抓出金額類特徵進行常態化分位數轉換，矯正長尾偏態
        amt_cols = [c for c in numeric_cols if "amt" in c.lower() or "amount" in c.lower()]
        if amt_cols:
            transformers.append((
                "feat_nonlinear_scale",
                NonLinearScaler(target_cols=amt_cols, strategy="quantile"),
                amt_cols,
            ))
            print(f"  [Assembler] 🧪 注入非線性長尾偏態矯正 分支！(目標特徵: {amt_cols})")
            self.applied_feature_steps.append({
                "type": "nonlinear_scale",
                "label": "非線性長尾矯正",
                "detail": f"對 {amt_cols} 做 QuantileTransformer 常態化",
                "columns": list(amt_cols),
            })

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

    # 💡 隊友新增的功能 5：Mutual Information 特徵選擇 (保持不動)
    def select_features_by_mi(self, X_clean, y: pd.Series, feature_names: list, threshold: float = 0.01, top_k: int = None) -> tuple:
        """
        計算每個特徵對 target 的互資訊（Mutual Information）分數，
        過濾掉「對 target 幾乎沒有資訊量」的垃圾特徵。

        為什麼需要 MI 篩選？
        Router + Assembler 把所有欄位全部處理後可能產生數百個特徵
        （OHE 展開、TF-IDF、日期展開等），
        其中很多特徵與 target 幾乎無關，反而增加模型 overfitting 的風險。
        MI 分數 = 0 表示「完全沒有資訊量」，應直接丟棄。

        Parameters
        ----------
        X_clean      : fit_transform 後的 numpy array（已處理完的特徵矩陣）
        y            : 訓練標籤 Series
        feature_names: get_feature_names_out() 回傳的特徵名稱清單
        threshold    : MI 分數低於此值的特徵會被過濾掉（預設 0.01）
        top_k        : 若設定，無視 threshold，只保留分數最高的 top_k 個特徵

        Returns
        -------
        selected_mask : boolean array，True 表示保留，可直接用於 X_clean[:, selected_mask]
        mi_scores_df  : pd.DataFrame，包含 feature_name / mi_score / keep 三欄，
                        方便印出或存成 CSV 給組員看
        """
        import scipy.sparse as sp
        X_arr = X_clean.toarray() if sp.issparse(X_clean) else np.asarray(X_clean)

        # 自動判斷分類 / 迴歸
        is_classification = y.nunique() <= 50
        mi_fn = mutual_info_classif if is_classification else mutual_info_regression
        scores = mi_fn(X_arr, y, random_state=42)

        mi_df = pd.DataFrame({
            "feature_name": feature_names,
            "mi_score": scores,
        }).sort_values("mi_score", ascending=False).reset_index(drop=True)

        if top_k is not None:
            mi_df["keep"] = mi_df.index < top_k
        else:
            mi_df["keep"] = mi_df["mi_score"] >= threshold

        selected_mask = np.array(
            [mi_df.loc[mi_df["feature_name"] == fn, "keep"].values[0] for fn in feature_names],
            dtype=bool,
        )
        n_kept    = selected_mask.sum()
        n_dropped = len(selected_mask) - n_kept
        print(f"[MI 特徵選擇] 保留 {n_kept} 個特徵，丟棄 {n_dropped} 個（threshold={threshold}）")
        return selected_mask, mi_df