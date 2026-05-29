# preprocessing/core/router.py
import re
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class AutoRouter:
    """
    AutoML 系統的自動欄位分類器（Auto Feature Router）。

    掃描 DataFrame 中的每個欄位，將其歸類為六種類型之一：
      - numeric         : 數值特徵（float / int）
      - categorical     : 低基數類別（唯一值 ≤ cat_threshold）
      - high_cardinality: 高基數類別（唯一值多但字串短，如郵遞區號）
      - text            : 自由文字（長字串，適合 TF-IDF）
      - datetime        : 時間特徵（datetime64 或可解析的日期字串）
      - dropped         : 無用欄（常數欄、ID 欄、缺失 > 60%、字串型 ID）

    設計原則（方案 A：中控樞紐模式）：
        Router 與 data_health 保持獨立，各自實作相同的剔除邏輯。
        由 interface.py 統一協調兩者。這樣 data_health 可以單獨給 UI 使用，
        Router 也不需要依賴 data_health 的輸出，模組邊界清晰。
    """

    VALID_OVERRIDE_TYPES = frozenset(
        ["numeric", "categorical", "high_cardinality", "text", "datetime", "dropped"]
    )

    def __init__(
        self,
        categorical_threshold: int = 20,
        text_length_threshold: float = 20.0,
        id_ratio_threshold: float = 0.95,
        missing_drop_threshold: float = 0.60,
        schema_override: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        categorical_threshold : int
            唯一值上限。欄位唯一值 ≤ 此值 → 低基數類別。預設 20。
        text_length_threshold : float
            平均字串長度門檻（字元）。超過此值 → 自由文字。預設 20.0。
        id_ratio_threshold : float
            唯一值比例門檻。整數欄或短字串欄比例 ≥ 此值 → 視為 ID 欄。預設 0.95。
        missing_drop_threshold : float
            缺失率門檻。缺失率 > 此值 → 整欄刪除。預設 0.60（60%）。
            與 data_health.py 的警告門檻一致，確保「言行一致」。
        schema_override : dict, optional
            手動覆蓋自動判斷，優先於所有自動邏輯。
            合法值：numeric / categorical / high_cardinality / text / datetime / dropped
            範例：{"zipcode": "high_cardinality", "event_ts": "datetime"}
        """
        self.cat_threshold       = categorical_threshold
        self.text_threshold      = text_length_threshold
        self.id_ratio_threshold  = id_ratio_threshold
        self.missing_drop_threshold = missing_drop_threshold
        self.schema_override     = schema_override or {}

        self.feature_groups: Dict[str, List[str]] = {
            "numeric":          [],
            "categorical":      [],   # OHE（唯一值少）
            "high_cardinality": [],   # OrdinalEncoder（唯一值多，OHE 會爆炸）
            "text":             [],   # TF-IDF + SVD
            "datetime":         [],   # 提取年/月/日/星期等特徵
            "dropped":          [],   # 不進模型
        }
        self._routing_log: List[str] = []

    # ──────────────────────────────────────────────────────────────
    # 私有輔助方法
    # ──────────────────────────────────────────────────────────────

    def _log(self, tag: str, col: str, reason: str) -> None:
        self._routing_log.append(f"  [{tag:<15}] '{col}' → {reason}")

    def _try_parse_datetime(self, series: pd.Series) -> bool:
        """
        採樣前 50 筆，嘗試解析為日期。
        抑制 Pandas 2.0+ 的 UserWarning（不影響功能，避免汙染輸出）。
        移除了已棄用的 infer_datetime_format 參數。
        """
        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False

        # 💡 使用 with 區塊來靜音 Pandas 的格式警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
            
        success_ratio = parsed.notna().mean()
        return success_ratio > 0.5

    def _looks_like_string_id(
        self, sample: pd.Series, unique_ratio: float, avg_len: float
    ) -> bool:
        """
        判斷字串欄位是否為明確的 ID 型字串（UUID、Email、Hex Hash）。

        為什麼只靠 pattern 判斷，不靠唯一比例？
        「高唯一比例 + 短字串」這個條件在小資料集時過於敏感：
        40 筆資料包含 40 種郵遞區號 → unique_ratio=100%，
        但它是合法的高基數類別，不應被誤判為 ID 刪掉。
        因此只有具體格式能確認（UUID / Email / Hash）才判定為 ID。
        其他高基數短字串一律進 high_cardinality 群組。
        """
        # 1. UUID（8-4-4-4-12 格式的 16 進位字串）
        uuid_re = (
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
            r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        )
        if sample.str.match(uuid_re, na=False).mean() > 0.5:
            return True

        # 2. Email（含 @）
        if sample.str.contains("@", na=False).mean() > 0.5:
            return True

        # 3. Hex Hash（固定長度：MD5=32, SHA1=40, SHA256=64，全為 16 進位）
        if avg_len in (32.0, 40.0, 64.0):
            hex_re = r"^[0-9a-fA-F]+$"
            if sample.str.match(hex_re, na=False, case=False).mean() > 0.5:
                return True

        return False

    def _is_pandas_nullable_numeric(self, dtype) -> bool:
        """
        偵測 Pandas 可空整數 / 浮點型別（Int64, Float64 等）。
        這些型別不會被 pd.api.types.is_numeric_dtype() 標準函式偵測到。
        """
        return hasattr(dtype, "numpy_dtype") and np.issubdtype(
            dtype.numpy_dtype, np.number
        )

    # ──────────────────────────────────────────────────────────────
    # 主要分類邏輯
    # ──────────────────────────────────────────────────────────────

    def fit_predict(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        掃描 DataFrame，回傳各欄位的分類結果字典。

        Parameters
        ----------
        df : pd.DataFrame
            待掃描的資料框（唯讀，不修改原始資料）。
        target_col : str, optional
            目標欄位名稱，排除在特徵分類之外。

        Returns
        -------
        dict
            feature_groups：numeric / categorical / high_cardinality /
                            text / datetime / dropped
        """
        if len(df) == 0:
            raise ValueError("傳入的 DataFrame 是空的（0 列），無法進行分類。")

        n_rows = len(df)

        for col in df.columns:

            # ── 0. 跳過目標欄 ─────────────────────────────────────
            if target_col and col == target_col:
                self._log("TARGET", col, "目標欄，排除在特徵分類之外")
                continue

            # ── 1. schema_override 優先（最高優先權）──────────────
            if col in self.schema_override:
                override_type = self.schema_override[col]
                if override_type in self.VALID_OVERRIDE_TYPES:
                    self.feature_groups[override_type].append(col)
                    self._log("OVERRIDE", col, f"{override_type}（手動覆蓋）")
                else:
                    print(
                        f"[AutoRouter] ⚠ schema_override['{col}'] = '{override_type}' 不合法，"
                        f"合法值：{sorted(self.VALID_OVERRIDE_TYPES)}。改用自動判斷。"
                    )
                continue

            series       = df[col]
            dtype        = series.dtype
            n_unique     = series.nunique(dropna=True)
            missing_rate = series.isna().mean()

            # ── 2. 高缺失率（> 60%）→ dropped ────────────────────
            # 與 data_health.py 的警告邏輯一致（「言行一致」修正）。
            # 缺失 > 60% 時，無論用何種方式填補，補出來的幾乎都是假的，
            # 引入雜訊遠大於有效資訊。
            if missing_rate > self.missing_drop_threshold:
                self.feature_groups["dropped"].append(col)
                self._log(
                    "DROPPED",
                    col,
                    f"缺失率 {missing_rate:.0%} > {self.missing_drop_threshold:.0%}，整欄剔除",
                )
                continue

            # ── 3. 常數欄（唯一值 ≤ 1）→ dropped ─────────────────
            # 對模型無資訊量；StandardScaler 計算標準差=0 → 除以零崩潰。
            if n_unique <= 1:
                self.feature_groups["dropped"].append(col)
                self._log("DROPPED", col, f"常數欄（唯一值 {n_unique} 個）")
                continue

            # ── 4. 原生 datetime64 型別 ───────────────────────────
            if pd.api.types.is_datetime64_any_dtype(dtype):
                self.feature_groups["datetime"].append(col)
                self._log("DATETIME", col, "datetime64 型別")
                continue

            # ── 5. 布林型別 → categorical ─────────────────────────
            # 布林是 int 的子型別，必須在 numeric 判斷之前處理，
            # 否則 True/False 會被 StandardScaler 縮放成無意義的小數。
            if pd.api.types.is_bool_dtype(dtype):
                self.feature_groups["categorical"].append(col)
                self._log("CATEGORICAL", col, "布林型別，視為二元類別")
                continue

            # ── 6. Pandas 類別 dtype ──────────────────────────────
            if isinstance(dtype, pd.CategoricalDtype):
                self.feature_groups["categorical"].append(col)
                self._log("CATEGORICAL", col, f"pd.CategoricalDtype（{n_unique} 個類別）")
                continue

            # ── 7. 數值型別（含 Pandas 可空整數 / 浮點）────────────
            is_numeric = (
                pd.api.types.is_numeric_dtype(dtype)
                or self._is_pandas_nullable_numeric(dtype)
            )
            if is_numeric:
                unique_ratio = n_unique / n_rows
                is_integer   = (
                    pd.api.types.is_integer_dtype(dtype)
                    or (
                        hasattr(dtype, "numpy_dtype")
                        and np.issubdtype(dtype.numpy_dtype, np.integer)
                    )
                )
                # 整數型且幾乎每筆都不同 → 疑似 ID
                if is_integer and unique_ratio >= self.id_ratio_threshold:
                    self.feature_groups["dropped"].append(col)
                    self._log(
                        "DROPPED",
                        col,
                        f"疑似整數 ID（唯一比例 {unique_ratio:.0%}）",
                    )
                else:
                    self.feature_groups["numeric"].append(col)
                    self._log("NUMERIC", col, f"數值型 dtype={dtype.name}，唯一值 {n_unique}")
                continue

            # ── 8. 字串 / Object 型別 ────────────────────────────
            if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):

                # 8a. 日期字串偵測（優先）
                if self._try_parse_datetime(series):
                    self.feature_groups["datetime"].append(col)
                    self._log("DATETIME", col, "日期字串（可自動解析）")
                    continue

                # 採樣 100 筆計算平均字串長度（避免全欄掃描拖垮效能）
                valid_samples = series.dropna().astype(str)
                sample_size   = min(100, len(valid_samples))

                if sample_size == 0:
                    self.feature_groups["dropped"].append(col)
                    self._log("DROPPED", col, "全為空值，已剔除")
                    continue

                sample       = valid_samples.sample(n=sample_size, random_state=42)
                avg_len      = sample.str.len().mean()
                unique_ratio = n_unique / n_rows

                # 8b. 字串型 ID（UUID / Email / Hash / 高唯一短字串）→ dropped
                # 必須在長度判斷之前，因 UUID 長度 36 字元會超過 text_threshold。
                if self._looks_like_string_id(sample, unique_ratio, avg_len):
                    self.feature_groups["dropped"].append(col)
                    self._log(
                        "DROPPED",
                        col,
                        f"字串型 ID（UUID/Email/Hash 或唯一比例 {unique_ratio:.0%}）",
                    )
                    continue

                # 💡 8c. 先檢查是否為長字串 → text（NLP）
                # 必須在 Categorical 之前，否則幾句重複出現的長文本會被誤認為 One-Hot 類別
                if avg_len > self.text_threshold:
                    self.feature_groups["text"].append(col)
                    self._log(
                        "TEXT",
                        col,
                        f"自由文字（平均 {avg_len:.1f} 字元 > 門檻 {self.text_threshold}）",
                    )
                    continue

                # 💡 8d. 確定字串不長後，再檢查是否為低基數類別 → categorical
                if n_unique <= self.cat_threshold:
                    self.feature_groups["categorical"].append(col)
                    self._log(
                        "CATEGORICAL",
                        col,
                        f"低基數字串（{n_unique} 種 ≤ 門檻 {self.cat_threshold}）",
                    )
                    continue

                # 8e. 高基數短字串 → high_cardinality
                # 唯一值多（如 1000 種郵遞區號），若用 OHE 會讓欄位暴增 1000 倍。
                # 改用 OrdinalEncoder 保持 1 欄，安全且有效。
                self.feature_groups["high_cardinality"].append(col)
                self._log(
                    "HIGH_CARD",
                    col,
                    f"高基數字串（{n_unique} 種，平均 {avg_len:.1f} 字元），改用 OrdinalEncoder",
                )
                continue

        self._print_summary()
        return self.feature_groups

    # ──────────────────────────────────────────────────────────────
    # 輸出與查詢
    # ──────────────────────────────────────────────────────────────

    def _print_summary(self) -> None:
        print("\n[Auto-Router] ══════════ 分類結果摘要 ══════════")
        total = sum(len(v) for v in self.feature_groups.values())
        for group, cols in self.feature_groups.items():
            if cols:
                print(f"  {group:<18}({len(cols):3d} 欄): {cols}")
        print(f"  {'共計':<17}({total:3d} 欄)")
        print("[Auto-Router] ════════════════════════════════\n")

    def get_routing_log(self) -> List[str]:
        """回傳每個欄位的詳細判斷記錄，方便 debug"""
        return self._routing_log

    def print_routing_log(self) -> None:
        print("\n[Auto-Router] 詳細判斷記錄：")
        for line in self._routing_log:
            print(line)
        print()

    def reset(self) -> None:
        """清空分類結果，讓同一個實例可以重複使用"""
        for key in self.feature_groups:
            self.feature_groups[key] = []
        self._routing_log = []
