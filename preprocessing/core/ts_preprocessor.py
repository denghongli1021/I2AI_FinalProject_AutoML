# preprocessing/core/ts_preprocessor.py
import pandas as pd
import numpy as np

# 🌟 匯入模組內「實際存在」的底層元件
#   - DatetimeFeatureExtractor：日期欄→年/月/日/星期/週期特徵（即時計算，無洩漏）
#   - build_tree_category_pipeline：OrdinalEncoder 類別編碼（含未知值防呆）
#   - reduce_mem_usage：記憶體壓縮
from preprocessing.processors.time_processor import DatetimeFeatureExtractor
from preprocessing.processors.category_processor import build_tree_category_pipeline
from preprocessing.utils.memory_optimizer import reduce_mem_usage


class TSDataProcessor:
    """
    時序專用核心前處理器 (Orchestrator 模式)。
    負責調度底層 processors，確保時序安全 (排序、僅 ffill、train 段 fit)。

    時序安全準則
    ------------
    1. 只用「過去」補值：僅 ffill，**絕不 bfill**（bfill 會把未來值往回灌 → 洩漏）。
    2. 學習型轉換只在訓練段 fit：類別編碼的對應表只用前 ``n_train`` 列學習，
       再套用到全部資料；如此 test 段的分布不會洩漏回 train。
    3. 日期特徵採即時計算（年/月/日/星期…），每列只看自己的時間戳，無未來洩漏。

    Parameters
    ----------
    time_col : str, optional
        時間欄位名稱。若提供則按時間排序並萃取日曆特徵。
    target_col : str, optional
        目標欄位名稱（類別編碼時排除）。
    n_train : int, optional
        訓練段的列數（呼叫端若有 concat([train, test]) 應傳入 len(train)）。
        提供時，類別編碼只用前 n_train 列 fit → 零洩漏；
        未提供時退回用全部資料 fit（bfill 洩漏已消除，但合併後 fit 的殘留洩漏仍在）。
    """
    def __init__(self, time_col: str = None, target_col: str = None, n_train: int = None):
        self.time_col = time_col
        self.target_col = target_col
        self.n_train = n_train

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── 1. 🕒 時間排序 + 日曆特徵萃取 ──────────────────────────────
        if self.time_col and self.time_col in df.columns:
            print(f"  [TS 模組] 偵測到時間欄位 '{self.time_col}'，強制按時間先後排序...")
            df[self.time_col] = pd.to_datetime(df[self.time_col], errors="coerce")
            df = df.sort_values(by=self.time_col).reset_index(drop=True)

            # 🌟 用 DatetimeFeatureExtractor 萃取週期特徵（即時計算，安全）
            print("  [TS 模組] 萃取日曆/週期特徵 (年/月/日/星期/sin-cos)...")
            extractor = DatetimeFeatureExtractor(extract_time=True)
            cal_arr = extractor.transform(df[[self.time_col]])
            cal_df = pd.DataFrame(
                cal_arr,
                columns=extractor.get_feature_names_out([self.time_col]),
                index=df.index,
            )
            # 移除原始時間欄，併入萃取後的日曆特徵
            df = pd.concat([df.drop(columns=[self.time_col]), cal_df], axis=1)

        # ── 2. 🛡️ 時序安全補值：只 ffill，不 bfill ───────────────────
        n_nans = int(df.isna().sum().sum())
        if n_nans > 0:
            print(f"  [TS 模組] 發現 {n_nans} 個缺失值，執行時序安全補值 (僅 ffill)...")
            # 只用「過去」的值往前補（安全）。bfill 會用到未來，已移除。
            df = df.ffill()
            # 開頭沒有過去可補的殘留 NaN，用 0 兜底（不引入未來資訊）
            remaining = int(df.isna().sum().sum())
            if remaining > 0:
                print(f"  [TS 模組] 開頭仍有 {remaining} 個無過去可補的缺失，以 0 兜底。")
                df = df.fillna(0)

        # ── 3. 🔠 類別欄位安全編碼（只在 train 段 fit）─────────────────
        cat_cols = [
            c for c in df.select_dtypes(include=['object', 'category']).columns
            if c != self.target_col
        ]
        if cat_cols:
            print(f"  [TS 模組] 類別欄位 Ordinal 編碼 (train 段 fit): {cat_cols}")
            enc = build_tree_category_pipeline()
            # 只用訓練段學習對應表，避免 test 分布洩漏回 train
            if self.n_train is not None:
                fit_rows = df[cat_cols].iloc[:self.n_train]
            else:
                fit_rows = df[cat_cols]
            enc.fit(fit_rows)
            # OrdinalEncoder 內建 handle_unknown=use_encoded_value → test 新類別給 -1，不洩漏不崩潰
            df[cat_cols] = enc.transform(df[cat_cols])

        # ── 4. 🗑️ 捨棄無意義的 ID 欄位 ────────────────────────────────
        if 'id' in df.columns and 'id' != self.target_col:
            df = df.drop(columns=['id'])

        # ── 5. 🗜️ 記憶體優化 ──────────────────────────────────────────
        print("  [TS 模組] 進行記憶體壓縮...")
        df = reduce_mem_usage(df)

        return df
