# preprocessing/core/ts_preprocessor.py
import pandas as pd
import numpy as np

# 🌟 匯入你辛辛苦苦寫好的底層模組！
from preprocessing.processors.time_processor import TimeProcessor       # 假設你的時間處理器叫這個
from preprocessing.processors.category_processor import CategoryProcessor # 假設你的類別處理器叫這個
from preprocessing.utils.memory_optimizer import memory_optimizer # 假設你的記憶體優化器叫這個

class TSDataProcessor:
    """
    時序專用核心前處理器 (Orchestrator 模式)。
    負責調度底層 processors，確保時序安全 (排序、ffill)。
    """
    def __init__(self, time_col: str = None, target_col: str = None):
        self.time_col = time_col
        self.target_col = target_col

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # ── 1. 🕒 時間排序 (這步是時序專屬邏輯，必須在這裡做) ──
        if self.time_col and self.time_col in df.columns:
            print(f"  [TS 模組] 偵測到時間欄位 '{self.time_col}'，強制按時間先後排序...")
            df[self.time_col] = pd.to_datetime(df[self.time_col])
            df = df.sort_values(by=self.time_col).reset_index(drop=True)
            
            # 🌟 呼叫你既有的 TimeProcessor 進行特徵萃取 (月、日、星期等)
            print("  [TS 模組] 呼叫 TimeProcessor 萃取週期特徵...")
            time_processor = TimeProcessor(time_cols=[self.time_col]) 
            df = time_processor.fit_transform(df) # 依你實際的 API 為準
            
            # 確保原始時間字串欄位被移除
            if self.time_col in df.columns:
                df = df.drop(columns=[self.time_col])

        # ── 2. 🛡️ 安全補值 (時序專屬：Forward Fill) ──
        n_nans = df.isna().sum().sum()
        if n_nans > 0:
            print(f"  [TS 模組] 發現 {n_nans} 個缺失值，執行時序安全補值 (ffill -> bfill)...")
            # 時序不能呼叫 numeric.py 的平均補值，必須在這裡強制 ffill
            df = df.ffill().bfill()

        # ── 3. 🔠 類別欄位安全編碼 ──
        cat_cols = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != self.target_col]
        if cat_cols:
            print(f"  [TS 模組] 呼叫 CategoryProcessor 處理類別欄位: {cat_cols}")
            # 🌟 呼叫你既有的 CategoryProcessor (Label Encoding / Factorize)
            cat_processor = CategoryProcessor(cat_cols=cat_cols, encoding_type='label') # 依你實際的 API 為準
            df = cat_processor.fit_transform(df)

        # ── 4. 🗑️ 捨棄無意義的 ID 欄位 ──
        if 'id' in df.columns and 'id' != self.target_col:
            df = df.drop(columns=['id'])

        # ── 5. 🗜️ 記憶體優化 ──
        print("  [TS 模組] 呼叫 memory_optimizer 進行記憶體壓縮...")
        # 🌟 呼叫你既有的防爆記憶體優化器
        df = memory_optimizer(df)

        return df