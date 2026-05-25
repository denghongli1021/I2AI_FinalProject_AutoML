import pandas as pd
import gc  # 🆕 引入 Python 內建的垃圾回收模組
from typing import List, Union

from preprocessing.utils.memory_optimizer import reduce_mem_usage

def load_and_merge_data(data_source: Union[pd.DataFrame, str, List[str]], main_file_index: int = 0) -> pd.DataFrame:
    """
    AutoML 智慧資料載入器
    支援三種輸入：
    1. 已經讀好的 pd.DataFrame
    2. 單一 CSV 檔案路徑 (str)
    3. 多個 CSV 檔案路徑清單 (List[str])，會自動尋找共同 ID 並合併
    """
    # 情況 1: 使用者已經自己讀好 DataFrame 傳進來了
    if isinstance(data_source, pd.DataFrame):
        print("📥 偵測到 DataFrame 輸入，進行記憶體壓縮...")
        return reduce_mem_usage(data_source)
    
    # 情況 2: 使用者傳入單一檔案路徑
    if isinstance(data_source, str):
        print(f"📥 讀取單一檔案: {data_source}")
        df = pd.read_csv(data_source)
        return reduce_mem_usage(df)
    
    # 情況 3: 使用者傳入多個檔案路徑，啟動智慧合併 (Multi-table Join)
    if isinstance(data_source, list):
        print(f"📥 偵測到 {len(data_source)} 個檔案，準備進行智慧合併...")
        dfs = [pd.read_csv(p) for p in data_source]
        
        # 先個別壓縮，防止合併時 OOM
        dfs = [reduce_mem_usage(df) for df in dfs]
        
        main_df = dfs[main_file_index]
        
        for i, df in enumerate(dfs):
            if i == main_file_index: continue
            
            # 尋找共同欄位 (Primary/Foreign Keys)
            common_cols = list(set(main_df.columns) & set(df.columns))
            
            if common_cols:
                print(f"🔗 偵測到共同 ID {common_cols}，執行 Left Join...")
                main_df = main_df.merge(df, on=common_cols, how='left')
            else:
                print(f"⚠️ 無法在檔案 {i} 找到與主表的共同 ID，已跳過合併。")
                
        # 🆕 =========================================================
        # 🧹 主動式垃圾回收機制 (Active Garbage Collection)
        # 斬斷暫存清單與過期變數的參照，防止在大數據合併後 RAM 居高不下
        # =========================================================
        del dfs  # 徹底銷毀包含所有原始子表的清單
        collected = gc.collect()  # 強制核心立刻回收無效的記憶體碎片
        print(f"🧹 [記憶體清理] 成功回收 {collected} 個快取與過期參考，確保巔峰記憶體安全降落！")
        # =========================================================
                
        return main_df
    
    raise ValueError("❌ data_source 格式錯誤！必須是 DataFrame, 字串路徑, 或路徑清單。")