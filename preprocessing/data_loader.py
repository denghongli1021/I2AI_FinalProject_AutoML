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
        print("[DataLoader] 偵測到 DataFrame 輸入，進行記憶體壓縮...")
        return reduce_mem_usage(data_source)

    # 情況 2: 使用者傳入單一檔案路徑
    if isinstance(data_source, str):
        print(f"[DataLoader] 讀取單一檔案: {data_source}")
        df = pd.read_csv(data_source)
        return reduce_mem_usage(df)

    # 情況 3: 使用者傳入多個檔案路徑，啟動智慧合併 (Multi-table Join)
    if isinstance(data_source, list):
        print(f"[DataLoader] 偵測到 {len(data_source)} 個檔案，準備進行智慧合併...")
        dfs = [pd.read_csv(p) for p in data_source]

        # 先個別壓縮，防止合併時 OOM
        dfs = [reduce_mem_usage(df) for df in dfs]

        main_df = dfs[main_file_index]

        for i, df in enumerate(dfs):
            if i == main_file_index: continue

            for i, df in enumerate(dfs):
                if i == main_file_index: continue

                # 1. 尋找所有共同欄位
                all_common = list(set(main_df.columns) & set(df.columns))
                
                # 2. 🚀 智慧篩選：只挑選含有 'id', 'key', 'no' 的欄位作為 Join Key
                # 如果沒有明顯的 ID，才退而求其次使用第一個共同欄位
                join_keys = [c for c in all_common if any(k in c.lower() for k in ['id', 'key', 'no'])]
                
                if not join_keys and all_common:
                    join_keys = [all_common[0]] # 最差情況：拿第一個共同欄位硬上
                    print(f"[DataLoader] ⚠️ 找不到明確的 ID 欄位，退而使用 '{join_keys[0]}' 嘗試合併")

                if join_keys:
                    print(f"[DataLoader] 偵測到 Join Key {join_keys}，執行 Left Join...")
                    # 🚀 加入 suffixes：防止非 Join Key 的共同欄位衝突 (例如兩邊都有 status -> status_main, status_ext)
                    main_df = main_df.merge(df, on=join_keys, how='left', suffixes=('', f'_ext{i}'))
                else:
                    print(f"[DataLoader] ❌ 警告：無法在檔案 {i} 找到與主表的共同欄位，已跳過合併。")

        del dfs
        collected = gc.collect()
        print(f"[DataLoader] 記憶體清理：回收 {collected} 個快取參考。")

        return main_df

    raise ValueError("data_source 格式錯誤！必須是 DataFrame, 字串路徑, 或路徑清單。")