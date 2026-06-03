import pandas as pd
import gc  # 🆕 引入 Python 內建的垃圾回收模組
from typing import List, Union
import re

from .utils.memory_optimizer import reduce_mem_usage

def load_and_merge_data(data_source: Union[pd.DataFrame, str, List[str]], main_file_index: int = 0) -> pd.DataFrame:
    """
    AutoML 智慧資料載入器 (極速防彈版)
    """
    if isinstance(data_source, pd.DataFrame):
        print("[DataLoader] 偵測到 DataFrame 輸入，進行記憶體壓縮...")
        return reduce_mem_usage(data_source)

    if isinstance(data_source, str):
        print(f"[DataLoader] 讀取單一檔案: {data_source}")
        df = pd.read_csv(data_source)
        return reduce_mem_usage(df)

    if isinstance(data_source, list):
        print(f"[DataLoader] 偵測到 {len(data_source)} 個檔案，準備進行智慧合併...")
        dfs = [pd.read_csv(p) for p in data_source]
        dfs = [reduce_mem_usage(df) for df in dfs]
        
        main_df = dfs[main_file_index]

        for i, df in enumerate(dfs):
            if i == main_file_index: continue

            all_common = list(set(main_df.columns) & set(df.columns))
            
            # 🛡️ 修正地雷 2：使用正則表達式確保 'id', 'key', 'no' 是獨立詞彙或後綴，避免誤判 'valid'
            # 匹配: 開頭或結尾是 id/key/no，或者帶有底線 _id, id_
            join_keys = [
                c for c in all_common 
                if re.search(r'(^|_)(id|key|no)($|_)', c.lower())
            ]
            
            if not join_keys and all_common:
                join_keys = [all_common[0]] 
                print(f"[DataLoader] ⚠️ 找不到明確的 ID 欄位，退而使用 '{join_keys[0]}' 嘗試合併")

            if join_keys:
                print(f"[DataLoader] 偵測到 Join Key {join_keys}，準備執行合併...")
                
                # 🛡️ 修正地雷 1：剔除副表中 Join Key 是 NaN 的幽靈資料，防止笛卡爾積爆炸 OOM
                initial_len = len(df)
                df = df.dropna(subset=join_keys)
                if len(df) < initial_len:
                    print(f"[DataLoader] ⚠️ 已清理副表中 {initial_len - len(df)} 筆 Join Key 為空的資料。")
                
                if df.duplicated(subset=join_keys).any():
                    print(f"[DataLoader] ⚠️ 偵測到副表存在重複的 Key！啟動高速聚合 (Fast Aggregation)...")
                    
                    # 🛡️ 修正地雷 3：字串取非空的 'first'
                    # 自訂一個極速過濾函數，確保拿到的是真正的字串而非 NaN
                    def first_valid(series):
                        val = series.dropna().head(1)
                        return val.iloc[0] if not val.empty else None

                    agg_funcs = {}
                    for col in df.columns:
                        if col not in join_keys:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                agg_funcs[col] = 'mean'  
                            else:
                                agg_funcs[col] = first_valid # 使用安全函數
                    
                    if agg_funcs:
                        df = df.groupby(join_keys).agg(agg_funcs).reset_index()
                else:
                    print(f"[DataLoader] ⚡ 副表 Key 唯一，直接啟動光速合併！")

                suffix = f'_ext{i}'
                rename_dict = {
                    col: f"{col}{suffix}" 
                    for col in df.columns 
                    if col in main_df.columns and col not in join_keys
                }
                if rename_dict:
                    df = df.rename(columns=rename_dict)

                main_df = main_df.merge(df, on=join_keys, how='left')
            else:
                print(f"[DataLoader] ❌ 警告：無法在檔案 {i} 找到與主表的共同欄位，已跳過合併。")

        del dfs
        collected = gc.collect()
        print(f"[DataLoader] 記憶體清理：回收 {collected} 個快取參考。")

        return main_df

    raise ValueError("data_source 格式錯誤！必須是 DataFrame, 字串路徑, 或路徑清單。")