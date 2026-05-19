import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def reduce_mem_usage(df, use_float32_safeguard=True):
    """
    自動掃描 DataFrame 的數值欄位進行壓縮。
    :param use_float32_safeguard: 若為 True，則浮點數最低只會壓縮到 float32，避免特徵工程時發生 float16 溢位 (inf)。
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'🔧 [記憶體壓縮] 初始佔用大小: {start_mem:.2f} MB')
    
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            col_type = df[col].dtype
            
            c_min = df[col].dropna().min()
            c_max = df[col].dropna().max()
            
            # 處理整數 (給予嚴格的邊界，確保不會剛好卡在邊緣)
            if str(col_type)[:3] == 'int':
                # 我們不使用 >= 或 <=，保留一點點運算的 buffer
                if c_min > np.iinfo(np.int8).min + 5 and c_max < np.iinfo(np.int8).max - 5:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min + 100 and c_max < np.iinfo(np.int16).max - 100:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            
            # 處理浮點數
            else:
                if use_float32_safeguard:
                    # ML 業界安全標準：浮點數一律最低壓到 float32 就好
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                else:
                    # 激進壓縮 (風險自負)
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'✅ [記憶體壓縮] 壓縮後大小: {end_mem:.2f} MB (減少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df