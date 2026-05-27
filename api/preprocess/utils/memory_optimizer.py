import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype

def reduce_mem_usage(df, use_float32_safeguard=True):
    """
    自動掃描 DataFrame 的數值與字串欄位進行壓縮。
    :param use_float32_safeguard: 若為 True，則浮點數最低只會壓縮到 float32，避免特徵工程時發生 float16 溢位 (inf)。
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'🔧 [記憶體壓縮] 初始佔用大小: {start_mem:.2f} MB')

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            # Pandas min/max 預設 skipna,省 dropna()
            c_min = df[col].min()
            c_max = df[col].max()

            # 整數:全為正數時用 uint 再省一半
            if is_integer_dtype(df[col]):
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max - 5:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < np.iinfo(np.uint16).max - 100:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min + 5 and c_max < np.iinfo(np.int8).max - 5:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min + 100 and c_max < np.iinfo(np.int16).max - 100:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)

            # 浮點數
            elif is_float_dtype(df[col]):
                if use_float32_safeguard:
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        # 字串 Object → Category (低基數時節省記憶體)
        elif df[col].dtype == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_total > 0 and num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'✅ [記憶體壓縮] 壓縮後大小: {end_mem:.2f} MB (減少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')

    return df