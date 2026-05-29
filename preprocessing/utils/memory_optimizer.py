import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype

def reduce_mem_usage(df, use_float32_safeguard=True):
    """
    自動掃描 DataFrame 的數值與字串欄位進行壓縮。
    :param use_float32_safeguard: 若為 True，則浮點數最低只會壓縮到 float32，避免特徵工程時發生 float16 溢位 (inf)。
    """
    start_mem = df.memory_usage().sum() / 1024**2
<<<<<<< HEAD
    print(f'[記憶體壓縮] 初始佔用大小: {start_mem:.2f} MB')
=======
    print(f'🔧 [記憶體壓縮] 初始佔用大小: {start_mem:.2f} MB')
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
    
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            # 🚀 優化 1: 直接取 min/max，Pandas 預設會 skipna，省去 dropna() 的運算開銷
            c_min = df[col].min()
            c_max = df[col].max()
            
<<<<<<< HEAD
            # 🛡️ 核心防呆：檢查該欄位是否包含 NaN
            has_nan = df[col].isnull().any()
            
=======
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
            # 🚀 優化 3: 使用原生 API 判斷整數
            if is_integer_dtype(df[col]):
                # 🚀 優化 2: 引入 uint (無號整數)，如果全為正數，記憶體省一半
                if c_min >= 0:
                    if c_max < np.iinfo(np.uint8).max - 5:
<<<<<<< HEAD
                        # 🛡️ 動態切換：有 NaN 就用 Pandas 型態 ('UInt8')，否則用 Numpy 型態 (np.uint8)
                        df[col] = df[col].astype('UInt8' if has_nan else np.uint8)
                    elif c_max < np.iinfo(np.uint16).max - 100:
                        df[col] = df[col].astype('UInt16' if has_nan else np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype('UInt32' if has_nan else np.uint32)
                    else:
                        df[col] = df[col].astype('UInt64' if has_nan else np.uint64)
                else:
                    if c_min > np.iinfo(np.int8).min + 5 and c_max < np.iinfo(np.int8).max - 5:
                        df[col] = df[col].astype('Int8' if has_nan else np.int8)
                    elif c_min > np.iinfo(np.int16).min + 100 and c_max < np.iinfo(np.int16).max - 100:
                        df[col] = df[col].astype('Int16' if has_nan else np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype('Int32' if has_nan else np.int32)
                    else:
                        df[col] = df[col].astype('Int64' if has_nan else np.int64)
                        
            # 處理浮點數 (使用原生 API)
            elif is_float_dtype(df[col]):
                # 🚀 優化 (1): 檢查是否所有非 NaN 的數值，其實都是整數
                # 如果是，我們就可以用 Pandas 的 Nullable Integer (大寫開頭) 來壓縮
                if (df[col].dropna() % 1 == 0).all():
                    if c_min >= 0:
                        if c_max < np.iinfo(np.uint8).max:
                            df[col] = df[col].astype('UInt8')
                        elif c_max < np.iinfo(np.uint16).max:
                            df[col] = df[col].astype('UInt16')
                        elif c_max < np.iinfo(np.uint32).max:
                            df[col] = df[col].astype('UInt32')
                        else:
                            df[col] = df[col].astype('UInt64')
                    else:
                        if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                            df[col] = df[col].astype('Int8')
                        elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                            df[col] = df[col].astype('Int16')
                        elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                            df[col] = df[col].astype('Int32')
                        else:
                            df[col] = df[col].astype('Int64')
                
                # 🚀 優化 (2): 如果真的包含小數，則執行原本的浮點數降階邏輯
                else:
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
=======
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
                        
            # 處理浮點數 (使用原生 API)
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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
                        
        # 處理字串 (Object -> Category)
        elif df[col].dtype == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
<<<<<<< HEAD
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
            else:
                # 針對高基數且無法轉 category 的字串，使用 pyarrow 引擎 (需安裝 pyarrow)
                try:
                    df[col] = df[col].astype("string[pyarrow]")
                except:
                    pass # 若環境不支援則保留 object

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'[記憶體壓縮] 壓縮後大小: {end_mem:.2f} MB (減少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
=======
            
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'✅ [記憶體壓縮] 壓縮後大小: {end_mem:.2f} MB (減少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
    
    return df