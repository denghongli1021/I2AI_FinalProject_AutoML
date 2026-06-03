import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_integer_dtype, is_float_dtype

def reduce_mem_usage(df, use_float32_safeguard=True):
    """
    自動掃描 DataFrame 的數值與字串欄位進行壓縮。
    [Sklearn 安全版] 嚴格禁止使用 Pandas Nullable Integer (Int8, UInt8 等)，
    避免產生 pd.NA 導致 Cython / Scikit-Learn 底層崩潰。
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'[記憶體壓縮] 初始佔用大小: {start_mem:.2f} MB')
    
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()
            has_nan = df[col].isnull().any()
            
            # 🛡️ 核心防呆：只要有 NaN，一律強制使用 float 儲存！
            # 因為 Numpy 的整數型態不支援 NaN，而 Pandas 的 Int64 會產出致命的 pd.NA
            if has_nan:
                if use_float32_safeguard:
                    df[col] = df[col].astype(np.float32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    else:
                        df[col] = df[col].astype(np.float32)
                continue  # 處理完直接跳下一個欄位，絕對不進入整數轉換！

            # 🚀 執行到這裡，代表「確定沒有 NaN」，可以安全地壓縮成 Numpy 整數
            # 檢查是否為整數，或是「數值全為整數的浮點數」
            is_int = is_integer_dtype(df[col]) or (is_float_dtype(df[col]) and (df[col].dropna() % 1 == 0).all())
            
            if is_int:
                if c_min >= 0:  # 無號整數 (可省一半記憶體)
                    if c_max < np.iinfo(np.uint8).max:
                        df[col] = df[col].astype(np.uint8)
                    elif c_max < np.iinfo(np.uint16).max:
                        df[col] = df[col].astype(np.uint16)
                    elif c_max < np.iinfo(np.uint32).max:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:           # 有號整數
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    else:
                        df[col] = df[col].astype(np.int64)
            
            # 若真的是帶有小數點的浮點數
            elif is_float_dtype(df[col]):
                if use_float32_safeguard:
                    df[col] = df[col].astype(np.float32)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    else:
                        df[col] = df[col].astype(np.float32)

        # 處理字串 (Object -> Category)
        elif df[col].dtype == 'object':
            num_unique = df[col].nunique()
            num_total = len(df[col])
            if num_unique / num_total < 0.5:
                df[col] = df[col].astype('category')
            else:
                # 🛡️ 安全起見，移除了 string[pyarrow]，因為部分 Scikit-Learn 模組遇到 pyarrow 依然會報錯
                pass 

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'[記憶體壓縮] 壓縮後大小: {end_mem:.2f} MB (減少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    
    return df