# preprocessing/utils/data_health.py
import pandas as pd
from typing import Dict, Any

def generate_health_report(df: pd.DataFrame, target_col: str = None) -> Dict[str, Any]:
    """
    掃描 DataFrame，產出供 UI 使用的健康診斷報告。
    """
    report = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "missing_summary": {},
        "perfect_columns": 0,
        "warnings": []
    }
    
    # 1. 檢查缺失值
    missing_counts = df.isnull().sum()
    missing_only = missing_counts[missing_counts > 0]
    
    report["perfect_columns"] = df.shape[1] - len(missing_only)
    
    if not missing_only.empty:
        # 將缺失狀況轉為 Dict，並計算缺失率
        for col, count in missing_only.items():
            ratio = count / df.shape[0]
            report["missing_summary"][col] = {
                "count": int(count),
                "ratio": float(ratio)
            }
            # 如果缺失率大於 30%，加入警告清單
            if ratio > 0.3:
                report["warnings"].append(f"欄位 '{col}' 缺失率高達 {ratio:.1%}，請特別留意。")
                
    # 2. 檢查目標欄位是否存在 (如果有給定的話)
    if target_col and target_col not in df.columns:
        report["warnings"].append(f"致命錯誤：找不到預測目標欄位 '{target_col}'！")
    
    # 修改最後的判定，即使只是微量缺失也要讓使用者知道
    if not missing_only.empty:
        report["warnings"].append(f"注意：偵測到 {len(missing_only)} 個欄位有缺失值，系統將自動填補。")
        
    return report