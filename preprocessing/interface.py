# preprocessing/interface.py
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

from .core import AutoRouter, PipelineAssembler
from .utils.data_health import generate_health_report

def run_data_audit(raw_df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    [提供給 UI 組] 產生預處理審查報告 (Preprocessing Audit Report)。
    """
    print(f"\n[健檢中心] 正在針對 {raw_df.shape[0]} 筆資料進行全身健康檢查...")
    report = generate_health_report(raw_df, target_col)
    
    if report["warnings"]:
        print(f"⚠️ 發現 {len(report['warnings'])} 個潛在問題！")
    else:
        print("✅ 資料健康狀況良好！")
        
    return report

def preprocess_for_training(
    raw_df: pd.DataFrame, 
    target_col: str, 
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    [提供給 模型組] 執行端對端的預處理管線。
    包含偵測有害資料、處理缺失值以及特徵提取。
    """
    print("[預處理模組] 開始執行...")
    
    # 1. 切割特徵 (X) 與標籤 (y)
    X = raw_df.drop(columns=[target_col])
    y = raw_df[target_col]
    
    # 2. 嚴格執行時間與空間的切割，防止資料洩漏
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"[預處理模組] 正在針對 {len(X_train_raw)} 筆訓練資料進行分析...")
    
    # 3. 啟動分類大腦 (Router) 掃描訓練集
    # 使用我們設定好的參數：超過 50 種算文字，平均字串長度大於 20 算 NLP 文字
    router = AutoRouter(categorical_threshold=50, text_length_threshold=20)
    feature_groups = router.fit_predict(X_train_raw)
    
    # 4. 啟動組裝廠 (Assembler)
    print("[預處理模組] 正在組裝處理管線...")
    assembler = PipelineAssembler(feature_groups)
    fitted_preprocessor = assembler.build()
    
    # 5. 正式擬合 (Fit) 與轉換 (Transform) 訓練集
    print("[預處理模組] 正在擬合訓練集資料...")
    # 確保這裡是呼叫 fit_transform，這會同時進行「學習」與「轉換」
    X_train_clean_array = fitted_preprocessor.fit_transform(X_train_raw, y_train)
    
    # 取得欄位名稱 (ColumnTransformer 的特有方法)
    feature_names = fitted_preprocessor.get_feature_names_out()
    X_train_clean = pd.DataFrame(X_train_clean_array, columns=feature_names)
    
    # 6. 僅轉換 (Transform) 測試集
    print("[預處理模組] 正在轉換測試集資料...")
    # 這裡絕對不能再寫 fit，只能寫 transform
    X_test_clean_array = fitted_preprocessor.transform(X_test_raw)
    X_test_clean = pd.DataFrame(X_test_clean_array, columns=feature_names)
    
    print("[預處理模組] 處理完成！")
    return X_train_clean, X_test_clean, y_train, y_test, fitted_preprocessor

def preprocess_for_inference(
    new_data_df: pd.DataFrame, 
    fitted_preprocessor: Any
) -> pd.DataFrame:
    """
    [提供給 系統推論/預測使用]
    使用已經擬合好的參數處理新資料，確保結果的一致性。
    """
    # 直接套用之前存下來的轉換規則
    return new_data_df