import os
import pandas as pd
from preprocessing.interface import preprocess_for_training

if __name__ == "__main__":
    # 1. 設定真實 Kaggle 檔案路徑
    # 溫馨提醒：請確保這兩個檔案已經下載並放在與 test_run.py 同一個資料夾內
    file_main = "train_transaction.csv"
    file_side = "train_identity.csv" 
    
    # 如果你真的是想測試把 transaction 跟 transaction 自己 Join，
    # 可以把 file_side 改成 "train_transaction.csv"。
    
    csv_files = [file_main, file_side]
    
    # 2. 安全檢查：檔案到底在不在？
    missing_files = [f for f in csv_files if not os.path.exists(f)]
    if missing_files:
        print(f"❌ 找不到真實資料集！請確認以下檔案與 test_run.py 放在同一目錄：\n{missing_files}")
        print("💡 提示：你可以去 Kaggle 下載 'IEEE-CIS Fraud Detection' 的資料集。")
        exit(1)

    print(f"📂 成功尋獲真實資料！準備讀取: {csv_files}")
    print("🚀 啟動 AutoML 2.0 預處理引擎 (真實壓力測試)...\n" + "="*60)
    custom_schema = {
        "TransactionDT": "numeric",     # 救回被誤殺的時間差特徵
        "M4": "categorical",            # 糾正 M4 被誤判為日期的問題
        "DeviceType": "categorical",    # 救回裝置類型特徵
    }
    
    try:
        # 3. 呼叫大腦：把路徑 List 直接餵給引擎
        X_train, X_test, y_train, y_test, preprocessor = preprocess_for_training(
            data_source=csv_files,
            target_col='isFraud',  # Kaggle 這題的目標變數是 isFraud
            test_size=0.2,
            main_file_index=0,     # 指定第 0 個檔案 (transaction) 為主表進行 Left Join
            schema_override=custom_schema
        )
        
        print("\n" + "="*60)
        print("🏆 恭喜！挑戰 Kaggle 魔王成功！引擎成功存活！")
        print(f"✅ 訓練集 X_train 形狀: {X_train.shape}")
        print(f"✅ 產生的特徵範例: {list(X_train.columns)[:10]} ...")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"💀 引擎在真實世界的壓力下崩潰了！請檢查以下錯誤訊息：\n{e}")