# test_run.py
import pandas as pd
import numpy as np
import os
from preprocessing import preprocess_for_training, run_data_audit

if __name__ == "__main__":
    print("="*60)
    print("🚀 AutoML 預處理模組：真實 CSV 實戰測試 (train1.csv) 🚀")
    print("="*60)

    # 1. 設定檔案路徑與目標欄位
    csv_path = 'train1.csv'
    # 💡 請務必確認 train1.csv 裡的標籤欄位名稱，如果是 'Label' 或 'Target' 請修改下方：
    target_column = 'SalePrice' 

    if not os.path.exists(csv_path):
        print(f"❌ 找不到檔案: {csv_path}。請確認檔案已放在專案根目錄。")
    else:
        # 2. 讀取資料
        try:
            # 如果 CSV 有中文字，可以嘗試加上 encoding='utf-8' 或 'big5'
            raw_df = pd.read_csv(csv_path) 
            
            # 💡 關鍵步驟：嘗試將可能是日期的字串轉為 datetime 物件，否則 Router 會判定為 Text 或 Category
            # 如果你知道哪幾欄是日期，可以手動轉換，例如：
            # for col in ['Date', 'Timestamp']:
            #     if col in raw_df.columns:
            #         raw_df[col] = pd.to_datetime(raw_df[col], errors='coerce')

            print(f"\n[載入成功] 資料維度: {raw_df.shape[0]} 筆資料, {raw_df.shape[1]} 個欄位")
            
            # 3. 執行健檢中心
            print("\n" + "-"*30)
            audit_report = run_data_audit(raw_df, target_col=target_column)
            print("[健檢報告摘要]:")
            if audit_report["missing_summary"]:
                for col, count in audit_report["missing_summary"].items():
                    print(f"  - {col}: 缺失 {count} 筆")
            else:
                print("  ✅ 恭喜！原始資料沒有缺失值。")
            
            # 4. 執行預處理管線
            print("\n" + "-"*30)
            X_train, X_test, y_train, y_test, preprocessor = preprocess_for_training(
                raw_df=raw_df, 
                target_col=target_column, 
                test_size=0.2
            )
            
            print("\n" + "="*60)
            print("🎉 預處理完畢！")
            print(f"👉 原始特徵數: {raw_df.shape[1] - 1}")
            print(f"👉 處理後特徵數: {X_train.shape[1]}")
            print("-" * 60)
            print("\n[處理後的資料前 5 筆]:")
            print(X_train.head())
            
            # 驗證最終結果
            final_missing = X_train.isnull().sum().sum()
            print(f"\n[最終檢查] 轉換後矩陣空值總數: {final_missing}")
            if final_missing == 0:
                print("✨ 成果：資料已完全洗淨，可直接餵給模型訓練！")

        except Exception as e:
            print(f"\n❌ 發生錯誤: {e}")
            # 如果發生錯誤，印出更詳細的資訊方便 Debug
            import traceback
            traceback.print_exc()