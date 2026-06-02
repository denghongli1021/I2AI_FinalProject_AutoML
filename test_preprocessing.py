import os
import time
import argparse
import numpy as np
import pandas as pd

# ==========================================
# 📥 1. 匯入介面 (Facade API)
# ==========================================
try:
    from preprocessing.interface import (
        run_data_audit,
        preprocess_for_training,
        preprocess_for_inference
    )
    # 預先載入 DataLoader 來處理多檔合併，方便給 Audit 用
    from preprocessing.data_loader import load_and_merge_data
except ImportError as e:
    print(f"⚠️ 匯入模組失敗，請確認檔案路徑是否正確: {e}")
    print("請確保你在專案根目錄執行此腳本。")

# ==========================================
# 🚀 2. 主測試流程
# ==========================================
def main(args):
    print("="*70)
    print(" 🧪 AutoML 預處理模組 [全功能介面 API] 整合測試")
    print("="*70)
    
    # ---------------------------------------------------------
    # 🎯 設定測試環境 (接收命令列參數)
    # ---------------------------------------------------------
    train_files = args.train
    test_files = args.test
    target_col = args.target

    # 🛡️ 防呆：如果沒有給定訓練集，或者檔案不存在，自動生成假資料
    if not train_files or not os.path.exists(train_files[0]):
        print(f"⚠️ 未提供訓練集或檔案不存在，自動生成輕量級測試資料...")
        df_mock = pd.DataFrame({
            "age": [25, 30, np.nan, 22, 45, 50, 31, 29, 35, 40],
            "salary": [50000, 60000, 55000, 120000, np.nan, 80000, 65000, 70000, 90000, 100000],
            "city": ["Taipei", "Hsinchu", "Taichung", "Taipei", "?", "Tainan", "Hsinchu", "Taipei", "Taichung", "Tainan"],
            "target": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0] # 假目標
        })
        df_mock.to_csv("train1.csv", index=False)
        df_mock.drop(columns=["target"]).to_csv("test1.csv", index=False)
        
        train_files = ["train1.csv"]
        test_files = ["test1.csv"]
        target_col = "target"
        print("✅ 假資料生成完畢！")
    else:
        print(f"📂 接收到訓練集檔案: {train_files}")
        if test_files:
            print(f"📂 接收到測試集檔案: {test_files}")
        print(f"🎯 目標欄位設定為: {target_col}")

    # =========================================================
    # 🏥 階段一：啟動資料健檢中心 (run_data_audit)
    # =========================================================
    print(f"\n>>> [API 測試 1] 啟動前端 UI 健檢報告 (run_data_audit)")
    t0 = time.time()
    
    # 為了支援多檔案健檢，我們先呼叫底層的 load_and_merge_data 把它們接起來
    raw_for_audit = load_and_merge_data(train_files) 
    
    try:
        report = run_data_audit(
            raw_df=raw_for_audit, 
            target_col=target_col,
            export_html_path="audit_report.html",  # 自動產生 HTML！
            export_json_path="audit_report.json"
        )
        print(f"📊 報告摘要: 總列數 {report.get('total_rows')}, 總欄數 {report.get('total_columns')}")
        print(f"⚠️ 潛在警告數: {len(report.get('warnings', []))}")
        print(f"⏱️ 健檢耗時: {time.time() - t0:.2f} 秒")
    except Exception as e:
        print(f"❌ 健檢中心執行失敗: {e}")
        print("👉 如果你還沒寫完 data_health.py，可以先略過這個錯誤。")

    # =========================================================
    # 🚂 階段二：模型訓練組雙軌前處理 (preprocess_for_training)
    # =========================================================
    print(f"\n>>> [API 測試 2] 啟動訓練前處理 (preprocess_for_training)")
    t1 = time.time()
    
    # 🚨 傳入的是 List[str]，介面會自動處理多檔合併
    X_train_dict, X_test_dict, y_train, y_test, preprocessors = preprocess_for_training(
        data_source=train_files, 
        target_col=target_col,
        test_data_source=test_files,  # 👈 支援多個真實測試集！
        enable_adv_val=True,          # 👈 開啟對抗驗證防護盾！
        test_size=0.2 
    )
    print(f"⏱️ 訓練前處理耗時: {time.time() - t1:.2f} 秒\n")
    
    # 🔎 驗證雙軌結果，並印出詳細執行成果
    print(" 🏆 [雙軌驗證與結果印出]")
    tree_df = X_train_dict["tree"]
    dl_df = X_train_dict["dl"]
    
    print(f"  🌳 Tree (生肉) 形狀: {tree_df.shape} | 殘留 NaN 數: {tree_df.isna().sum().sum()} (有 NaN 是好事！)")
    print("  👇 Tree 軌道前 3 筆轉換結果:")
    print(tree_df.head(3))
    
    print(f"\n  🧠 DL   (熟肉) 形狀: {dl_df.shape} | 殘留 NaN 數: {dl_df.isna().sum().sum()} (應該要是 0)")
    print("  👇 DL 軌道前 3 筆轉換結果:")
    print(dl_df.head(3))
    print(f"\n  🎯 Label (y_train) 長度: {len(y_train)}")

    # =========================================================
    # 🚀 階段三：上線部署組預測推論 (preprocess_for_inference)
    # =========================================================
    print(f"\n>>> [API 測試 3] 啟動推論預測前處理 (preprocess_for_inference)")
    if not test_files:
        print(f"⚠️ 未提供測試集，直接拿 Train 集來假裝推論...")
        test_files = train_files

    t2 = time.time()
    
    # 測試推論 DL 軌道
    fitted_dl_preprocessor = preprocessors["dl"]
    
    # 從訓練好的管線提取特徵名單
    training_features = list(fitted_dl_preprocessor.feature_names_in_)
    
    X_inference_dl = preprocess_for_inference(
        data_source=test_files, # 👈 支援多檔推論
        fitted_preprocessor=fitted_dl_preprocessor,
        training_features=training_features
    )
    
    print(f"⏱️ 推論前處理耗時: {time.time() - t2:.2f} 秒\n")
    
    # 🔎 驗證推論結果並印出
    print(" 🏆 [推論驗證與結果印出]")
    print(f"  ✅ 推論 DL 矩陣形狀: {X_inference_dl.shape}")
    print(f"  ✅ 是否與訓練集欄位數目對齊？: {'是' if X_inference_dl.shape[1] == dl_df.shape[1] else '否 ❌'}")
    print("  👇 推論用 DL 矩陣前 3 筆資料:")
    print(X_inference_dl.head(3))
    
    print("\n" + "="*70)
    print(" 🎉 恭喜！介面 API (Facade) 全功能測試完美通過！")
    print("="*70)

if __name__ == "__main__":
    # 🛠️ 建立指令列參數解析器
    parser = argparse.ArgumentParser(description="AutoML 預處理模組整合測試腳本")
    
    # nargs='+' 允許傳入多個參數，並自動轉為 List[str]
    parser.add_argument("--train", nargs='+', help="輸入訓練集檔案路徑 (可傳入多個，以空白分隔)")
    parser.add_argument("--test", nargs='+', help="輸入測試集檔案路徑 (可傳入多個，以空白分隔)")
    parser.add_argument("--target", type=str, default="SalePrice", help="預測目標的欄位名稱 (預設: SalePrice)")
    
    args = parser.parse_args()
    
    main(args)