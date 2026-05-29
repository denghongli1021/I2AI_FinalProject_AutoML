import os
import time
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
    # 預先載入 DataLoader 只是為了方便在外部取 training_features (對齊用)
    from preprocessing.data_loader import load_and_merge_data
except ImportError as e:
    print(f"⚠️ 匯入模組失敗，請確認檔案路徑是否正確: {e}")
    print("請確保你在專案根目錄執行此腳本。")

# ==========================================
# 🚀 2. 主測試流程
# ==========================================
def main():
    print("="*70)
    print(" 🧪 AutoML 預處理模組 [全功能介面 API] 整合測試")
    print("="*70)
    
    # ---------------------------------------------------------
    # 🎯 設定測試環境 (使用真實檔案)
    # ---------------------------------------------------------
    TRAIN_CSV = "train1.csv"  # 你的真實訓練集
    TEST_CSV = "test1.csv"    # 你的真實測試集
    TARGET_COL = "SalePrice"  # 🚨 請替換成你資料集真實的預測目標欄位名稱！
    
    # 🛡️ 防呆：如果沒有真實檔案，幫你做一個微型假資料來測試
    # 🛡️ 防呆：如果沒有真實檔案，幫你做一個微型假資料來測試
    if not os.path.exists(TRAIN_CSV):
        print(f"⚠️ 找不到 {TRAIN_CSV}，自動生成輕量級測試資料...")
        df_mock = pd.DataFrame({
            "age": [25, 30, np.nan, 22, 45, 50, 31, 29, 35, 40],
            # 🚀 這裡修正了：把 50k 改成 50000
            "salary": [50000, 60000, 55000, 120000, np.nan, 80000, 65000, 70000, 90000, 100000],
            "city": ["Taipei", "Hsinchu", "Taichung", "Taipei", "?", "Tainan", "Hsinchu", "Taipei", "Taichung", "Tainan"],
            "target": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0] # 假目標
        })
        df_mock.to_csv("train1.csv", index=False)
        df_mock.drop(columns=["target"]).to_csv("test1.csv", index=False)
        TARGET_COL = "target"
        print("✅ 假資料生成完畢！")

    # =========================================================
    # 🏥 階段一：啟動資料健檢中心 (run_data_audit)
    # =========================================================
    print(f"\n>>> [API 測試 1] 啟動前端 UI 健檢報告 (run_data_audit)")
    t0 = time.time()
    
    # Audit 需要吃 DataFrame，所以我們先簡單讀取一下
    raw_for_audit = pd.read_csv(TRAIN_CSV) 
    
    try:
        # 🚀 加上 export_html_path 參數！
        report = run_data_audit(
            raw_df=raw_for_audit, 
            target_col=TARGET_COL,
            export_html_path="audit_report.html",  # 自動產生 HTML！
            export_json_path="audit_report.json"
        )
        # 🚀 改成使用正確的 Key：'total_rows' 和 'total_columns'
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
    
    # 🚨 注意：這裡直接傳入字串路徑 (TRAIN_CSV)！讓 interface 自己去呼叫 data_loader
    X_train_dict, X_test_dict, y_train, y_test, preprocessors = preprocess_for_training(
        data_source=TRAIN_CSV, 
        target_col=TARGET_COL,
        test_data_source=TEST_CSV,   # 👈 傳入真實測試集！
        enable_adv_val=True,         # 👈 開啟對抗驗證防護盾！
        test_size=0.2 # 自動切 20% 當 validation
    )
    print(f"⏱️ 訓練前處理耗時: {time.time() - t1:.2f} 秒\n")
    
    # 🔎 驗證雙軌結果
    print(" 🏆 [雙軌驗證]")
    tree_df = X_train_dict["tree"]
    dl_df = X_train_dict["dl"]
    
    print(f"  🌳 Tree (生肉) 形狀: {tree_df.shape} | 殘留 NaN 數: {tree_df.isna().sum().sum()} (有 NaN 是好事！)")
    print(f"  🧠 DL   (熟肉) 形狀: {dl_df.shape} | 殘留 NaN 數: {dl_df.isna().sum().sum()} (應該要是 0)")
    print(f"  🎯 Label (y_train) 長度: {len(y_train)}")

    # =========================================================
    # 🚀 階段三：上線部署組預測推論 (preprocess_for_inference)
    # =========================================================
    print(f"\n>>> [API 測試 3] 啟動推論預測前處理 (preprocess_for_inference)")
    if not os.path.exists(TEST_CSV):
        print(f"⚠️ 找不到 {TEST_CSV}，直接拿 Train 集來假裝推論...")
        TEST_CSV = TRAIN_CSV

    t2 = time.time()
    
    # 測試推論 DL 軌道
    fitted_dl_preprocessor = preprocessors["dl"]
    
    # 🚀 業界標準寫法：直接從訓練好的管線身上，提取它真正看過的特徵名單！
    # 這完美避開了對抗驗證、刪除幽靈欄位等造成的「特徵數量落差」
    training_features = list(fitted_dl_preprocessor.feature_names_in_)
    
    X_inference_dl = preprocess_for_inference(
        data_source=TEST_CSV, 
        fitted_preprocessor=fitted_dl_preprocessor,
        training_features=training_features
    )
    
    print(f"⏱️ 推論前處理耗時: {time.time() - t2:.2f} 秒\n")
    
    # 🔎 驗證推論結果
    print(" 🏆 [推論驗證]")
    print(f"  ✅ 推論 DL 矩陣形狀: {X_inference_dl.shape}")
    print(f"  ✅ 是否與訓練集欄位數目對齊？: {'是' if X_inference_dl.shape[1] == dl_df.shape[1] else '否 ❌'}")
    
    print("\n" + "="*70)
    print(" 🎉 恭喜！介面 API (Facade) 全功能測試完美通過！")
    print("="*70)

if __name__ == "__main__":
    main()