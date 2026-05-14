# test_run.py
import warnings
import pandas as pd

# 關閉未來版本的 Pandas 警告，讓終端機輸出更乾淨
warnings.simplefilter(action='ignore', category=FutureWarning)

from preprocessing.interface import run_data_audit, preprocess_for_training
from preprocessing.utils.data_health import print_health_report

if __name__ == "__main__":
    print("="*70)
    print("🚀 AutoML Preprocessing Engine - 真實資料實戰測試")
    print("="*70)
    
    # ==========================================
    # 1. 讀取真實資料
    # ==========================================
    try:
        # 請確認 train1.csv 放在與這個腳本相同的目錄下
        raw_df = pd.read_csv('train1.csv')
        print(f"✅ 成功讀取 train1.csv，共 {raw_df.shape[0]} 列 × {raw_df.shape[1]} 欄")
    except FileNotFoundError:
        print("❌ 找不到 train1.csv，請確認檔案路徑是否正確！")
        exit()

    # ⚠️ 這裡非常重要：請把 'Your_Target_Column' 換成你 train1.csv 裡面的真實「標籤/預測目標」欄位名稱
    # 例如：'label', 'is_malware', 'click', 'price' 等等
    TARGET_COL = 'SalePrice'  
    
    if TARGET_COL not in raw_df.columns:
        print(f"\n❌ 錯誤：在資料集中找不到目標欄位 '{TARGET_COL}'。")
        print(f"現有欄位包含: {list(raw_df.columns)[:10]} ...等")
        print("👉 請修改 test_run.py 裡的 TARGET_COL 變數！")
        exit()

    # ==========================================
    # Phase 1: 健檢中心 (Data Audit)
    # ==========================================
    print("\n\n>>> 🟢 Phase 1: 啟動健檢中心")
    report = run_data_audit(raw_df, target_col=TARGET_COL)
    print_health_report(report)
    
    # ==========================================
    # Phase 2: 訓練管線 (Training Pipeline)
    # ==========================================
    print("\n\n>>> 🔵 Phase 2: 啟動訓練管線")
    
    try:
        # 如果有特定欄位你想強制覆蓋處理邏輯，可以寫在這裡
        # config = {'某個郵遞區號欄位': 'high_cardinality'}
        config = {}
        
        X_train, X_test, y_train, y_test, preprocessor = preprocess_for_training(
            raw_df=raw_df,
            target_col=TARGET_COL,
            test_size=0.2,
            schema_override=config
        )
        
        print("\n✅ 訓練管線執行成功！")
        print(f"   [原始資料維度]: {raw_df.shape[0]} 列 × {raw_df.shape[1]-1} 欄 (不含目標)")
        print(f"   [最終訓練特徵]: {X_train.shape[0]} 列 × {X_train.shape[1]} 欄")
        
        print("\n   [X_train 預覽 (前 5 筆, 顯示前 8 個特徵)]:")
        print(X_train.iloc[:, :8].head(5).round(3).to_string())
        
    except Exception as e:
        print(f"\n❌ 訓練管線執行失敗: {e}")