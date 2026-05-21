import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import gc
import time
import warnings

# 引入你的神級 Data Loader
from preprocessing.data_loader import load_and_merge_data

warnings.filterwarnings('ignore')

TARGET_COL = 'isFraud'
ID_COL = 'TransactionID'
DATA_DIR = 'data/'

def run_naive_baseline():
    print("==================================================")
    print("🔵 啟動 藍方: V1.0 暴力對照組 (Naive Baseline)")
    print("==================================================")
    
    # 1. 讀取資料 (新增讀取 Test 資料)
    print("📥 [1/4] 讀取 Train 與 Test 原始資料...")
    train_files = [f"{DATA_DIR}train_transaction.csv", f"{DATA_DIR}train_identity.csv"]
    test_files = [f"{DATA_DIR}test_transaction.csv", f"{DATA_DIR}test_identity.csv"]
    
    train_raw = load_and_merge_data(train_files)
    test_raw = load_and_merge_data(test_files)
    
    # 備份測試集的 ID，準備最後生成 submission 使用
    test_ids = test_raw[ID_COL].copy()

    # 2. 最暴力的預處理 (Train 與 Test 同步處理)
    print("🔨 [2/4] 執行最暴力的預處理 (特徵型態對齊)...")
    y_train = train_raw[TARGET_COL]
    
    # 移除不需要進模型的欄位
    X_train = train_raw.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
    X_test = test_raw.drop(columns=[TARGET_COL, ID_COL], errors='ignore')
    
    # 過河拆橋，釋放原始大表
    del train_raw, test_raw; gc.collect()

    # 防呆機制：確保 Train 和 Test 的欄位順序完全一致
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    # 🎯 終極防漏網之魚：掃描每一個欄位
    for col in X_train.columns:
        col_type = str(X_train[col].dtype).lower()
        
        # 如果這個欄位不是整數、不是浮點數、也不是布林值 (代表它是 object, string 等雜七雜八的型態)
        if 'int' not in col_type and 'float' not in col_type and 'bool' not in col_type:
            # 暴力解法：強制轉成純字串 (把隱藏的 NaN 變成 'nan')，再統一轉成 category
            X_train[col] = X_train[col].astype(str).astype('category')
            X_test[col] = X_test[col].astype(str).astype('category')
        else:
            # 數值欄位，暴力補 -999
            X_train[col] = X_train[col].fillna(-999)
            X_test[col] = X_test[col].fillna(-999)

    # 3. 嚴格的時間序列切割 (OOT)
    print("⏳ [3/4] 嚴格時間序列切割 (前 80% 訓練, 後 20% 驗證)...")
    split_idx = int(len(X_train) * 0.8)
    Xt, Xv = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    yt, yv = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
    
    train_data = lgb.Dataset(Xt, label=yt)
    valid_data = lgb.Dataset(Xv, label=yv, reference=train_data)

    # 🎯 對齊紅方：保證模型腦袋結構完全相同，純粹比較「資料預處理」的威力
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,           
        'max_depth': 8,             
        'min_child_samples': 100,   
        'reg_alpha': 0.1,           
        'reg_lambda': 0.1,          
        'n_jobs': -1,
        'random_state': 42,
        'verbose': -1
    }

    print(f"\n🚀 開始訓練 V1.0 LightGBM (原始特徵數量: {X_train.shape[1]})...")
    start_time = time.time()
    clf = lgb.train(
        params, 
        train_data, 
        num_boost_round=1500, 
        valid_sets=[train_data, valid_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    val_preds = clf.predict(Xv)
    val_auc = roc_auc_score(yv, val_preds)
    
    print("\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
    print(f"✅ V1.0 暴力對照組 訓練完成！耗時: {time.time() - start_time:.1f} 秒")
    print(f"🏆 [V1.0 藍方] 真實 OOT AUC : {val_auc:.5f}")
    print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")

    # 4. 預測 Test Set 並產出檔案
    print("\n🔮 [4/4] 正在預測 Kaggle Test Set 並產出繳交檔...")
    test_preds = clf.predict(X_test)
    
    # 檔名自動附上 Validation AUC 分數，方便你管理多個 submission
    filename = f"submission_v1_baseline_auc_{val_auc:.4f}.csv"
    submission = pd.DataFrame({ID_COL: test_ids, TARGET_COL: test_preds})
    submission.to_csv(filename, index=False)
    print(f"💾 已儲存 V1.0 預測結果 -> {filename}")

if __name__ == "__main__":
    run_naive_baseline()