import pandas as pd
import time
import gc

# 引入你剛修改好的主入口
from preprocessing.interface import preprocess_for_training

def run_test():
    print("==================================================")
    print("🚀 啟動測試：AutoML 2.0 裝甲預處理引擎")
    print("==================================================")
    
    # 1. 設定資料路徑 (請確認你的路徑正確)
    DATA_DIR = 'data/'
    train_files = [f"{DATA_DIR}train_transaction.csv", f"{DATA_DIR}train_identity.csv"]
    
    start_time = time.time()
    
    try:
        # 2. 直接呼叫你的裝甲引擎！
        # 這裡設定 test_size=0.1 讓它切 10% 出來當驗證集測試
        X_train_clean, X_test_clean, y_train, y_test, fitted_preprocessor = preprocess_for_training(
            data_source=train_files, 
            target_col='isFraud', 
            test_size=0.1
        )
        
        print("\n🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟")
        print("✅ 測試大成功！引擎完美運轉，沒有崩潰！")
        print("🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟\n")
        
        print(f"⏱️ 總耗時: {time.time() - start_time:.1f} 秒")
        print(f"📊 產出的訓練集特徵矩陣大小: {X_train_clean.shape}")
        print(f"📊 產出的測試集特徵矩陣大小: {X_test_clean.shape}")
        
        # 抽查一下最終煉金出來的特徵長什麼樣子
        features = fitted_preprocessor.get_feature_names_out()
        print(f"🔍 總特徵數量: {len(features)}")
        print(f"🔍 抽查前 10 個特徵名稱:\n {features[:10]}")
        
        # 檢查是否還有殘留的字串或缺失值
        print("\n🛡️ 防禦盾牌檢查報告：")
        nan_count = X_train_clean.isna().sum().sum()
        print(f"   ➤ 殘留空值 (NaN) 數量: {nan_count} (必須為 0)")
        object_cols = X_train_clean.select_dtypes(include=['object']).columns
        print(f"   ➤ 殘留未處理的字串欄位數: {len(object_cols)} (必須為 0)")

    except Exception as e:
        print("\n❌ 引擎啟動失敗！捕捉到錯誤訊息：")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()