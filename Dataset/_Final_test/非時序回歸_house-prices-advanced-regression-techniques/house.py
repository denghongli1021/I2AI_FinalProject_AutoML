import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score

# ==========================================
# 1. 讀取資料
# ==========================================
try:
    df_train = pd.read_csv('train.csv')
except FileNotFoundError:
    raise FileNotFoundError("請確保 'train.csv' 檔案與此腳本放在同一個資料夾內。")

# ==========================================
# 2. 訓練集特徵清洗與編碼
# ==========================================
if 'Id' in df_train.columns:
    df_train = df_train.drop(columns=['Id'])

X = df_train.drop(columns=['SalePrice'])
y = df_train['SalePrice']

# 用字典來儲存每一欄的編碼器，以便後續套用到 test.csv
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
encoders = {}
X_processed = X.copy()

for col in categorical_cols:
    X_processed[col] = X_processed[col].fillna('Missing')
    # 建立編碼器
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_processed[col] = encoder.fit_transform(X_processed[[col]])
    # 存起來留給測試集用
    encoders[col] = encoder

# ==========================================
# 3. 資料集切割與模型訓練
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_depth=6,
    l2_regularization=1.5,
    random_state=42
)

print("模型訓練中...")
model.fit(X_train, y_train)

# 驗證集 R2 評估
y_pred = model.predict(X_val)
r2 = r2_score(y_val, y_pred)
print("-" * 30)
print(f"【本地驗證結果】驗證集 R² 分數: {r2:.4f}")
print("-" * 30)

# ==========================================
# 4. 偵測並處理 test.csv 輸出
# ==========================================
try:
    df_test = pd.read_csv('test.csv')
    print("偵測到 'test.csv'，開始進行測試集預測...")
    
    # 1. 保留 Id 供最後輸出使用
    if 'Id' in df_test.columns:
        test_ids = df_test['Id']
        X_test = df_test.drop(columns=['Id'])
    else:
        test_ids = range(1, len(df_test) + 1)
        X_test = df_test.copy()
        
    # 2. 如果 test 裡面有 SalePrice 欄位則剔除 (防呆)
    if 'SalePrice' in X_test.columns:
        X_test = X_test.drop(columns=['SalePrice'])

    # 3. 完全同步訓練集的類別變換 (使用剛剛存下來的 encoders)
    X_test_processed = X_test.copy()
    for col in categorical_cols:
        if col in X_test_processed.columns:
            X_test_processed[col] = X_test_processed[col].fillna('Missing')
            # 注意：這裡是用 transform，絕對不能用 fit_transform
            X_test_processed[col] = encoders[col].transform(X_test_processed[[col]])

    # 4. 確保測試集的特徵欄位順序跟訓練時完全一致
    X_test_processed = X_test_processed[X_processed.columns]

    # 5. 預測房價
    test_predictions = model.predict(X_test_processed)

    # 6. 打包匯出
    submission = pd.DataFrame({
        'Id': test_ids,
        'SalePrice': test_predictions
    })
    
    output_filename = 'submission.csv'
    submission.to_csv(output_filename, index=False)
    print(f"🎉 預測成功！已成功輸出檔案至 '{output_filename}'")
    print(f"檔案預覽:\n{submission.head()}")

except FileNotFoundError:
    print("說明：未在當前目錄下找到 'test.csv'，因此跳過測試集預測步驟。")