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
# 2. 資料清洗與時間特徵工程
# ==========================================
# 剔除目標變數 num_sold 為空的欄位，確保訓練品質
df_train = df_train.dropna(subset=['num_sold']).copy()

# 將 id 欄位移除 (若存在)
if 'id' in df_train.columns:
    df_train = df_train.drop(columns=['id'])

# 核心時間特徵轉換
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['year'] = df_train['date'].dt.year
df_train['month'] = df_train['date'].dt.month
df_train['day'] = df_train['date'].dt.day
df_train['dayofweek'] = df_train['date'].dt.dayofweek

# 區分特徵與目標
X = df_train.drop(columns=['date', 'num_sold'])
y = df_train['num_sold']

# 類別變數處理與編碼器儲存
categorical_cols = ['country', 'store', 'product']
encoders = {}
X_processed = X.copy()

for col in categorical_cols:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_processed[col] = encoder.fit_transform(X_processed[[col]])
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

print("新模型訓練中...")
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
    print("偵測到 'test.csv'，開始進行銷量預測...")
    
    # 保留 id 供最後輸出使用
    if 'id' in df_test.columns:
        test_ids = df_test['id']
    else:
        test_ids = range(1, len(df_test) + 1)
        
    # 同步對測試集進行時間特徵轉換
    df_test['date'] = pd.to_datetime(df_test['date'])
    df_test['year'] = df_test['date'].dt.year
    df_test['month'] = df_test['date'].dt.month
    df_test['day'] = df_test['date'].dt.day
    df_test['dayofweek'] = df_test['date'].dt.dayofweek

    # 套用訓練好的編碼器
    X_test_processed = df_test.copy()
    for col in categorical_cols:
        X_test_processed[col] = X_test_processed[col].fillna('Missing')
        X_test_processed[col] = encoders[col].transform(X_test_processed[[col]])

    # 確保特徵順序與訓練集完全一致
    X_test_processed = X_test_processed[X_processed.columns]

    # 預測銷量
    test_predictions = model.predict(X_test_processed)

    # 打包匯出
    submission = pd.DataFrame({
        'id': test_ids,
        'num_sold': test_predictions
    })
    
    output_filename = 'submission.csv'
    submission.to_csv(output_filename, index=False)
    print(f"🎉 預測成功！銷量預測結果已輸出至 '{output_filename}'")
    print(f"檔案預覽:\n{submission.head()}")

except FileNotFoundError:
    print("說明：未在當前目錄下找到 'test.csv'，因此跳過測試集預測步驟。")