import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 讀取資料
# ==========================================
# 請確保 CSV 檔案與此腳本在同一目錄下
df = pd.read_csv('時序回歸_DailyDelhiClimateTrain.csv')

# ==========================================
# 2. 資料預處理與特徵工程
# ==========================================
# 轉換日期格式並確保按時間排序 (時序資料切忌打亂)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 萃取時間特徵 (捕捉季節與週期性)
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['year'] = df['date'].dt.year

# 加入「滯後特徵 (Lag Features)」(前一天的數值)
df['meantemp_lag1'] = df['meantemp'].shift(1)
df['humidity_lag1'] = df['humidity'].shift(1)

# 移除因為 shift 產生的第一筆 NaN 值
df = df.dropna()

# 定義特徵變數 (X) 與目標變數 (y)
X = df.drop(columns=['date', 'meantemp'])
y = df['meantemp']

# ==========================================
# 3. 資料切割 (時序切割)
# ==========================================
# 前 80% 作為訓練，後 20% 作為測試
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ==========================================
# 4. 模型選擇與參數優化
# ==========================================
# 建立隨機森林回歸模型
rf = RandomForestRegressor(random_state=42)

# 設定參數搜索範圍
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 使用時間序列交叉驗證避免未來數據洩漏
tscv = TimeSeriesSplit(n_splits=3)

# 使用 RandomizedSearchCV 進行超參數優化
random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=param_dist, 
    n_iter=10, 
    cv=tscv, 
    scoring='r2', 
    random_state=42,
    n_jobs=-1 # 使用所有 CPU 核心加速
)

print("模型訓練與參數優化中，請稍候...")
random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

print(f"\n✅ 找到最佳參數: {random_search.best_params_}")

# ==========================================
# 5. 預測與模型評估
# ==========================================
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print("-" * 40)
print(f"🎯 最終測試集 R2 分數 (R-squared): {r2:.4f}")
print("-" * 40)