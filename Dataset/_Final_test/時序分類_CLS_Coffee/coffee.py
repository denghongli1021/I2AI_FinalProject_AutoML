import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# 1. 讀取資料
# 假設檔案與程式碼在同一目錄下
file_path = 'CLS_Coffee.csv'
print("正在讀取資料...")
df = pd.read_csv(file_path)

# 2. 定義特徵 (X) 與目標變數 (y)
# 這裡自動假設「最後一欄」是我們要預測的目標 (y)，其餘為特徵 (X)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# 3. 切割訓練集與測試集 (80% 訓練, 20% 測試)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"資料切割完成：訓練集包含 {X_train.shape[0]} 筆，測試集包含 {X_test.shape[0]} 筆。")

# 4. 建立自動化機器學習管線 (Pipeline)
# 步驟：標準化 -> 自動特徵選擇 (剔除無用變數) -> 隨機森林迴歸
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42))),
    ('model', RandomForestRegressor(random_state=42))
])

# 5. 定義超參數的優化空間 (優化器會在這裡面找最佳組合)
param_distributions = {
    'feature_selection__threshold': ['median', 'mean'], # 決定保留特徵的嚴格程度
    'model__n_estimators': [100, 200, 300],             # 森林裡樹木的數量
    'model__max_depth': [None, 10, 20, 30],             # 樹的最大深度
    'model__min_samples_split': [2, 5, 10]              # 節點再分裂所需的最小樣本數
}

# 6. 使用 RandomizedSearchCV 進行模型優化與訓練
print("開始進行自動特徵選擇與超參數優化，這可能需要一點時間...")
search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=10,        # 隨機嘗試 10 種參數組合 (可依需求增加以求更高精確度)
    cv=5,             # 5 折交叉驗證
    scoring='r2',     # 優化目標為 R2 分數
    random_state=42,
    n_jobs=-1         # 使用所有 CPU 核心加速計算
)

search.fit(X_train, y_train)

# 7. 預測與評估
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# 觀察最終保留了多少個變數
selected_features_mask = best_model.named_steps['feature_selection'].get_support()
num_selected_features = selected_features_mask.sum()
total_features = X.shape[1]

print("\n" + "="*40)
print("🎯 訓練完成！結果報告：")
print("="*40)
print(f"保留變數數量: {num_selected_features} / {total_features} (已自動剔除 {total_features - num_selected_features} 個低貢獻變數)")
print(f"最佳模型參數: {search.best_params_}")
print(f"🌟 最終測試集 R2 分數: {r2:.4f}")
print("="*40)