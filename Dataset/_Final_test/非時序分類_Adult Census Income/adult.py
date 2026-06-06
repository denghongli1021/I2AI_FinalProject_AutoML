import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

# 1. 讀取與載入資料
print("正在載入資料集...")
df = pd.read_csv('adult.csv')

# 2. 特徵工程與資料清洗
# 移除冗餘文字欄位與高噪聲權重欄位
df = df.drop(columns=['education', 'fnlwgt'])

# 清理目標標籤字串中的空格與點號，並進行二元編碼 (<=50K 設為 0, >50K 設為 1)
df['income'] = df['income'].str.strip().str.replace('.', '', regex=False)
df['target'] = df['income'].map({'<=50K': 0, '>50K': 1})
df = df.drop(columns=['income'])

# 區分特徵矩陣 X 與目標向量 y
X = df.drop(columns=['target'])
y = df['target']

# 識別類別型欄位、填補缺失值，並轉換為 category 型態以觸發原生的樹分裂優化
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].fillna('Unknown').astype('category')

# 建立類別型欄位的布林遮罩 (HistGradientBoosting 識別專用)
categorical_mask = [col in cat_cols for col in X.columns]

# 3. 自動切割資料集 (80% 訓練集, 20% 測試集，並使用分層抽樣保持標籤比例)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 模型初始化與訓練（加入優化參數）
print("模型訓練中，已啟用原生類別特徵優化...")
model = HistGradientBoostingClassifier(
    random_state=42,
    max_iter=200,            # 迭代棵數
    learning_rate=0.05,       # 穩健的學習率
    categorical_features=categorical_mask  # 告訴模型哪些是類別變數
)
model.fit(X_train, y_train)

# 5. 模型預測與多維度評估
# 計算 R2 分數：抽取預測為 1 (>50K) 的連續機率值來計算
y_pred_prob = model.predict_proba(X_test)[:, 1]
r2 = r2_score(y_test, y_pred_prob)

# 同時預測硬分類標籤，計算常規準確率
y_pred_class = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_class)

# 6. 輸出最終結果
print("\n" + "="*30)
print(" 最終模型評估結果")
print("="*30)
print(f"測試集 R2 分數          : {r2:.4f}")
print(f"測試集 分類準確率 (Acc)  : {accuracy * 100:.2f}%")
print("="*30)