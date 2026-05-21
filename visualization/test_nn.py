"""
神經網路測試：驗證 AutoMLVisualizer 的 fallback 路徑
-------------------------------------------------------
用 PyTorch MLP 訓練一個簡單分類模型，
確認 visualizer.py 能自動偵測到非樹模型並切換到 shap.Explainer。

資料集：HR Attrition（Kaggle CSV）
任務：員工離職預測（分類）
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from visualizer import AutoMLVisualizer


# ── 1. 定義 PyTorch MLP ───────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ── 2. 包裝成 sklearn 相容介面 ────────────────────────────────────
# shap.Explainer 需要模型有 predict 方法，且能接受 np.ndarray 輸入
class SklearnWrapper:
    """
    把 PyTorch 模型包裝成 sklearn 相容介面。
    這是模型訓練組需要提供給可視化模組的 wrapper。
    """
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
        self.model.eval()

    def predict(self, X):
        # 接受 np.ndarray 或 pd.DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            proba = self.model(X_tensor).numpy().flatten()
        # 回傳二元預測（0/1）
        return (proba > 0.5).astype(int)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            proba = self.model(X_tensor).numpy().flatten()
        return np.column_stack([1 - proba, proba])

    def __call__(self, X):
        # shap.Explainer 需要模型是 callable，這裡代理到 predict
        return self.predict(X)


# ── 3. 主程式 ─────────────────────────────────────────────────────
def run_nn_test():
    print("Step 1: 載入 HR Attrition 資料集...")
    df = pd.read_csv('dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    df['Target'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.select_dtypes(include=['int64', 'float64']).drop(['Target'], axis=1)
    y = df['Target']
    print(f"  資料規模：{X.shape[0]} 筆 × {X.shape[1]} 個特徵")

    print("\nStep 2: 切分資料集並標準化...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\nStep 3: 訓練 PyTorch MLP...")
    input_dim = X_train.shape[1]
    model = MLP(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # 轉成 tensor
    X_tr = torch.FloatTensor(X_train_scaled)
    y_tr = torch.FloatTensor(y_train.values).unsqueeze(1)
    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(30):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/30, Loss: {total_loss/len(loader):.4f}")

    # 評估
    model.eval()
    with torch.no_grad():
        preds = (model(torch.FloatTensor(X_test_scaled)).numpy().flatten() > 0.5).astype(int)
    acc = (preds == y_test.values).mean()
    print(f"  Test Accuracy: {acc:.4f}")

    print("\nStep 4: 包裝成 SklearnWrapper...")
    wrapped_model = SklearnWrapper(model, scaler)

    print("\nStep 5: 產出 SHAP 視覺化圖表（fallback 路徑）...")
    print("  注意：神經網路使用近似解，計算時間比樹模型長")

    # visualizer 會自動優化樣本數，不需要手動限制

    viz = AutoMLVisualizer(
        model=wrapped_model,
        X_test=X_test,
        output_dir="nn_results"
    )

    viz.generate_all_plots(
        sample_index=0,
        target_feature='MonthlyIncome',
        prefix="mlp_hr"
    )

    print("\n完成！圖表輸出至 nn_results/ 目錄")
    print("  - mlp_hr_global.png     : 全局特徵重要性")
    print("  - mlp_hr_waterfall.png  : 局部瀑布圖")
    print("  - mlp_hr_dependence.png : 依賴散佈圖")
    print("\n如果三張圖都正常輸出，代表 fallback 路徑運作正常。")


if __name__ == "__main__":
    run_nn_test()