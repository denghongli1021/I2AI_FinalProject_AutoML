"""
時序資料壓力測試：ETTh1 資料集
---------------------------------------
資料來源：https://github.com/zhouhaoyi/ETDataset
下載方式：
  wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv

任務：預測電力變壓器油溫（OT，Oil Temperature）
輸入特徵：lag features + rolling statistics（由本腳本自動產生）
輸出：三張 SHAP 視覺化圖表
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from visualizer import AutoMLVisualizer


def make_lag_features(df, target_col, lags, rolling_windows):
    """
    把時序資料轉成監督式學習格式。
    這一步模擬特徵工程組的輸出——
    他們做完之後，X 就是一個普通的 DataFrame，
    和表格資料格式完全相同，可直接餵給視覺化模組。
    """
    feat_df = df.copy()

    # Lag features：過去 N 小時的值
    for lag in lags:
        feat_df[f"{target_col}_lag{lag}"] = feat_df[target_col].shift(lag)

    # Rolling statistics：滑動窗口統計
    for window in rolling_windows:
        feat_df[f"{target_col}_roll_mean{window}"] = (
            feat_df[target_col].shift(1).rolling(window).mean()
        )
        feat_df[f"{target_col}_roll_std{window}"] = (
            feat_df[target_col].shift(1).rolling(window).std()
        )

    # 日曆特徵
    feat_df["hour"]    = feat_df.index.hour
    feat_df["weekday"] = feat_df.index.weekday
    feat_df["month"]   = feat_df.index.month

    # 刪除 NaN（lag 產生的空值）
    feat_df = feat_df.dropna()
    return feat_df


def run_ett_timeseries():
    print("Step 1: 載入 ETTh1 資料集...")
    # 請先執行：
    # wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
    df = pd.read_csv("dataset/ETTh1.csv", parse_dates=["date"], index_col="date")

    print(f"  原始資料：{df.shape[0]} 筆 × {df.shape[1]} 個欄位")
    print(f"  欄位：{list(df.columns)}")
    print(f"  時間範圍：{df.index[0]} ~ {df.index[-1]}")

    # 預測目標：OT（油溫）
    TARGET = "OT"

    print("\nStep 2: 產生時序特徵（lag features + rolling statistics）...")
    feat_df = make_lag_features(
        df,
        target_col=TARGET,
        lags=[1, 2, 3, 6, 12, 24],          # 過去 1/2/3/6/12/24 小時
        rolling_windows=[6, 24]              # 6小時、24小時滑動窗口
    )

    # 特徵欄位 = 除了 OT 以外的所有欄（原始特徵 + 衍生特徵）
    feature_cols = [c for c in feat_df.columns if c != TARGET]
    X = feat_df[feature_cols]
    y = feat_df[TARGET]

    print(f"  特徵工程後：{X.shape[0]} 筆 × {X.shape[1]} 個特徵")
    print(f"  特徵列表：{list(X.columns)}")

    # 時序資料必須按時間順序切分，不能隨機 shuffle
    print("\nStep 3: 時序切分（保留時間順序，不 shuffle）...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    print(f"  Train: {len(X_train)} 筆，Test: {len(X_test)} 筆")

    print("\nStep 4: 訓練 XGBoost 回歸模型...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)

    # 簡單評估
    from sklearn.metrics import mean_squared_error
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"  Test RMSE: {rmse:.4f}")

    print("\nStep 5: 產出 SHAP 視覺化圖表...")
    viz = AutoMLVisualizer(
        model=model,
        X_test=X_test,
        output_dir="timeseries_results"
    )

    # OT_lag1 是最直觀的目標特徵（上一小時油溫 vs SHAP 值）
    viz.generate_all_plots(
        sample_index=10,
        target_feature="OT_lag1",
        prefix="ett"
    )

    print("\n完成！圖表輸出至 timeseries_results/ 目錄")
    print("  - ett_global.png     : 全局特徵重要性")
    print("  - ett_waterfall.png  : 局部瀑布圖（第 10 筆樣本）")
    print("  - ett_dependence.png : 依賴散佈圖（OT_lag1 × 交互特徵）")


if __name__ == "__main__":
    run_ett_timeseries()