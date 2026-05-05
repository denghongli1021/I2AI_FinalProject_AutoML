"""
驗證腳本（validation script）：
    - 從 openml_cc18_data/ 取 5 個非時序資料集
    - 從 ucr_ts_80(時序資料)/ 取 5 個時序資料集
逐一跑完整的 AutoMLPipeline 並輸出評估結果到 results/ 資料夾。
"""
import os
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error, r2_score
)
from sklearn.preprocessing import LabelEncoder

from .pipeline import AutoMLPipeline

warnings.filterwarnings("ignore")

# 各資料夾的絕對路徑（以本檔案位置往上推一層為專案根目錄）
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABULAR_DIR = os.path.join(PROJECT_DIR, "openml_cc18_data")
TS_DIR      = os.path.join(PROJECT_DIR, "ucr_ts_80(時序資料)")
REPORT_DIR  = os.path.join(PROJECT_DIR, "automl_platform", "results")


def _detect_task(y: pd.Series, filename: str = "") -> str:
    """依檔名前綴或標籤分布自動判斷分類 / 回歸任務。"""
    basename = os.path.basename(filename)
    # UCR 資料集的命名規則：CLS_ 開頭固定為分類、REG_ 開頭固定為回歸
    if basename.startswith("CLS_"):
        return "classification"
    if basename.startswith("REG_"):
        return "regression"
    # 字串/布林型 → 一定是分類
    if y.dtype == bool or y.dtype == object:
        return "classification"
    # 整數型：唯一值 ≤ 50 且唯一比例 < 30% 視為分類，否則回歸
    n_unique = y.nunique()
    ratio = n_unique / len(y)
    if n_unique <= 50 and ratio < 0.3:
        return "classification"
    return "regression"


def _get_target_col(df: pd.DataFrame, filename: str) -> str:
    """從 DataFrame 中找出目標欄位名稱（依常見命名嘗試，失敗就取最後一欄）。"""
    if "target" in df.columns:
        return "target"
    # openml 資料集常見的目標欄為 'c'
    if "c" in df.columns:
        return "c"
    return df.columns[-1]


def _load_dataset(filepath: str):
    """讀取 CSV，分離特徵與標籤，並去除全為 NaN 的欄位。"""
    df = pd.read_csv(filepath)
    target_col = _get_target_col(df, filepath)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    # 全為 NaN 的欄位無資訊量，直接丟棄
    X = X.dropna(axis=1, how="all")
    return X, y


def _evaluate_clf(y_true, y_pred, label):
    """分類評估：Accuracy + F1-macro。"""
    le = LabelEncoder()
    # 統一以字串編碼，避免 train/test 標籤型別不一致
    y_true_enc = le.fit_transform(y_true.astype(str))
    y_pred_enc = le.transform(pd.Series(y_pred).astype(str))
    acc = accuracy_score(y_true_enc, y_pred_enc)
    f1  = f1_score(y_true_enc, y_pred_enc, average="macro", zero_division=0)
    print(f"    Accuracy: {acc:.4f}  |  F1-macro: {f1:.4f}")
    return {"accuracy": acc, "f1_macro": f1}


def _evaluate_reg(y_true, y_pred, label):
    """回歸評估：RMSE + R²。"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"    RMSE: {rmse:.4f}  |  R2: {r2:.4f}")
    return {"rmse": rmse, "r2": r2}


def run_validation(
    n_tabular: int = 5,             # 表格資料集數量
    n_ts: int = 5,                  # 時序資料集數量
    n_hpo_trials: int = 20,         # 每個模型的 HPO 試驗次數
    use_nas: bool = True,           # 是否啟用 NAS（會明顯拉長時間）
    test_size: float = 0.2,
    random_state: int = 42,
):
    """主驗證流程：跑兩類資料集，將結果存成 CSV 報表。"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    all_results = []

    # ------------------------------------------------------------------ #
    # 1. 非時序資料集
    # ------------------------------------------------------------------ #
    tabular_files = sorted(os.listdir(TABULAR_DIR))[:n_tabular]
    print(f"\n{'='*60}")
    print(f"  Non-time-series datasets  ({len(tabular_files)} datasets)")
    print(f"{'='*60}")

    for fname in tabular_files:
        fpath = os.path.join(TABULAR_DIR, fname)
        dataset_name = os.path.splitext(fname)[0]
        print(f"\n[Dataset] {dataset_name}")

        try:
            X, y = _load_dataset(fpath)
            task = _detect_task(y, fname)
            print(f"  Shape: {X.shape}  |  Task: {task}  |  Target unique: {y.nunique()}")

            # 分類採分層切分以維持類別比例
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if task == "classification" else None,
            )

            t0 = time.time()
            # 表格資料：開啟 SHAP 剪枝，is_timeseries=False
            pipeline = AutoMLPipeline(
                task=task,
                is_timeseries=False,
                n_hpo_trials=n_hpo_trials,
                n_folds=5,
                top_k_hpo=5,
                use_nas=use_nas,
                mi_k=50,
                poly_max_cols=10,
                use_shap_pruning=True,
            )
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - t0

            y_pred = pipeline.predict(X_test)

            # 收集本次跑分的所有 metadata 與評估結果
            row = {
                "dataset": dataset_name,
                "type": "tabular",
                "task": task,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features_raw": X.shape[1],
                "elapsed_sec": round(elapsed, 1),
            }
            if task == "classification":
                metrics = _evaluate_clf(y_test, y_pred, dataset_name)
            else:
                metrics = _evaluate_reg(y_test, y_pred, dataset_name)
            row.update(metrics)
            all_results.append(row)
            print(f"  Time: {elapsed:.1f}s")

        except Exception as e:
            # 單一資料集失敗不應中斷整個驗證流程，保留錯誤訊息繼續跑下一個
            import traceback
            print(f"  [ERROR] {dataset_name}: {e}")
            traceback.print_exc()
            all_results.append({
                "dataset": dataset_name, "type": "tabular", "task": "unknown",
                "error": str(e)
            })

    # ------------------------------------------------------------------ #
    # 2. 時序資料集
    # ------------------------------------------------------------------ #
    ts_files = sorted(os.listdir(TS_DIR))[:n_ts]
    print(f"\n{'='*60}")
    print(f"  Time-series datasets  ({len(ts_files)} datasets)")
    print(f"{'='*60}")

    for fname in ts_files:
        fpath = os.path.join(TS_DIR, fname)
        dataset_name = os.path.splitext(fname)[0]
        print(f"\n[Dataset] {dataset_name}")

        try:
            X, y = _load_dataset(fpath)
            task = _detect_task(y, fname)
            print(f"  Shape: {X.shape}  |  Task: {task}  |  Target unique: {y.nunique()}")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y if task == "classification" else None,
            )

            t0 = time.time()
            # 時序資料：is_timeseries=True、SHAP 剪枝關閉（時序統計特徵本就少）
            pipeline = AutoMLPipeline(
                task=task,
                is_timeseries=True,
                n_hpo_trials=n_hpo_trials,
                n_folds=5,
                top_k_hpo=5,
                use_nas=use_nas,
                mi_k=50,
                poly_max_cols=10,
                use_shap_pruning=False,
            )
            pipeline.fit(X_train, y_train)
            elapsed = time.time() - t0

            y_pred = pipeline.predict(X_test)

            row = {
                "dataset": dataset_name,
                "type": "timeseries",
                "task": task,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_features_raw": X.shape[1],
                "elapsed_sec": round(elapsed, 1),
            }
            if task == "classification":
                metrics = _evaluate_clf(y_test, y_pred, dataset_name)
            else:
                metrics = _evaluate_reg(y_test, y_pred, dataset_name)
            row.update(metrics)
            all_results.append(row)
            print(f"  Time: {elapsed:.1f}s")

        except Exception as e:
            import traceback
            print(f"  [ERROR] {dataset_name}: {e}")
            traceback.print_exc()
            all_results.append({
                "dataset": dataset_name, "type": "timeseries", "task": "unknown",
                "error": str(e)
            })

    # ------------------------------------------------------------------ #
    # 3. 將結果存成 CSV 報表
    # ------------------------------------------------------------------ #
    df_results = pd.DataFrame(all_results)
    csv_path = os.path.join(REPORT_DIR, "validation_results.csv")
    # 使用 utf-8-sig 避免 Excel 開啟中文出現亂碼
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Report] Saved to {csv_path}")
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(df_results.to_string(index=False))
    return df_results


if __name__ == "__main__":
    # 直接執行：跑 5 表格 + 5 時序，每模型 20 trials，啟用 NAS
    run_validation(n_tabular=5, n_ts=5, n_hpo_trials=20, use_nas=True)
