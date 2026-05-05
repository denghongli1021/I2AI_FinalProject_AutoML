"""
AutoML 快速建模入口
用法:
  python run_automl.py --csv 你的資料集.csv
  python run_automl.py --csv 你的資料集.csv --target 目標欄位名稱
  python run_automl.py --csv 你的資料集.csv --task classification --ts
  python run_automl.py --csv 你的資料集.csv --trials 50 --no-nas
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── 確保從專案根目錄執行 ──────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from automl_platform.pipeline import AutoMLPipeline


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def auto_detect_task(y: pd.Series) -> str:
    """自動判斷任務類型。"""
    if y.dtype == object or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    ratio = n_unique / len(y)
    if n_unique <= 50 and ratio < 0.30:
        return "classification"
    return "regression"


def auto_detect_ts(X: pd.DataFrame) -> bool:
    """若欄位名稱全為數字（時間步驟索引），視為時序資料。"""
    cols = X.columns.tolist()
    try:
        [int(c) for c in cols]
        return True
    except (ValueError, TypeError):
        return False


def find_target_col(df: pd.DataFrame) -> str:
    """依優先順序尋找目標欄位。"""
    for cand in ("target", "label", "class", "y", "c"):
        if cand in df.columns:
            return cand
    return df.columns[-1]   # 預設最後一欄


def print_metrics(task: str, y_true, y_pred):
    if task == "classification":
        le = LabelEncoder()
        yt = le.fit_transform(pd.Series(y_true).astype(str))
        yp = le.transform(pd.Series(y_pred).astype(str))
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1-macro : {f1:.4f}")
    else:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2   = r2_score(y_true, y_pred)
        print(f"  RMSE : {rmse:.4f}")
        print(f"  R2   : {r2:.4f}")


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AutoML 快速建模：對任意 CSV 資料集執行完整 AutoML 流程"
    )
    parser.add_argument("--csv",     required=True,  help="CSV 檔案路徑")
    parser.add_argument("--target",  default=None,   help="目標欄位名稱（預設自動偵測）")
    parser.add_argument("--task",    default=None,   choices=["classification", "regression"],
                        help="任務類型（預設自動偵測）")
    parser.add_argument("--ts",      action="store_true",
                        help="強制視為時序資料（預設自動偵測）")
    parser.add_argument("--trials",  type=int, default=20, help="HPO 試驗次數（預設 20）")
    parser.add_argument("--no-nas",  action="store_true", help="停用 NAS 神經架構搜索")
    parser.add_argument("--test-size", type=float, default=0.2, help="測試集比例（預設 0.2）")
    parser.add_argument("--seed",    type=int, default=42, help="隨機種子（預設 42）")
    parser.add_argument("--output",  default=None,
                        help="輸出預測結果到 CSV（可選）")
    args = parser.parse_args()

    # ── 載入資料 ─────────────────────────────────────────────────────────────
    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"[錯誤] 找不到檔案：{csv_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  AutoML 快速建模")
    print(f"{'='*60}")
    print(f"  資料集 : {os.path.basename(csv_path)}")

    df = pd.read_csv(csv_path)
    print(f"  形狀   : {df.shape[0]} 筆 × {df.shape[1]} 欄")

    # ── 目標欄位 ─────────────────────────────────────────────────────────────
    target_col = args.target if args.target else find_target_col(df)
    if target_col not in df.columns:
        print(f"[錯誤] 找不到目標欄位 '{target_col}'，可用欄位：{df.columns.tolist()}")
        sys.exit(1)

    y = df[target_col]
    X = df.drop(columns=[target_col])
    X = X.dropna(axis=1, how="all")

    # ── 自動偵測 ─────────────────────────────────────────────────────────────
    task = args.task if args.task else auto_detect_task(y)
    is_ts = args.ts if args.ts else auto_detect_ts(X)
    use_nas = not args.no_nas

    print(f"  目標欄 : {target_col}")
    print(f"  任務   : {task}")
    print(f"  時序   : {is_ts}")
    print(f"  HPO 試驗次數 : {args.trials}")
    print(f"  NAS   : {'啟用' if use_nas else '停用'}")

    # ── 切割訓練/測試集 ──────────────────────────────────────────────────────
    stratify = y if task == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

    print(f"\n  訓練集 : {len(X_train)} 筆  |  測試集 : {len(X_test)} 筆")

    # ── 建模 ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    pipeline = AutoMLPipeline(
        task=task,
        is_timeseries=is_ts,
        n_hpo_trials=args.trials,
        n_folds=5,
        top_k_hpo=5,
        use_nas=use_nas,
        mi_k=50,
        poly_max_cols=10,
        use_shap_pruning=(not is_ts),
    )
    pipeline.fit(X_train, y_train)
    elapsed = time.time() - t0

    # ── 評估 ─────────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_test)

    print(f"\n{'='*60}")
    print(f"  測試集結果")
    print(f"{'='*60}")
    print_metrics(task, y_test, y_pred)
    print(f"  耗時   : {elapsed:.1f} 秒")

    # ── 輸出預測結果（可選）─────────────────────────────────────────────────
    if args.output:
        out_df = X_test.copy()
        out_df["y_true"] = y_test.values
        out_df["y_pred"] = y_pred
        out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n  預測結果已儲存至 : {args.output}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
