"""
run_baseline.py  ─  對照組：AutoGluon TabularPredictor

用法：
  # 批次模式（openml_cc18_data/ 前 N 個 + ucr_ts_80(時序資料)/ 前 N 個）
  python run_baseline.py --batch
  python run_baseline.py --batch --top-n 5 --time-budget 120 --presets medium_quality

  # 單一資料集
  python run_baseline.py --csv openml_cc18_data/22_mfeat-zernike.csv
  python run_baseline.py --csv openml_cc18_data/22_mfeat-zernike.csv --time-budget 120
"""
import argparse
import os
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from autogluon.tabular import TabularPredictor

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# ── 工具函式 ─────────────────────────────────────────────────────────────────

def auto_detect_task(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    return "classification" if (n_unique <= 50 and n_unique / len(y) < 0.30) else "regression"


def find_target_col(df: pd.DataFrame) -> str:
    for cand in ("target", "label", "class", "y", "c"):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def print_metrics(task: str, y_true, y_pred, label: str = ""):
    prefix = f"  [{label}]" if label else " "
    if task == "classification":
        le = LabelEncoder()
        yt = le.fit_transform(pd.Series(y_true).astype(str))
        try:
            yp = le.transform(pd.Series(y_pred).astype(str))
        except ValueError:
            yp = np.zeros_like(yt)
        print(f"{prefix} Accuracy : {accuracy_score(yt, yp):.4f}")
        print(f"{prefix} F1-macro : {f1_score(yt, yp, average='macro', zero_division=0):.4f}")
    else:
        print(f"{prefix} RMSE     : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
        print(f"{prefix} R2       : {r2_score(y_true, y_pred):.4f}")


def get_metrics(task: str, y_true, y_pred) -> dict:
    if task == "classification":
        le = LabelEncoder()
        yt = le.fit_transform(pd.Series(y_true).astype(str))
        try:
            yp = le.transform(pd.Series(y_pred).astype(str))
        except ValueError:
            yp = np.zeros_like(yt)
        return {
            "accuracy": round(accuracy_score(yt, yp), 4),
            "f1_macro": round(f1_score(yt, yp, average="macro", zero_division=0), 4),
        }
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": round(rmse, 4), "r2": round(r2, 4)}


def _find_datasets(openml_dir: str, ts_dir: str, top_n: int):
    """回傳 (csv_path, is_ts) 的列表：前 top_n 個 OpenML + 前 top_n 個 UCR。"""
    openml_files = sorted(f for f in os.listdir(openml_dir) if f.endswith(".csv"))[:top_n]
    ts_files = sorted(f for f in os.listdir(ts_dir) if f.endswith(".csv"))[:top_n]
    return (
        [(os.path.join(openml_dir, f), False) for f in openml_files] +
        [(os.path.join(ts_dir, f), True) for f in ts_files]
    )


# ── 批次模式 ──────────────────────────────────────────────────────────────────

def run_batch(args):
    openml_dir = os.path.join(HERE, args.openml_dir)
    ts_dir = os.path.join(HERE, args.ts_dir)

    if not os.path.isdir(openml_dir):
        print(f"[錯誤] 找不到 OpenML 目錄：{openml_dir}")
        sys.exit(1)
    if not os.path.isdir(ts_dir):
        print(f"[錯誤] 找不到 UCR 目錄：{ts_dir}")
        sys.exit(1)

    datasets = _find_datasets(openml_dir, ts_dir, args.top_n)
    print(f"\n{'='*65}")
    print(f"  AutoGluon 批次基準  ─  {len(datasets)} 個資料集  "
          f"（OpenML×{args.top_n} + UCR×{args.top_n}）")
    print(f"{'='*65}")

    results = []

    for csv_path, is_ts in datasets:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        dtype_label = "TS" if is_ts else "Tab"
        print(f"\n[{dtype_label}] {dataset_name}")

        t0 = time.time()
        try:
            df = pd.read_csv(csv_path)
            target_col = find_target_col(df)
            y = df[target_col]
            X = df.drop(columns=[target_col]).dropna(axis=1, how="all")
            task = auto_detect_task(y)

            ag_task = (
                ("multiclass" if y.nunique() > 2 else "binary")
                if task == "classification" else task
            )

            stratify = y if task == "classification" else None
            try:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
                )
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=args.test_size, random_state=args.seed
                )

            train_df = X_tr.copy()
            train_df[target_col] = y_tr.values
            test_df = X_te.copy()
            test_with_label = test_df.copy()
            test_with_label[target_col] = y_te.values

            ag_dir = os.path.join(HERE, "autogluon_models", dataset_name)
            if os.path.exists(ag_dir):
                shutil.rmtree(ag_dir, ignore_errors=True)

            predictor = TabularPredictor(
                label=target_col,
                problem_type=ag_task,
                path=ag_dir,
                verbosity=0,
            ).fit(
                train_df,
                time_limit=args.time_budget,
                presets=args.presets,
                dynamic_stacking=False,
                excluded_model_types=["FASTAI", "NeuralNetTorch"],
                ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
            )

            y_pred = predictor.predict(test_df).values
            metrics = get_metrics(task, y_te.values, y_pred)
            elapsed = round(time.time() - t0, 1)

            row = {
                "dataset":  dataset_name,
                "type":     dtype_label,
                "task":     task,
                "n_train":  len(X_tr),
                "n_test":   len(X_te),
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "score":    metrics.get("f1_macro", metrics.get("r2")),
                "elapsed_s": elapsed,
            }
            results.append(row)

            if task == "classification":
                print(f"  Accuracy={metrics['accuracy']:.4f}  "
                      f"F1={metrics['f1_macro']:.4f}  ({elapsed}s)")
            else:
                print(f"  RMSE={metrics.get('rmse', '?'):.4f}  "
                      f"R2={metrics.get('r2', '?'):.4f}  ({elapsed}s)")

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            print(f"  [ERROR] {exc}")
            results.append({
                "dataset":  dataset_name,
                "type":     dtype_label,
                "task":     "?",
                "n_train":  0,
                "n_test":   0,
                "accuracy": None,
                "f1_macro": None,
                "score":    None,
                "error":    str(exc)[:80],
                "elapsed_s": elapsed,
            })

    # ── 總結 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  BATCH SUMMARY — AutoGluon Baseline")
    print(f"{'='*65}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))

    out_path = os.path.join(HERE, "baseline_batch_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="對照組：AutoGluon TabularPredictor"
    )
    parser.add_argument("--csv",         default=None,   help="CSV 檔案路徑（單一模式）")
    parser.add_argument("--target",      default=None,   help="目標欄位名稱（預設自動偵測）")
    parser.add_argument("--task",        default=None,   choices=["classification", "regression"])
    parser.add_argument("--time-budget", type=int, default=1000, help="訓練時間上限（秒，預設 120）")
    parser.add_argument("--presets",     default="medium_quality",
                        choices=["medium_quality", "good_quality", "best_quality"],
                        help="AutoGluon presets（預設 medium_quality）")
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output-dir",  default="autogluon_models", help="AutoGluon 模型儲存目錄（單一模式）")

    # ── 批次模式 ──────────────────────────────────────────────────────────────
    parser.add_argument("--batch",      action="store_true",
                        help="批次模式：自動跑前 top-n 個 OpenML + UCR 資料集")
    parser.add_argument("--openml-dir", default="openml_cc18_data",
                        help="OpenML CSV 目錄（預設 openml_cc18_data）")
    parser.add_argument("--ts-dir",     default="ucr_ts_80(時序資料)",
                        help="UCR 時序 CSV 目錄")
    parser.add_argument("--top-n",      type=int, default=1,
                        help="每個目錄取前幾個資料集（預設 5）")
    
    args = parser.parse_args()

    # ── 批次模式 ──────────────────────────────────────────────────────────────
    if args.batch:
        run_batch(args)
        return

    # ── 單一 CSV 模式 ─────────────────────────────────────────────────────────
    if not args.csv:
        parser.error("單一模式需指定 --csv，或使用 --batch 執行批次模式")

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"[錯誤] 找不到檔案：{csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    target_col = args.target if args.target else find_target_col(df)
    if target_col not in df.columns:
        print(f"[錯誤] 找不到欄位 '{target_col}'，可用：{df.columns.tolist()}")
        sys.exit(1)

    y = df[target_col]
    X = df.drop(columns=[target_col]).dropna(axis=1, how="all")
    task = args.task if args.task else auto_detect_task(y)

    print(f"\n{'='*60}")
    print(f"  AutoGluon 對照組基準")
    print(f"{'='*60}")
    print(f"  資料集  : {os.path.basename(csv_path)}  ({df.shape[0]} × {df.shape[1]})")
    print(f"  目標欄  : {target_col}  |  任務 : {task}")
    print(f"  Presets : {args.presets}  |  時間上限 : {args.time_budget}s")

    stratify = y if task == "classification" else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

    print(f"  訓練集  : {len(X_train)}  |  測試集 : {len(X_test)}\n")

    train_df = X_train.copy()
    train_df[target_col] = y_train.values
    test_df = X_test.copy()

    ag_task = ("multiclass" if y.nunique() > 2 else "binary") if task == "classification" else task

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir, ignore_errors=True)

    print(f"[AutoGluon] 開始訓練...")
    t0 = time.time()
    predictor = TabularPredictor(
        label=target_col,
        problem_type=ag_task,
        path=args.output_dir,
        verbosity=2,
    ).fit(
        train_df,
        time_limit=args.time_budget,
        presets=args.presets,
        dynamic_stacking=False,
        excluded_model_types=["FASTAI", "NeuralNetTorch"],
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    ag_elapsed = time.time() - t0

    y_pred_ag = predictor.predict(test_df).values

    print(f"\n{'='*60}")
    print(f"  AutoGluon 測試集結果  （耗時 {ag_elapsed:.1f}s）")
    print(f"{'='*60}")
    print_metrics(task, y_test.values, y_pred_ag, "AutoGluon")

    print(f"\n  --- AutoGluon 模型排行榜 ---")
    leaderboard = predictor.leaderboard(test_df.assign(**{target_col: y_test.values}), silent=True)
    print(leaderboard[["model", "score_test", "score_val", "fit_time"]].to_string(index=False))

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
