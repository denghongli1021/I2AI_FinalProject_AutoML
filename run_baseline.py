"""
run_baseline.py  ─  對照組：AutoGluon TabularPredictor

用法：
  # 批次模式（openml_cc18_data/ 前 N 個 + ucr_ts_80_new(時序資料)/ 前 N 個）
  python run_baseline.py --batch
  python run_baseline.py --batch --top-n 5 --time-budget 120 --presets medium_quality

  # 單一 CSV（自動 80/20 切分）
  python run_baseline.py --csv my_data.csv
  python run_baseline.py --csv my_ts_data.csv --ts        # 時序：回歸改用 chronological split

  # 預切分模式（手動指定 TRAIN / TEST，不再自行切分）
  python run_baseline.py --train CLS_Adiac_TRAIN.csv --test CLS_Adiac_TEST.csv
  python run_baseline.py --train REG_Foo_TRAIN.csv   --test REG_Foo_TEST.csv
"""
import argparse
import os
import shutil
import sys
import time
import traceback
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


def _get_new_ts_datasets(new_ts_dir: str, n_per_type: int = 3) -> list:
    """Return first n_per_type CLS and REG base names from ucr_ts_80_new directory."""
    train_files = sorted(f for f in os.listdir(new_ts_dir) if f.endswith("_TRAIN.csv"))
    cls_bases = [f[:-10] for f in train_files if f.startswith("CLS_")][:n_per_type]
    reg_bases = [f[:-10] for f in train_files if f.startswith("REG_")][:n_per_type]
    return cls_bases + reg_bases


_TIME_RESULT_COLS = [
    "source", "dataset", "type", "task", "n_train", "n_test",
    "accuracy", "f1_macro", "rmse", "r2", "score", "elapsed_s",
]


def _append_time_result(out_path: str, row: dict):
    """Append one result row to time_results.csv; write header only if file is new."""
    df = pd.DataFrame([{c: row.get(c) for c in _TIME_RESULT_COLS}])
    df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)


def run_new_ts_batch(args):
    """AutoGluon baseline on 3 CLS + 3 REG datasets from ucr_ts_80_new(時序資料)."""
    new_ts_dir = os.path.join(HERE, "ucr_ts_80_new(時序資料)")
    if not os.path.isdir(new_ts_dir):
        print(f"[錯誤] 找不到目錄：{new_ts_dir}")
        sys.exit(1)

    selected = _get_new_ts_datasets(new_ts_dir, n_per_type=3)
    out_path = os.path.join(HERE, "time_results.csv")

    print(f"\n{'='*65}")
    print(f"  AutoGluon 新TS批次  ─  {len(selected)} 個資料集  (source=baseline)")
    print(f"{'='*65}")

    for base_name in selected:
        train_path = os.path.join(new_ts_dir, base_name + "_TRAIN.csv")
        test_path  = os.path.join(new_ts_dir, base_name + "_TEST.csv")
        task = "regression" if base_name.startswith("REG_") else "classification"
        print(f"\n[TS] {base_name}  ({task})")
        t0 = time.time()

        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            target_col = find_target_col(train_df)

            train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
            test_df  = test_df.dropna(subset=[target_col]).reset_index(drop=True)

            y_tr = train_df[target_col]
            y_te = test_df[target_col]

            train_X = train_df.drop(columns=[target_col]).dropna(axis=1, how="all")
            test_X  = test_df.drop(columns=[target_col]).dropna(axis=1, how="all")

            # Keep only columns common to both splits
            common_cols = [c for c in train_X.columns if c in test_X.columns]
            train_X = train_X[common_cols]
            test_X  = test_X[common_cols]

            ag_task = (
                ("multiclass" if y_tr.nunique() > 2 else "binary")
                if task == "classification" else "regression"
            )

            ag_dir = os.path.join(HERE, "autogluon_models", base_name)
            if os.path.exists(ag_dir):
                shutil.rmtree(ag_dir, ignore_errors=True)

            train_ag = train_X.copy()
            train_ag[target_col] = y_tr.values

            predictor = TabularPredictor(
                label=target_col,
                problem_type=ag_task,
                path=ag_dir,
                verbosity=0,
            ).fit(
                train_ag,
                time_limit=args.time_budget,
                presets=args.presets,
                dynamic_stacking=False,
                excluded_model_types=["FASTAI", "NeuralNetTorch"],
                ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
            )

            y_pred = predictor.predict(test_X).values
            metrics = get_metrics(task, y_te.values, y_pred)
            elapsed = round(time.time() - t0, 1)

            row = {
                "source":   "baseline",
                "dataset":  base_name,
                "type":     "TS",
                "task":     task,
                "n_train":  len(y_tr),
                "n_test":   len(y_te),
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "rmse":     metrics.get("rmse"),
                "r2":       metrics.get("r2"),
                "score":    metrics.get("f1_macro", metrics.get("r2")),
                "elapsed_s": elapsed,
            }
            _append_time_result(out_path, row)

            if task == "classification":
                print(f"  Accuracy={metrics['accuracy']:.4f}  "
                      f"F1={metrics['f1_macro']:.4f}  ({elapsed}s)")
            else:
                print(f"  RMSE={metrics['rmse']:.4f}  "
                      f"R2={metrics['r2']:.4f}  ({elapsed}s)")

        except Exception as exc:
            elapsed = round(time.time() - t0, 1)
            print(f"  [ERROR] {exc}")
            traceback.print_exc()
            _append_time_result(out_path, {
                "source": "baseline", "dataset": base_name, "type": "TS",
                "task": task, "n_train": 0, "n_test": 0,
                "accuracy": None, "f1_macro": None, "rmse": None, "r2": None,
                "score": None, "elapsed_s": elapsed,
            })

    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


def _find_datasets(openml_dir: str, ts_dir: str, top_n: int, last: bool = False,
                   reg_dir: str = None, reg_top_n: int = None) -> list:
    """回傳 (dataset_name, is_ts, path_a, path_b, force_task) 列表。
    OpenML: path_b=None, force_task=None → run_batch 做 80/20 split 並自動偵測任務。
    TS (ucr_ts_80_new): path_a=TRAIN CSV, path_b=TEST CSV → 直接使用預切資料。
    Reg (openml_regression_data): path_b=None, force_task="regression"。
    """
    openml_all = sorted(f for f in os.listdir(openml_dir) if f.endswith(".csv"))
    ts_trains = sorted(f for f in os.listdir(ts_dir) if f.endswith("_TRAIN.csv"))
    openml_files = openml_all[-top_n:] if last else openml_all[:top_n]
    # 分開取 CLS 和 REG，確保各取 top_n 個（不因混排而漏掉 REG）
    cls_trains = [f for f in ts_trains if f.startswith("CLS_")]
    reg_trains = [f for f in ts_trains if f.startswith("REG_")]
    ts_selected = (cls_trains[-top_n:] if last else cls_trains[:top_n]) + \
                  (reg_trains[-top_n:] if last else reg_trains[:top_n])
    result = []
    for f in openml_files:
        result.append((os.path.splitext(f)[0], False, os.path.join(openml_dir, f), None, None))
    for f in ts_selected:
        base = f[:-10]  # strip "_TRAIN.csv"
        result.append((base, True,
                       os.path.join(ts_dir, f),
                       os.path.join(ts_dir, base + "_TEST.csv"), None))
    if reg_dir and os.path.isdir(reg_dir):
        n = reg_top_n if reg_top_n else top_n
        reg_all = sorted(f for f in os.listdir(reg_dir) if f.endswith(".csv"))
        reg_files = reg_all[-n:] if last else reg_all[:n]
        for f in reg_files:
            result.append((os.path.splitext(f)[0], False, os.path.join(reg_dir, f), None, "regression"))
    return result


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

    reg_dir  = os.path.join(HERE, args.reg_dir)
    datasets = _find_datasets(openml_dir, ts_dir, args.top_n, last=args.last,
                               reg_dir=reg_dir, reg_top_n=args.reg_top_n)
    print(f"\n{'='*65}")
    print(f"  AutoGluon 批次基準  ─  {len(datasets)} 個資料集  "
          f"（OpenML×{args.top_n} + UCR×{args.top_n} + Reg×{args.reg_top_n}）")
    print(f"{'='*65}")

    results = []

    for dataset_name, is_ts, path_a, path_b, force_task in datasets:
        dtype_label = "TS" if is_ts else "Tab"
        print(f"\n[{dtype_label}] {dataset_name}")

        t0 = time.time()
        try:
            if is_ts:
                # 時序資料：直接讀取預切好的 TRAIN / TEST（ucr_ts_80_new 格式）
                train_raw = pd.read_csv(path_a)
                test_raw  = pd.read_csv(path_b)
                target_col = find_target_col(train_raw)
                train_raw = train_raw.dropna(subset=[target_col]).reset_index(drop=True)
                test_raw  = test_raw.dropna(subset=[target_col]).reset_index(drop=True)
                y_tr = train_raw[target_col]
                y_te = test_raw[target_col]
                train_X = train_raw.drop(columns=[target_col]).dropna(axis=1, how="all")
                test_X  = test_raw.drop(columns=[target_col]).dropna(axis=1, how="all")
                common_cols = [c for c in train_X.columns if c in test_X.columns]
                train_X = train_X[common_cols]
                test_X  = test_X[common_cols]
                task = "regression" if dataset_name.startswith("REG_") else auto_detect_task(y_tr)
                ag_task = (("multiclass" if y_tr.nunique() > 2 else "binary")
                           if task == "classification" else "regression")
                train_df = train_X.copy()
                train_df[target_col] = y_tr.values
                test_df = test_X
                print(f"  [Split] Pre-split  n_train={len(y_tr)}  n_test={len(y_te)}")
            else:
                # 表格資料：讀取單一 CSV 並做 80/20 split
                df = pd.read_csv(path_a)
                target_col = find_target_col(df)
                n_before = len(df)
                df = df.dropna(subset=[target_col]).reset_index(drop=True)
                if len(df) < n_before:
                    print(f"  [info] dropped {n_before - len(df)} rows with NaN target")
                y = df[target_col]
                X = df.drop(columns=[target_col]).dropna(axis=1, how="all")
                task = force_task if force_task else auto_detect_task(y)
                ag_task = (("multiclass" if y.nunique() > 2 else "binary")
                           if task == "classification" else task)
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
                "source":   "baseline",
                "dataset":  dataset_name,
                "type":     dtype_label,
                "task":     task,
                "n_train":  len(y_tr),
                "n_test":   len(y_te),
                "accuracy": metrics.get("accuracy"),
                "f1_macro": metrics.get("f1_macro"),
                "rmse":     metrics.get("rmse"),
                "r2":       metrics.get("r2"),
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

    out_path = args.result_file if args.result_file else os.path.join(HERE, "baseline_batch_results.csv")
    summary.to_csv(out_path, index=False)
    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


# ── 預切分單一模式 ────────────────────────────────────────────────────────────

def run_presplit(args):
    """用戶手動提供 TRAIN / TEST 兩個 CSV，直接使用不再自行切分。"""
    train_path = os.path.abspath(args.train)
    test_path  = os.path.abspath(args.test)
    for p, label in [(train_path, "TRAIN"), (test_path, "TEST")]:
        if not os.path.exists(p):
            print(f"[錯誤] 找不到{label}檔案：{p}")
            sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    target_col = args.target if args.target else find_target_col(train_df)
    if target_col not in train_df.columns:
        print(f"[錯誤] 找不到欄位 '{target_col}'，可用：{train_df.columns.tolist()}")
        sys.exit(1)

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=[target_col]).reset_index(drop=True)

    y_tr = train_df[target_col]
    y_te = test_df[target_col]
    train_X = train_df.drop(columns=[target_col]).dropna(axis=1, how="all")
    test_X  = test_df.drop(columns=[target_col]).dropna(axis=1, how="all")
    common_cols = [c for c in train_X.columns if c in test_X.columns]
    train_X = train_X[common_cols]
    test_X  = test_X[common_cols]

    # 任務判斷：--task 優先；其次看 TRAIN 檔名前綴；再自動偵測
    base = os.path.splitext(os.path.basename(train_path))[0]
    dataset_name = base[:-6] if base.endswith("_TRAIN") else base
    if args.task:
        task = args.task
    elif dataset_name.startswith("REG_"):
        task = "regression"
    else:
        task = auto_detect_task(y_tr)

    ag_task = (("multiclass" if y_tr.nunique() > 2 else "binary")
               if task == "classification" else "regression")

    print(f"\n{'='*60}")
    print(f"  AutoGluon 對照組基準（預切分模式）")
    print(f"{'='*60}")
    print(f"  訓練集  : {os.path.basename(train_path)}  ({len(train_df)} × {train_df.shape[1]})")
    print(f"  測試集  : {os.path.basename(test_path)}  ({len(test_df)} × {test_df.shape[1]})")
    print(f"  目標欄  : {target_col}  |  任務 : {task}  |  共同特徵 : {len(common_cols)}")
    print(f"  Presets : {args.presets}  |  時間上限 : {args.time_budget}s\n")

    train_ag = train_X.copy()
    train_ag[target_col] = y_tr.values

    ag_dir = os.path.join(HERE, "autogluon_models", dataset_name)
    if os.path.exists(ag_dir):
        shutil.rmtree(ag_dir, ignore_errors=True)

    print("[AutoGluon] 開始訓練...")
    t0 = time.time()
    predictor = TabularPredictor(
        label=target_col,
        problem_type=ag_task,
        path=ag_dir,
        verbosity=2,
    ).fit(
        train_ag,
        time_limit=args.time_budget,
        presets=args.presets,
        dynamic_stacking=False,
        excluded_model_types=["FASTAI", "NeuralNetTorch"],
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    ag_elapsed = time.time() - t0

    y_pred = predictor.predict(test_X).values

    print(f"\n{'='*60}")
    print(f"  AutoGluon 測試集結果  （耗時 {ag_elapsed:.1f}s）")
    print(f"{'='*60}")
    print_metrics(task, y_te.values, y_pred, "AutoGluon")

    print(f"\n  --- AutoGluon 模型排行榜 ---")
    leaderboard = predictor.leaderboard(
        test_X.assign(**{target_col: y_te.values}), silent=True)
    print(leaderboard[["model", "score_test", "score_val", "fit_time"]].to_string(index=False))
    print(f"\n{'='*60}\n")

    if args.result_file:
        metrics = get_metrics(task, y_te.values, y_pred)
        is_ts = base.endswith("_TRAIN")
        row = {
            "source":   "baseline",
            "dataset":  dataset_name,
            "type":     "TS" if is_ts else "Tab",
            "task":     task,
            "n_train":  len(y_tr),
            "n_test":   len(y_te),
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "rmse":     metrics.get("rmse"),
            "r2":       metrics.get("r2"),
            "score":    metrics.get("f1_macro", metrics.get("r2")),
            "elapsed_s": round(ag_elapsed, 1),
        }
        _append_time_result(args.result_file, row)
        print(f"  結果已附加 → {args.result_file}")


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="對照組：AutoGluon TabularPredictor"
    )
    parser.add_argument("--csv",         default=None,   help="CSV 檔案路徑（單一 CSV 模式）")
    parser.add_argument("--train",       default=None,   help="訓練集 CSV 路徑（搭配 --test 使用預切分模式）")
    parser.add_argument("--test",        default=None,   help="測試集 CSV 路徑（搭配 --train 使用預切分模式）")
    parser.add_argument("--target",      default=None,   help="目標欄位名稱（預設自動偵測）")
    parser.add_argument("--task",        default=None,   choices=["classification", "regression"])
    parser.add_argument("--ts",          action="store_true",
                        help="標記為時序資料（單一 CSV 模式下，回歸改用 chronological split）")
    parser.add_argument("--time-budget", type=int, default=1000, help="訓練時間上限（秒，預設 120）")
    parser.add_argument("--presets",     default="medium_quality",
                        choices=["medium_quality", "good_quality", "best_quality"],
                        help="AutoGluon presets（預設 medium_quality）")
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output-dir",  default="autogluon_models", help="AutoGluon 模型儲存目錄（單一模式）")
    parser.add_argument("--result-file", default=None,
                        help="結果輸出 CSV（附加模式；格式同 pipeline_batch_results.csv）")

    # ── 批次模式 ──────────────────────────────────────────────────────────────
    parser.add_argument("--batch",      action="store_true",
                        help="批次模式：自動跑前 top-n 個 OpenML + UCR 資料集")
    parser.add_argument("--openml-dir", default="openml_cc18_data",
                        help="OpenML CSV 目錄（預設 openml_cc18_data）")
    parser.add_argument("--ts-dir",     default="ucr_ts_80_new(時序資料)",
                        help="UCR 時序 CSV 目錄（預切分格式：含 _TRAIN.csv / _TEST.csv）")
    parser.add_argument("--top-n",      type=int, default=1,
                        help="每個目錄取前幾個資料集（預設 5）")
    parser.add_argument("--last",       action="store_true",
                        help="取每目錄最後 top-n 個資料集")
    parser.add_argument("--new-ts-batch", action="store_true",
                        help="新TS批次：讀 ucr_ts_80_new，各取3個CLS+REG，寫 time_results.csv")
    parser.add_argument("--reg-dir",     default="openml_regression_data",
                        help="非時序回歸 CSV 目錄（批次模式用，預設 openml_regression_data）")
    parser.add_argument("--reg-top-n",   type=int, default=5,
                        help="回歸目錄取前 N 個資料集（批次模式用）")

    args = parser.parse_args()

    # ── 新TS批次模式 ───────────────────────────────────────────────────────────
    if args.new_ts_batch:
        run_new_ts_batch(args)
        return

    # ── 批次模式 ──────────────────────────────────────────────────────────────
    if args.batch:
        run_batch(args)
        return

    # ── 預切分模式（--train + --test）─────────────────────────────────────────
    if args.train and args.test:
        run_presplit(args)
        return

    # ── 單一 CSV 模式（--csv）─────────────────────────────────────────────────
    if not args.csv:
        parser.error("請指定 --csv、--train/--test 或 --batch")

    csv_path = os.path.abspath(args.csv)
    if not os.path.exists(csv_path):
        print(f"[錯誤] 找不到檔案：{csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    target_col = args.target if args.target else find_target_col(df)
    if target_col not in df.columns:
        print(f"[錯誤] 找不到欄位 '{target_col}'，可用：{df.columns.tolist()}")
        sys.exit(1)

    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [info] dropped {n_before - len(df)} rows with NaN target")

    y = df[target_col]
    X = df.drop(columns=[target_col]).dropna(axis=1, how="all")
    task = args.task if args.task else auto_detect_task(y)

    print(f"\n{'='*60}")
    print(f"  AutoGluon 對照組基準")
    print(f"{'='*60}")
    print(f"  資料集  : {os.path.basename(csv_path)}  ({df.shape[0]} × {df.shape[1]})")
    print(f"  目標欄  : {target_col}  |  任務 : {task}  |  TS : {args.ts}")
    print(f"  Presets : {args.presets}  |  時間上限 : {args.time_budget}s")

    # 時序回歸：依序切分（避免未來資訊洩漏）；其餘：隨機切分
    if args.ts and task == "regression":
        split_idx = int(len(X) * (1 - args.test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"  [Split] Chronological  n_train={len(y_train)}  n_test={len(y_test)}")
    else:
        stratify = y if task == "classification" else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=args.test_size, random_state=args.seed
            )
        print(f"  訓練集  : {len(X_train)}  |  測試集 : {len(X_test)}")

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

    if args.result_file:
        metrics = get_metrics(task, y_test.values, y_pred_ag)
        ds_name = os.path.splitext(os.path.basename(csv_path))[0]
        row = {
            "source":   "baseline",
            "dataset":  ds_name,
            "type":     "Tab",
            "task":     task,
            "n_train":  len(X_train),
            "n_test":   len(X_test),
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "rmse":     metrics.get("rmse"),
            "r2":       metrics.get("r2"),
            "score":    metrics.get("f1_macro", metrics.get("r2")),
            "elapsed_s": round(ag_elapsed, 1),
        }
        _append_time_result(args.result_file, row)
        print(f"  結果已附加 → {args.result_file}")


if __name__ == "__main__":
    main()
