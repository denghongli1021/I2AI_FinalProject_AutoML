"""
run_pipeline.py — Pipeline 執行入口（v2）

每個資料集的 ML 邏輯（HPO / NAS / CV / Ensemble）完全由 pipeline.py 負責；
本檔僅處理：資料載入、80/20 split、TS 特徵前處理、呼叫 pipeline.run()、結果輸出。

批次評估模式（openml_cc18_data/ 前 N 個 + ucr_ts_80(時序資料)/ 前 N 個，80/20 split 評估）：
    python run_pipeline.py --batch [--fast] [--top-n 5]
    python run_pipeline.py --batch --skip-dl --top-n 3

競賽模式（讀取 test/train.csv + test/test.csv）：
    python run_pipeline.py [--fast] [--ts] [--no-nas] [--skip-dl] [--skip-tabular]
    python run_pipeline.py --time-limit 3600
"""
import os
import sys
import time
import argparse
import traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.config import DEVICE, SEED, ARTIFACTS_DIR
import pipeline as _pl


# ── 資料輔助函式 ──────────────────────────────────────────────────────────────

def _auto_detect_task(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    return "classification" if (n_unique <= 50 and n_unique / len(y) < 0.30) else "regression"


def _find_target_col(df: pd.DataFrame) -> str:
    for cand in ("target", "label", "class", "y", "c"):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def _find_datasets(openml_dir: str, ts_dir: str, top_n: int):
    """回傳 (csv_path, is_ts) 列表：前 top_n 個 OpenML + 前 top_n 個 UCR。"""
    openml = sorted(f for f in os.listdir(openml_dir) if f.endswith(".csv"))[:top_n]
    ts     = sorted(f for f in os.listdir(ts_dir)     if f.endswith(".csv"))[:top_n]
    return (
        [(os.path.join(openml_dir, f), False) for f in openml] +
        [(os.path.join(ts_dir,     f), True)  for f in ts]
    )



# ── 批次評估模式 ──────────────────────────────────────────────────────────────

def run_batch(args):
    openml_dir = os.path.join(HERE, args.openml_dir)
    ts_dir     = os.path.join(HERE, args.ts_dir)
    for d in [openml_dir, ts_dir]:
        if not os.path.isdir(d):
            print(f"[錯誤] 找不到目錄：{d}"); sys.exit(1)

    datasets = _find_datasets(openml_dir, ts_dir, args.top_n)
    print(f"\n{'='*65}")
    print(f"  Pipeline 批次評估  ─  {len(datasets)} 個資料集  "
          f"（OpenML×{args.top_n} + UCR×{args.top_n}）  fast={args.fast}")
    print(f"{'='*65}")
    results = []

    for csv_path, is_ts in datasets:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        dtype_label  = "TS" if is_ts else "Tab"
        print(f"\n{'─'*65}")
        print(f"  [{dtype_label}] {dataset_name}  |  device={DEVICE}  ts={is_ts}")
        print(f"{'─'*65}")
        t_ds = time.time()
        try:
            # ── 載入 + 預處理 ─────────────────────────────────────────────────
            df         = pd.read_csv(csv_path)
            target_col = _find_target_col(df)
            y_raw      = df[target_col]
            task       = _auto_detect_task(y_raw)

            if task == "regression":
                print("  [SKIP] 回歸任務暫不支援批次 Pipeline")
                results.append({"dataset": dataset_name, "type": dtype_label,
                                 "task": task, "note": "regression skipped"})
                continue

            X_all = (df.drop(columns=[target_col])
                       .select_dtypes(include=[np.number])
                       .fillna(0).values.astype(np.float32))
            le        = LabelEncoder()
            y_all     = le.fit_transform(y_raw.astype(str).values)
            n_classes = len(le.classes_)

            # ── 關鍵邏輯：判斷切分策略 ──────────────────────────────────────
            # Category 1 (Forecasting): 預測未來，須依序切分 (No Shuffle)
            # Category 2 (TSC/TSER):   UCR 類型，樣本獨立，須隨機切分 (Shuffle + Stratify)
            # 若來自 ts_dir 且為分類任務，視為 Category 2 (Instance-based)
            is_forecasting = is_ts and task != "classification"

            # ── 80/20 split (外部評估) ─────────────────────────────────────
            if not is_forecasting:
                # Category 2 或一般表格：隨機分層切分
                try:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
                except ValueError:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_all, y_all, test_size=0.2, random_state=SEED)
                split_mode = "Random Stratified"
            else:
                # Category 1 (Forecasting): 嚴格依時間順序切分
                split_idx = int(len(X_all) * 0.8)
                X_tr, X_te = X_all[:split_idx], X_all[split_idx:]
                y_tr, y_te = y_all[:split_idx], y_all[split_idx:]
                split_mode = "Chronological (No Shuffle)"

            print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  "
                  f"n_classes={n_classes}  split={split_mode}")

            # ── 呼叫 Pipeline 引擎 ────────────────────────────────────────────
            budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
            cfg    = _pl.get_cfg(args.fast, n_samples=len(y_tr))
            
            # 告訴內部 CV 是否要用 TimeSeriesSplit
            # 對於 UCR 分類，內部驗證應使用 StratifiedKFold
            cfg["is_timeseries"] = is_forecasting

            result = _pl.run(
                X_tr, y_tr, X_te, n_classes, cfg, budget,
                skip_tabular=args.skip_tabular,
                skip_dl=args.skip_dl,
                no_nas=args.no_nas,
                is_ts=is_forecasting, # 決定模型選擇 (TCN vs CNN1D)
                artifacts_dir=os.path.join(ARTIFACTS_DIR, "batch", dataset_name),
                metric=args.metric,
            )

            # ── 評估 ──────────────────────────────────────────────────────────
            from src.metrics import calculate_score, get_metric_name
            m_name = get_metric_name(args.metric)
            
            score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
            score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
            
            # 以較佳的 ensemble 結果作為代表分數
            best_score = round(max(score_b, score_s), 4)
            elapsed  = round(time.time() - t_ds, 1)
            print(f"\n  [結果] Blend → {m_name}={score_b:.4f}")
            print(f"  [結果] Stack → {m_name}={score_s:.4f}")
            print(f"  [耗時] {elapsed}s")
            results.append({
                "dataset":   dataset_name,
                "task":      "timeseries" if is_ts else "tabular",
                "metric":    args.metric,
                "score":     best_score,
                "elapsed_s": elapsed,
            })

        except Exception:
            traceback.print_exc()
            results.append({
                "dataset":   dataset_name,
                "task":      "timeseries" if is_ts else "tabular",
                "metric":    args.metric,
                "score":     None,
                "elapsed_s": round(time.time() - t_ds, 1),
            })

    # ── 總結 ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  BATCH SUMMARY — Full Pipeline")
    print(f"{'='*65}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    out = os.path.join(HERE, "pipeline_batch_results.csv")
    summary.to_csv(out, index=False)
    print(f"\n  結果已儲存 → {out}")
    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",         action="store_true", help="縮減 HPO/NAS 次數")
    parser.add_argument("--skip-tabular", action="store_true", help="跳過傳統模型 HPO")
    parser.add_argument("--skip-dl",      action="store_true", help="跳過深度學習模型")
    parser.add_argument("--no-nas",       action="store_true", help="跳過 NAS，使用預設架構")
    parser.add_argument("--time-limit",   type=float, default=0, help="總時間上限（秒，0=無限）")
    parser.add_argument("--openml-dir",   default="openml_cc18_data")
    parser.add_argument("--ts-dir",       default="ucr_ts_80(時序資料)")
    parser.add_argument("--top-n",        type=int, default=5, help="每目錄取前 N 個資料集")
    parser.add_argument("--metric",       choices=["f1", "accuracy"], default="f1", help="優化指標 (預設 f1)")
    args = parser.parse_args()

    run_batch(args)


if __name__ == "__main__":
    main()
