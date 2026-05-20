"""
run_pipeline_time.py — 時序專用 Pipeline 入口（v1）

僅掃 ucr_ts_80(時序資料)/，每個資料集自動偵測 task：
  - REG_* 前綴   → 回歸（用 pipeline_time.run_regression，Walk-forward 時序切分）
  - 其他（含 CLS_*）→ 分類（用 pipeline_time.run_classification，即 pipeline.run(is_ts=True)）

用法：
    # 批次（時序資料夾的前 N 個 / 後 N 個）
    python run_pipeline_time.py --batch --top-n 10
    python run_pipeline_time.py --batch --last --top-n 10
    python run_pipeline_time.py --batch --fast

    # 單一 CSV
    python run_pipeline_time.py --csv "ucr_ts_80(時序資料)/REG_VentilatorPressure.csv"

結果輸出：pipeline_time_batch_results.csv（每跑完一個資料集即 flush）
"""
import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.config import ARTIFACTS_DIR, DEVICE, SEED
import pipeline_time as _pt


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _find_target_col(df: pd.DataFrame) -> str:
    for c in ("target", "label", "class", "y", "c"):
        if c in df.columns:
            return c
    return df.columns[-1]


def _auto_detect_task(filename: str, y: pd.Series) -> str:
    """
    時序資料夾下的判斷規則：
      - 檔名以 REG_ 開頭 → regression
      - 否則：依 dtype + nunique 判斷（同 run_pipeline）
    """
    if os.path.basename(filename).startswith("REG_"):
        return "regression"
    if y.dtype == object or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    return "classification" if (n_unique <= 50 and n_unique / len(y) < 0.30) else "regression"


def _ts_datasets(ts_dir: str, top_n: int, last: bool) -> list:
    files = sorted(f for f in os.listdir(ts_dir) if f.endswith(".csv"))
    chosen = files[-top_n:] if last else files[:top_n]
    return [os.path.join(ts_dir, f) for f in chosen]


# ── 單一資料集執行 ───────────────────────────────────────────────────────────

def _process_one(csv_path: str, args, t_ds: float) -> dict:
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)
    target_col = args.target if args.target else _find_target_col(df)
    if target_col not in df.columns:
        raise ValueError(f"找不到目標欄 '{target_col}'，可用：{list(df.columns)}")

    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [info] dropped {n_before - len(df)} rows with NaN target")

    y_raw = df[target_col]
    task = _auto_detect_task(csv_path, y_raw)
    print(f"  [Task] {task}  target={target_col}")

    # 特徵：僅取數值欄、填 0
    X_all = (df.drop(columns=[target_col])
               .select_dtypes(include=[np.number])
               .fillna(0).values.astype(np.float32))

    if X_all.shape[1] == 0:
        raise ValueError("無可用數值特徵（select_dtypes 後 0 欄）")

    # ── 切分 ─────────────────────────────────────────────────────────────────
    if task == "classification":
        le = LabelEncoder()
        y_all = le.fit_transform(y_raw.astype(str).values)
        n_classes = len(le.classes_)
        # UCR 時序分類：樣本獨立，採隨機分層切分（與 run_pipeline.py 一致）
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=0.2, random_state=SEED)
        split_mode = "Random Stratified"
        print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  split={split_mode}")

        budget = _pt.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        # 分類沿用 pipeline.get_cfg
        import pipeline as _pl
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False  # UCR 分類視為樣本獨立

        result = _pt.run_classification(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            artifacts_dir=os.path.join(ARTIFACTS_DIR, "batch_time", dataset_name),
            metric=args.cls_metric,
        )
        from src.metrics import calculate_score
        score_b = calculate_score(y_te, result.test_blend, metric=args.cls_metric)
        score_s = calculate_score(y_te, result.test_stack, metric=args.cls_metric)
        best_preds = result.test_stack if score_s >= score_b else result.test_blend
        acc = round(calculate_score(y_te, best_preds, metric="accuracy"), 4)
        f1  = round(calculate_score(y_te, best_preds, metric="f1"), 4)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → {args.cls_metric}={score_b:.4f}")
        print(f"  [結果] Stack → {args.cls_metric}={score_s:.4f}")
        print(f"  [耗時] {elapsed}s")

        return {
            "dataset": dataset_name,
            "type": "TS",
            "task": task,
            "n_train": len(y_tr),
            "n_test": len(y_te),
            "accuracy": acc,
            "f1_macro": f1,
            "rmse": None,
            "r2": None,
            "score": round(max(score_b, score_s), 4),
            "elapsed_s": elapsed,
        }

    # ── 回歸 ────────────────────────────────────────────────────────────────
    y_all = np.asarray(y_raw.values, dtype=np.float32).ravel()
    # 時序回歸：嚴格依時間順序切分（最後 20% 為測試集）
    split_idx = int(len(X_all) * 0.8)
    X_tr, X_te = X_all[:split_idx], X_all[split_idx:]
    y_tr, y_te = y_all[:split_idx], y_all[split_idx:]
    split_mode = "Chronological (No Shuffle)"
    print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression  split={split_mode}")

    budget = _pt.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
    cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))

    result = _pt.run_regression(
        X_tr, y_tr, X_te, cfg, budget,
        skip_tabular=args.skip_tabular,
        skip_dl=args.skip_dl,
        artifacts_dir=os.path.join(ARTIFACTS_DIR, "batch_time", dataset_name),
        metric=args.reg_metric,
    )

    # 評估：固定回報 RMSE + R²（無論 metric）
    from sklearn.metrics import mean_squared_error, r2_score
    rmse_b = float(np.sqrt(mean_squared_error(y_te, result.test_blend)))
    rmse_s = float(np.sqrt(mean_squared_error(y_te, result.test_stack)))
    r2_b = float(r2_score(y_te, result.test_blend))
    r2_s = float(r2_score(y_te, result.test_stack))

    # 取較佳者：依 args.reg_metric 決定（rmse → 小者佳；r2 → 大者佳）
    if args.reg_metric == "r2":
        best_is_stack = r2_s >= r2_b
        primary_score = max(r2_b, r2_s)
    else:
        best_is_stack = rmse_s <= rmse_b
        primary_score = min(rmse_b, rmse_s)

    best_rmse = rmse_s if best_is_stack else rmse_b
    best_r2 = r2_s if best_is_stack else r2_b
    elapsed = round(time.time() - t_ds, 1)
    print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
    print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}")
    print(f"  [耗時] {elapsed}s")

    return {
        "dataset": dataset_name,
        "type": "TS",
        "task": task,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "accuracy": None,
        "f1_macro": None,
        "rmse": round(best_rmse, 4),
        "r2": round(best_r2, 4),
        "score": round(primary_score, 4),
        "elapsed_s": elapsed,
    }


# ── 批次模式 ────────────────────────────────────────────────────────────────

def run_batch(args):
    ts_dir = os.path.join(HERE, args.ts_dir)
    if not os.path.isdir(ts_dir):
        print(f"[錯誤] 找不到目錄：{ts_dir}"); sys.exit(1)

    datasets = _ts_datasets(ts_dir, args.top_n, args.last)
    print(f"\n{'='*65}")
    print(f"  Pipeline-Time 批次  ─  {len(datasets)} 個 TS 資料集  fast={args.fast}")
    print(f"{'='*65}")

    out_path = os.path.join(HERE, args.out)

    # 斷點續跑：載入已有結果，跳過已完成的 dataset
    done_datasets = set()
    results = []
    if os.path.exists(out_path):
        try:
            existing = pd.read_csv(out_path)
            results = existing.to_dict("records")
            # 只跳過有真實結果的列（task != "?"），錯誤列重新嘗試
            valid = existing[existing["task"] != "?"]
            done_datasets = set(valid["dataset"].tolist())
            retry = set(existing["dataset"].tolist()) - done_datasets
            if retry:
                print(f"  [Resume] 將重試失敗的 dataset: {sorted(retry)}")
                results = valid.to_dict("records")  # 移除舊的錯誤列
            print(f"  [Resume] 載入 {len(results)} 筆已有結果，將跳過: {sorted(done_datasets)}")
        except Exception as e:
            print(f"  [Warn] 無法載入既有結果（{e}），從頭開始")

    def _flush():
        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)

    for csv_path in datasets:
        name = os.path.splitext(os.path.basename(csv_path))[0]
        if name in done_datasets:
            print(f"\n  [Skip] {name}（已有結果，跳過）")
            continue
        print(f"\n{'─'*65}")
        print(f"  [TS] {name}  |  device={DEVICE}")
        print(f"{'─'*65}")
        t_ds = time.time()
        try:
            row = _process_one(csv_path, args, t_ds)
            results.append(row)
        except Exception:
            traceback.print_exc()
            results.append({
                "dataset": name, "type": "TS", "task": "?",
                "n_train": None, "n_test": None,
                "accuracy": None, "f1_macro": None,
                "rmse": None, "r2": None, "score": None,
                "elapsed_s": round(time.time() - t_ds, 1),
            })
        _flush()

    print(f"\n{'='*65}")
    print("  BATCH SUMMARY — Pipeline Time")
    print(f"{'='*65}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    _flush()
    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


# ── 單一 CSV 模式 ────────────────────────────────────────────────────────────

def run_single(args):
    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"[錯誤] 找不到檔案：{csv_path}"); sys.exit(1)
    name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n{'='*65}")
    print(f"  Pipeline-Time 單檔  ─  {name}  |  device={DEVICE}")
    print(f"{'='*65}")
    t_ds = time.time()
    try:
        row = _process_one(csv_path, args, t_ds)
        print("\n  [完成]", row)
    except Exception:
        traceback.print_exc()


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="時序專用 Pipeline 入口（分類 + 回歸）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--fast", action="store_true", help="縮減 HPO 次數")
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--no-nas", action="store_true",
                        help="分類模式下跳過 NAS（回歸無 NAS）")
    parser.add_argument("--time-limit", type=float, default=0,
                        help="總時間上限（秒，0=無限）")
    parser.add_argument("--cls-metric", choices=["f1", "accuracy"], default="f1",
                        help="分類優化指標")
    parser.add_argument("--reg-metric", choices=["rmse", "r2", "mae"], default="rmse",
                        help="回歸優化指標")

    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--ts-dir", default="ucr_ts_80(時序資料)")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--last", action="store_true",
                        help="取每目錄最後 top-n 個")
    parser.add_argument("--out", default="pipeline_time_batch_results.csv",
                        help="批次模式輸出 CSV")

    parser.add_argument("--csv", default=None)
    parser.add_argument("--target", default=None)

    args = parser.parse_args()
    if args.csv:
        run_single(args)
    elif args.batch:
        run_batch(args)
    else:
        parser.print_help()
        print("\n[提示] 請指定 --csv <path> 或 --batch。")


if __name__ == "__main__":
    main()
