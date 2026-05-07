"""
Baseline vs AutoMLPipeline 對比驗證腳本

執行順序（每個資料集）：
  1. 先跑 AutoMLPipeline，記錄耗時 p_elapsed
  2. 再跑 AutoGluon，time_limit = p_elapsed（兩者使用相同計算預算）

各取前 5 個表格資料集 + 前 5 個時序資料集，結果覆蓋至：
    automl_platform/results/validation_results.csv
"""
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

from automl_platform.pipeline import AutoMLPipeline

# ── 設定 ──────────────────────────────────────────────────────────────────────
TABULAR_DIR  = os.path.join(HERE, "openml_cc18_data")
TS_DIR       = os.path.join(HERE, "ucr_ts_80(時序資料)")
REPORT_DIR   = os.path.join(HERE, "automl_platform", "results")
OUT_CSV      = os.path.join(REPORT_DIR, "validation_results.csv")
AG_MODEL_DIR = os.path.join(REPORT_DIR, "_ag_tmp")   # AutoGluon 暫存，用完即刪

N_TABULAR    = 1
N_TS         = 1
N_HPO_TRIALS = 20
USE_NAS      = True
TEST_SIZE    = 0.2
RANDOM_STATE = 42
AG_PRESETS   = "good_quality"


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def detect_task(y: pd.Series, filename: str = "") -> str:
    base = os.path.basename(filename)
    if base.startswith("CLS_"):
        return "classification"
    if base.startswith("REG_"):
        return "regression"
    if y.dtype == bool or y.dtype == object:
        return "classification"
    n_unique = y.nunique()
    if n_unique <= 50 and n_unique / len(y) < 0.3:
        return "classification"
    return "regression"


def get_target_col(df: pd.DataFrame) -> str:
    for cand in ("target", "label", "class", "y", "c"):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def load_dataset(filepath: str):
    df = pd.read_csv(filepath)
    target_col = get_target_col(df)
    y = df[target_col]
    X = df.drop(columns=[target_col]).dropna(axis=1, how="all")
    return X, y, target_col


def eval_metrics(task: str, y_true, y_pred) -> dict:
    if task == "classification":
        le = LabelEncoder()
        yt = le.fit_transform(pd.Series(y_true).astype(str))
        try:
            yp = le.transform(pd.Series(y_pred).astype(str))
        except ValueError:
            yp = np.array([
                le.transform([str(v)])[0] if str(v) in le.classes_ else 0
                for v in y_pred
            ])
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", zero_division=0)
        return {"accuracy": round(acc, 6), "f1_macro": round(f1, 6)}
    else:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2   = float(r2_score(y_true, y_pred))
        return {"rmse": round(rmse, 6), "r2": round(r2, 6)}


# ── Baseline：AutoGluon（time_limit 由 pipeline 耗時決定）───────────────────────

def run_autogluon(X_train: pd.DataFrame, y_train: pd.Series,
                  X_test: pd.DataFrame, task: str,
                  target_col: str, time_limit: int) -> tuple:
    if os.path.exists(AG_MODEL_DIR):
        shutil.rmtree(AG_MODEL_DIR, ignore_errors=True)

    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    n_unique = y_train.nunique()
    if task == "classification":
        problem_type = "binary" if n_unique == 2 else "multiclass"
    else:
        problem_type = "regression"

    print(f"  [Baseline] AutoGluon ({AG_PRESETS}, time_limit={time_limit}s)...")
    t0 = time.time()
    predictor = TabularPredictor(
        label=target_col,
        problem_type=problem_type,
        path=AG_MODEL_DIR,
        verbosity=0,
    ).fit(
        train_df,
        time_limit=time_limit,
        presets=AG_PRESETS,
        dynamic_stacking=False,
        excluded_model_types=["FASTAI", "NeuralNetTorch", "LightGBMXT"],
        ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
    )
    y_pred = predictor.predict(X_test).values
    elapsed = time.time() - t0

    shutil.rmtree(AG_MODEL_DIR, ignore_errors=True)
    return y_pred, elapsed


# ── 主流程 ────────────────────────────────────────────────────────────────────

def _adapt_pipeline_config(n_train: int, n_features: int, is_timeseries: bool) -> dict:
    """根據訓練集大小自動調整 AutoMLPipeline 超參數，避免 OOM / 過擬合 / 超慢問題。"""
    shap = not is_timeseries
    if n_train < 200:                   # 極小：3-fold，停用 NAS / Poly / SHAP
        return dict(n_hpo_trials=15, n_folds=3, top_k_hpo=3,
                    use_nas=False, mi_k=min(30, n_features),
                    poly_max_cols=0, use_shap_pruning=False)
    if n_train < 500:                   # 小：CLAUDE.md 警告的三重過擬合區間
        return dict(n_hpo_trials=15, n_folds=5, top_k_hpo=3,
                    use_nas=False, mi_k=min(40, n_features),
                    poly_max_cols=5, use_shap_pruning=shap)
    if n_train < 20_000:                # 中：原始設定最適區間
        return dict(n_hpo_trials=N_HPO_TRIALS, n_folds=5, top_k_hpo=5,
                    use_nas=USE_NAS, mi_k=50,
                    poly_max_cols=10, use_shap_pruning=shap)
    if n_train < 100_000:               # 大：NAS OOF 太慢，縮小 poly
        return dict(n_hpo_trials=N_HPO_TRIALS, n_folds=3, top_k_hpo=5,
                    use_nas=False, mi_k=50,
                    poly_max_cols=5, use_shap_pruning=shap)
    # 超大：poly 會 GPU OOM，全部關閉昂貴操作
    return dict(n_hpo_trials=N_HPO_TRIALS, n_folds=3, top_k_hpo=3,
                use_nas=False, mi_k=50,
                poly_max_cols=0, use_shap_pruning=False)


def run_one_dataset(filepath: str, is_timeseries: bool) -> dict:
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    dtype = "timeseries" if is_timeseries else "tabular"
    print(f"\n{'─'*60}")
    print(f"  [{dtype.upper()}] {dataset_name}")
    print(f"{'─'*60}")

    try:
        X, y, target_col = load_dataset(filepath)
        task = detect_task(y, filepath)
        print(f"  Shape: {X.shape}  |  Task: {task}  |  Target unique: {y.nunique()}")

        try:
            stratify = y if task == "classification" else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

        print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

        # ── Step 1：先跑 Pipeline，記錄耗時 ──────────────────────────────
        print("\n  [Pipeline] AutoMLPipeline...")
        cfg = _adapt_pipeline_config(len(X_train), X_train.shape[1], is_timeseries)
        print(f"  [Pipeline] config={cfg}")
        pipeline = AutoMLPipeline(task=task, is_timeseries=is_timeseries, **cfg)
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        p_pred = pipeline.predict(X_test)
        p_elapsed = time.time() - t0
        p_metrics = eval_metrics(task, y_test, p_pred)
        print(f"  [Pipeline] elapsed={p_elapsed:.1f}s  metrics={p_metrics}")

        # ── Step 2：AutoGluon，time_limit = pipeline 耗時 ────────────────
        ag_time_limit = max(30, int(p_elapsed))   # 至少給 30 秒
        b_pred, b_elapsed = run_autogluon(
            X_train, y_train, X_test, task, target_col, ag_time_limit
        )
        b_metrics = eval_metrics(task, y_test, b_pred)
        print(f"  [Baseline] elapsed={b_elapsed:.1f}s  metrics={b_metrics}")

        # ── 組合成一列 ────────────────────────────────────────────────────
        row = {
            "dataset":        dataset_name,
            "type":           dtype,
            "task":           task,
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "n_features_raw": X.shape[1],
            "pipeline_elapsed_sec": round(p_elapsed, 1),
        }
        for k, v in p_metrics.items():
            row[f"pipeline_{k}"] = v
        row["baseline_elapsed_sec"] = round(b_elapsed, 1)
        for k, v in b_metrics.items():
            row[f"baseline_{k}"] = v

        return row

    except Exception as e:
        import traceback
        print(f"  [ERROR] {e}")
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "type":    dtype,
            "task":    "unknown",
            "error":   str(e),
        }


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    all_results = []

    # ── 表格資料集（前 N 個）────────────────────────────────────────────────
    tabular_files = sorted([
        f for f in os.listdir(TABULAR_DIR) if f.endswith(".csv")
    ])[:N_TABULAR]

    print(f"\n{'='*60}")
    print(f"  表格資料集  ({len(tabular_files)} 個)")
    print(f"{'='*60}")
    for fname in tabular_files:
        row = run_one_dataset(os.path.join(TABULAR_DIR, fname), is_timeseries=False)
        all_results.append(row)

    # ── 時序資料集（前 N 個）────────────────────────────────────────────────
    ts_files = sorted([
        f for f in os.listdir(TS_DIR) if f.endswith(".csv")
    ])[:N_TS]

    print(f"\n{'='*60}")
    print(f"  時序資料集  ({len(ts_files)} 個)")
    
    print(f"{'='*60}")
    for fname in ts_files:
        row = run_one_dataset(os.path.join(TS_DIR, fname), is_timeseries=True)
        all_results.append(row)

    # ── 輸出結果 ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)

    meta_cols     = ["dataset", "type", "task", "n_train", "n_test", "n_features_raw"]
    pipeline_cols = [c for c in df.columns if c.startswith("pipeline_")]
    baseline_cols = [c for c in df.columns if c.startswith("baseline_")]
    other_cols    = [c for c in df.columns
                     if c not in meta_cols + pipeline_cols + baseline_cols]
    ordered = meta_cols + pipeline_cols + baseline_cols + other_cols
    df = df[[c for c in ordered if c in df.columns]]

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n{'='*60}")
    print(f"  結果已儲存至 {OUT_CSV}")
    print(f"{'='*60}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
