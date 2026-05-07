"""
run_baseline.py  ─  對照組：AutoGluon TabularPredictor

用法：
  python run_baseline.py --csv openml_cc18_data/22_mfeat-zernike.csv
  python run_baseline.py --csv openml_cc18_data/22_mfeat-zernike.csv --time-budget 120
  python run_baseline.py --csv openml_cc18_data/22_mfeat-zernike.csv --compare --trials 20
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


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="對照組：AutoGluon TabularPredictor"
    )
    parser.add_argument("--csv",         required=True,  help="CSV 檔案路徑")
    parser.add_argument("--target",      default=None,   help="目標欄位名稱（預設自動偵測）")
    parser.add_argument("--task",        default=None,   choices=["classification", "regression"])
    parser.add_argument("--time-budget", type=int, default=120, help="訓練時間上限（秒，預設 120）")
    parser.add_argument("--presets",     default="medium_quality",
                        choices=["medium_quality", "good_quality", "best_quality"],
                        help="AutoGluon presets（預設 medium_quality）")
    parser.add_argument("--test-size",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--output-dir",  default="autogluon_models", help="AutoGluon 模型儲存目錄")
    parser.add_argument("--compare",     action="store_true",
                        help="同時執行自製 Pipeline 進行對比")
    parser.add_argument("--trials",      type=int, default=20,
                        help="自製 Pipeline HPO 試驗次數（--compare 時有效）")
    args = parser.parse_args()

    # 載入資料
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

    # 切割
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

    # ── AutoGluon ────────────────────────────────────────────────────────────
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    test_df = X_test.copy()

    ag_task = "multiclass" if (task == "classification" and y.nunique() > 2) else task

    # 清理舊的模型目錄，避免覆寫警告與殘留狀態
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

    # ── 自製 Pipeline（可選）────────────────────────────────────────────────
    if args.compare:
        print(f"\n{'='*60}")
        print(f"  自製 AutoML Pipeline  （HPO trials={args.trials}）")
        print(f"{'='*60}")
        try:
            from automl_platform.pipeline import AutoMLPipeline
            t0 = time.time()
            pipeline = AutoMLPipeline(
                task=task,
                n_hpo_trials=args.trials,
                n_folds=5,
                top_k_hpo=5,
                use_nas=True,
                mi_k=50,
                poly_max_cols=10,
                use_shap_pruning=True,
            )
            pipeline.fit(X_train, y_train)
            y_pred_custom = pipeline.predict(X_test)
            custom_elapsed = time.time() - t0

            print(f"\n  自製 Pipeline 測試集結果  （耗時 {custom_elapsed:.1f}s）")
            print_metrics(task, y_test.values, y_pred_custom, "CustomPipeline")

            # 差異摘要
            metric_key = "Accuracy" if task == "classification" else "R2"
            le = LabelEncoder()
            yt = le.fit_transform(pd.Series(y_test).astype(str))
            try:
                yp_ag  = le.transform(pd.Series(y_pred_ag).astype(str))
                yp_cus = le.transform(pd.Series(y_pred_custom).astype(str))
            except ValueError:
                yp_ag = yp_cus = np.zeros_like(yt)

            if task == "classification":
                ag_score  = accuracy_score(yt, yp_ag)
                cus_score = accuracy_score(yt, yp_cus)
            else:
                ag_score  = r2_score(y_test, y_pred_ag)
                cus_score = r2_score(y_test, y_pred_custom)

            delta = cus_score - ag_score
            sign  = "+" if delta >= 0 else ""
            print(f"\n  {metric_key} 差異：自製 {cus_score:.4f}  vs  AutoGluon {ag_score:.4f}  ({sign}{delta:.4f})")

        except Exception as e:
            print(f"  [CustomPipeline] 執行失敗：{e}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
