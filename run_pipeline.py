"""
run_pipeline.py — 非時序資料 Pipeline 執行入口（分類 + 回歸）

批次評估模式（openml_cc18_data/ 前 N 個，80/20 random split）：
    python run_pipeline.py --batch [--fast] [--top-n 5]
    python run_pipeline.py --batch --top-n 10 --last
    python run_pipeline.py --batch --skip-dl --top-n 3

單一 CSV 模式：
    python run_pipeline.py --csv mydata.csv
    python run_pipeline.py --csv mydata.csv --target label

預切分模式（手動提供 TRAIN / TEST）：
    python run_pipeline.py --train train.csv --test test.csv
    python run_pipeline.py --train train.csv --test test.csv --target label

注意：時序資料（UCR）請改用 run_pipeline_time.py
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
from sklearn.metrics import mean_squared_error, r2_score

import joblib # 💾 用來儲存預處理引擎大腦
from sklearn.pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.config import DEVICE, SEED, ARTIFACTS_DIR
import pipeline as _pl
import pipeline_time as _pt

# 🚀 引入 AutoML 2.0 煉金引擎
from preprocessing.interface import preprocess_for_training
from preprocessing.interface import load_and_merge_data

# 供預切分模式使用的底層組件
try:
    from preprocessing.interface import AutoRouter, PipelineAssembler
    from preprocessing.processors.feature_generator import RobustDataCleaner
except ImportError:
    pass # 避免部分環境下匯入層級問題

_RESULT_COLS = [
    "source", "dataset", "type", "task", "n_train", "n_test",
    "accuracy", "f1_macro", "rmse", "r2", "score", "elapsed_s",
]

def _append_result(out_path: str, row: dict):
    df = pd.DataFrame([{c: row.get(c) for c in _RESULT_COLS}])
    df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)


# ── 資料輔助函式 ──────────────────────────────────────────────────────────────

def _auto_detect_task(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    return "classification" if (n_unique <= 50 and n_unique / len(y) < 0.30) else "regression"


def _find_target_col(df: pd.DataFrame) -> str:
    for cand in ("target", "label", "class", "y", "c", "isFraud"):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def _find_datasets(openml_dir: str, top_n: int, last: bool = False, force_task: str = None):
    """回傳 (dataset_name, csv_path, force_task) 列表。"""
    all_csv = sorted(f for f in os.listdir(openml_dir) if f.endswith(".csv"))
    chosen = all_csv[-top_n:] if last else all_csv[:top_n]
    return [(os.path.splitext(f)[0], os.path.join(openml_dir, f), force_task) for f in chosen]


def _eval_regression(y_te, result, reg_metric: str):
    """計算回歸評估指標，回傳 (rmse_b, rmse_s, r2_b, r2_s, best_rmse, best_r2, primary_score)。"""
    rmse_b = float(np.sqrt(mean_squared_error(y_te, result.test_blend)))
    rmse_s = float(np.sqrt(mean_squared_error(y_te, result.test_stack)))
    r2_b   = float(r2_score(y_te, result.test_blend))
    r2_s   = float(r2_score(y_te, result.test_stack))
    if reg_metric == "r2":
        best_is_stack = r2_s >= r2_b
        primary_score = max(r2_b, r2_s)
    else:
        best_is_stack = rmse_s <= rmse_b
        primary_score = min(rmse_b, rmse_s)
    best_rmse = rmse_s if best_is_stack else rmse_b
    best_r2   = r2_s   if best_is_stack else r2_b
    return rmse_b, rmse_s, r2_b, r2_s, best_rmse, best_r2, primary_score


# ── 批次評估模式 ──────────────────────────────────────────────────────────────

def run_batch(args):
    openml_dir = os.path.join(HERE, args.openml_dir)
    reg_dir    = os.path.join(HERE, args.reg_dir)

    datasets = []
    if os.path.isdir(openml_dir):
        datasets += _find_datasets(openml_dir, args.top_n, last=args.last)
    if os.path.isdir(reg_dir):
        datasets += _find_datasets(reg_dir, args.reg_top_n, last=args.last, force_task="regression")
    if not datasets:
        print("[錯誤] 無可用資料集"); sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Pipeline 批次評估（非時序）  ─  {len(datasets)} 個資料集  fast={args.fast}")
    print(f"{'='*65}")
    results = []
    out_path = os.path.join(HERE, "pipeline_batch_results.csv")

    def _flush():
        if results:
            pd.DataFrame(results).to_csv(out_path, index=False)

    def _flush_result():
        if getattr(args, "result_file", None) and results:
            _append_result(args.result_file, {**results[-1], "source": "pipeline"})

    for dataset_name, csv_path, force_task in datasets:
        print(f"\n{'─'*65}")
        print(f"  [Tab] {dataset_name}  |  device={DEVICE}")
        print(f"{'─'*65}")
        t_ds = time.time()
        task = "?"
        try:
            df = pd.read_csv(csv_path)
            target_col = _find_target_col(df)
            
            # 🚀 啟動 AutoML 2.0 預處理引擎接管資料
            print("  [AutoML 2.0] 啟動雙軌預處理引擎...")
            X_tr_dict, X_te_dict, y_tr_raw, y_te_raw, fitted_preprocessors = preprocess_for_training(
                data_source=df,
                target_col=target_col,
                test_size=0.2
            )
            
            task = force_task if force_task else _auto_detect_task(y_tr_raw)
            
            # ==========================================
            # 🚀 雙軌制資料轉換 (float32 節省記憶體)
            # ==========================================
            X_tr_tree = X_tr_dict["tree"].values.astype(np.float32)
            X_te_tree = X_te_dict["tree"].values.astype(np.float32)
            X_tr_dl = X_tr_dict["dl"].values.astype(np.float32)
            X_te_dl = X_te_dict["dl"].values.astype(np.float32)

            # 應該把字典傳進去： 
            #X_tr = {"tree": X_tr_tree, "dl": X_tr_dl}
            X_tr = X_tr_tree
            X_te = X_te_tree
            # (等確認分數彈回來後，再跟模型組說把 _pl.run 改成接收雙軌字典)
            # ==========================================

            # 💾 儲存大腦 (注意變數名稱加了 s，因為現在是雙軌字典)
            art_dir = os.path.join(ARTIFACTS_DIR, "batch", dataset_name)
            os.makedirs(art_dir, exist_ok=True)
            joblib.dump(fitted_preprocessors, os.path.join(art_dir, "fitted_preprocessors.pkl"))

            if task == "classification":
                le = LabelEncoder()
                y_tr = le.fit_transform(y_tr_raw.astype(str).values)
                classes_set = set(le.classes_)
                y_te = np.array([le.transform([str(v)])[0] if str(v) in classes_set else 0 for v in y_te_raw], dtype=np.int64)
                n_classes = len(le.classes_)
                print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}")

                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
                cfg["is_timeseries"] = False
                result = _pl.run(
                    X_tr, y_tr, X_te, n_classes, cfg, budget,
                    skip_tabular=args.skip_tabular,
                    skip_dl=args.skip_dl,
                    no_nas=args.no_nas,
                    is_ts=False,
                    artifacts_dir=art_dir,
                    metric=args.metric,
                )
                from src.metrics import calculate_score, get_metric_name
                score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
                score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
                best_preds = result.test_stack if score_s >= score_b else result.test_blend
                best_score = round(max(score_b, score_s), 4)
                acc  = round(calculate_score(y_te, best_preds, metric="accuracy"), 4)
                f1   = round(calculate_score(y_te, best_preds, metric="f1"), 4)
                elapsed = round(time.time() - t_ds, 1)
                
                print(f"\n  [結果] Blend → {get_metric_name(args.metric)}={score_b:.4f}")
                print(f"  [結果] Stack → {get_metric_name(args.metric)}={score_s:.4f}")
                print(f"  [耗時] {elapsed}s")
                results.append({
                    "dataset": dataset_name, "type": "Tab", "task": task,
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "accuracy": acc, "f1_macro": f1,
                    "rmse": None, "r2": None,
                    "score": best_score, "elapsed_s": elapsed,
                })

            else:  # regression
                y_tr = np.asarray(y_tr_raw.values, dtype=np.float32).ravel()
                y_te = np.asarray(y_te_raw.values, dtype=np.float32).ravel()
                print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression")

                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
                result = _pt.run_regression(
                    X_tr, y_tr, X_te, cfg, budget,
                    skip_tabular=args.skip_tabular,
                    skip_dl=args.skip_dl,
                    artifacts_dir=art_dir,
                    metric=args.reg_metric,
                )
                rmse_b, rmse_s, r2_b, r2_s, best_rmse, best_r2, primary_score = _eval_regression(y_te, result, args.reg_metric)
                elapsed = round(time.time() - t_ds, 1)
                print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
                print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}")
                print(f"  [耗時] {elapsed}s")
                results.append({
                    "dataset": dataset_name, "type": "Tab", "task": task,
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "accuracy": None, "f1_macro": None,
                    "rmse": round(best_rmse, 4), "r2": round(best_r2, 4),
                    "score": round(primary_score, 4), "elapsed_s": elapsed,
                })

            _flush()
            _flush_result()

        except Exception:
            traceback.print_exc()
            results.append({
                "dataset": dataset_name, "type": "Tab", "task": task,
                "n_train": None, "n_test": None,
                "accuracy": None, "f1_macro": None,
                "rmse": None, "r2": None,
                "score": None, "elapsed_s": round(time.time() - t_ds, 1),
            })
            _flush()
            _flush_result()

    print(f"\n{'='*65}")
    print("  BATCH SUMMARY — Pipeline（非時序）")
    print(f"{'='*65}")
    print(pd.DataFrame(results).to_string(index=False))
    _flush()
    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


# ── 單一 CSV 模式 ──────────────────────────────────────────────────────────────

def run_single(args):
    csv_paths = args.csv # 現在這是一個 List[str]，例如 ['train_transaction.csv', 'train_identity.csv']
    
    # 檢查檔案是否存在
    for p in csv_paths:
        if not os.path.isfile(p):
            print(f"[錯誤] 找不到檔案：{p}"); sys.exit(1)

    # 取得主檔名用來命名 artifacts
    dataset_name = os.path.splitext(os.path.basename(csv_paths[0]))[0]
    print(f"\n{'='*65}")
    print(f"  Pipeline 單資料集模式（非時序）  ─  {dataset_name}")
    print(f"{'='*65}")
    t_ds = time.time()

    # 🚀 啟動 AutoML 2.0 預處理引擎 (包含 Data_loader 的多檔合併與壓縮)
    print("  [AutoML 2.0] 呼叫 Data_loader 與預處理引擎...")
    
    # 這裡直接傳入 csv_paths (List[str])，你們的 interface 會自動呼叫 data_loader 處理 join
    X_tr_df, X_te_df, y_tr_raw, y_te_raw, fitted_preprocessor = preprocess_for_training(
        data_source=csv_paths, 
        target_col=args.target,
        test_size=0.2
    )

    task = _auto_detect_task(y_tr_raw)
    
    # 轉換為 float32
    X_tr = X_tr_df.values.astype(np.float32)
    X_te = X_te_df.values.astype(np.float32)

    # 💾 儲存預處理大腦
    art_dir = os.path.join(ARTIFACTS_DIR, "single", dataset_name)
    os.makedirs(art_dir, exist_ok=True)
    joblib.dump(fitted_preprocessor, os.path.join(art_dir, "fitted_preprocessor.pkl"))

    if task == "classification":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_raw.astype(str).values)
        classes_set = set(le.classes_)
        y_te = np.array([le.transform([str(v)])[0] if str(v) in classes_set else 0 for v in y_te_raw], dtype=np.int64)
        n_classes = len(le.classes_)
        print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}")

        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False
        result = _pl.run(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            is_ts=False,
            artifacts_dir=art_dir,
            metric=args.metric,
        )
        from src.metrics import calculate_score, get_metric_name
        score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
        score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → {get_metric_name(args.metric)}={score_b:.4f}")
        print(f"  [結果] Stack → {get_metric_name(args.metric)}={score_s:.4f}")
        print(f"  [耗時] {elapsed}s")

    else:  # regression
        y_tr = np.asarray(y_tr_raw.values, dtype=np.float32).ravel()
        y_te = np.asarray(y_te_raw.values, dtype=np.float32).ravel()
        print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression")

        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
        result = _pt.run_regression(
            X_tr, y_tr, X_te, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            artifacts_dir=art_dir,
            metric=args.reg_metric,
        )
        rmse_b, rmse_s, r2_b, r2_s, _, _, _ = _eval_regression(y_te, result, args.reg_metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
        print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}")
        print(f"  [耗時] {elapsed}s")

    print(f"{'='*65}\n")


# ── 預切分模式 ────────────────────────────────────────────────────────────────

def run_presplit(args):
    """用戶手動提供 TRAIN / TEST 兩個 CSV，由 AutoML 2.0 進行無縫接軌轉換。"""
    # 1. 因為 nargs='+'，現在 args.train 和 args.test 都是 List[str]
    train_paths = args.train
    test_paths  = args.test

    # 2. 驗證所有檔案是否存在
    for p in train_paths + test_paths:
        if not os.path.isfile(p):
            print(f"[錯誤] 找不到檔案：{p}"); sys.exit(1)

    base = os.path.splitext(os.path.basename(train_paths[0]))[0]
    dataset_name = base[:-6] if base.endswith("_TRAIN") else base

    print(f"\n{'='*65}")
    print(f"  Pipeline 預切分模式（非時序）  ─  {dataset_name}  |  device={DEVICE}")
    print(f"{'='*65}")

    # 3. 建立多檔讀取與合併器 (支援 Left Join)
    print("  [AutoML 2.0] 呼叫底層 Data Loader 讀取並合併多重檔案...")
    train_df = load_and_merge_data(train_paths)
    test_df  = load_and_merge_data(test_paths)

    target_col = args.target if args.target else _find_target_col(train_df)
    if target_col not in train_df.columns:
        print(f"[錯誤] 找不到目標欄位 '{target_col}'，可用欄位：{list(train_df.columns)}")
        sys.exit(1)

    # =============== 處理 Train (一定有答案) ===============
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    y_tr_raw = train_df[target_col]
    X_tr_raw = train_df.drop(columns=[target_col])
    
    # =============== 處理 Test (盲測防呆) ===============
    has_test_labels = target_col in test_df.columns
    if has_test_labels:
        test_df = test_df.dropna(subset=[target_col]).reset_index(drop=True)
        y_te_raw = test_df[target_col]
        X_te_raw = test_df.drop(columns=[target_col])
    else:
        print("  [提示] 偵測到測試集無目標欄位，進入「盲測生成模式 (Inference)」")
        y_te_raw = pd.Series(np.zeros(len(test_df))) # 塞入假標籤防呆
        X_te_raw = test_df.copy()

    task = _auto_detect_task(y_tr_raw)

    # 🚀 在預切分模式下，手動掛載 AutoML 2.0 預處理引擎
    print("  [AutoML 2.0] 啟動預切分模式專屬煉金引擎...")
    
    # 進行 Phase 0 輕量探路
    temp_cleaner = RobustDataCleaner()
    X_sample = X_tr_raw.sample(n=min(5000, len(X_tr_raw)), random_state=SEED)
    X_sample_arr = temp_cleaner.fit_transform(X_sample)
    X_sample_df = pd.DataFrame(X_sample_arr, columns=temp_cleaner.get_feature_names_out())
    
    # Phase 1 藍圖分析
    router = AutoRouter(categorical_threshold=50, text_length_threshold=20)
    feature_groups = router.fit_predict(X_sample_df)
    assembler = PipelineAssembler(feature_groups)
    
    # 封裝終極大腦
    fitted_preprocessor = Pipeline([
        ('phase0', RobustDataCleaner()),
        ('phase1', assembler.build())
    ])
    
    # 正式轉換
    X_tr_arr = fitted_preprocessor.fit_transform(X_tr_raw, y_tr_raw)
    X_te_arr = fitted_preprocessor.transform(X_te_raw)
    
    X_tr = X_tr_arr.astype(np.float32)
    X_te = X_te_arr.astype(np.float32)
    
    # 對齊維度防爆
    min_cols = min(X_tr.shape[1], X_te.shape[1])
    X_tr = X_tr[:, :min_cols]
    X_te = X_te[:, :min_cols]

    # 💾 儲存預處理大腦
    art_dir = os.path.join(ARTIFACTS_DIR, "single", dataset_name)
    os.makedirs(art_dir, exist_ok=True)
    joblib.dump(fitted_preprocessor, os.path.join(art_dir, "fitted_preprocessor.pkl"))

    t_ds = time.time()

    if task == "classification":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_raw.astype(str).values)
        classes_set = set(le.classes_)
        y_te = np.array([le.transform([str(v)])[0] if str(v) in classes_set else 0 for v in y_te_raw], dtype=np.int64)
        n_classes = len(le.classes_)
        print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  task=classification")

        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False
        result = _pl.run(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            is_ts=False,
            artifacts_dir=art_dir,
            metric=args.metric,
        )
        from src.metrics import calculate_score, get_metric_name
        elapsed = round(time.time() - t_ds, 1)
        
        if has_test_labels:
            score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
            score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
            print(f"\n  [結果] Blend → {get_metric_name(args.metric)}={score_b:.4f}")
            print(f"  [結果] Stack → {get_metric_name(args.metric)}={score_s:.4f}")
        else:
            score_b, score_s = 0.0, 0.0
            print(f"\n  [結果] 未知測試集，無法計算分數，直接產生預測！")
        print(f"  [耗時] {elapsed}s")

        # ---------------------------------------------------------
        # 🚀 加上這段：自動產出 Submission CSV
        # ---------------------------------------------------------
        print("\n==================================================")
        print("🏆 產生最終預測結果 (Submission)")
        print("==================================================")
        
        # 1. 決定要用哪一個 Ensemble 的結果 (挑分數高的)
        # 假設分類任務的結果是機率分佈，我們取 argmax 轉回整數標籤
        # (如果 test_stack 已經是整數標籤，就不需要 argmax)
        if len(result.test_stack.shape) > 1 and result.test_stack.shape[1] > 1:
            best_preds = np.argmax(result.test_stack if score_s >= score_b else result.test_blend, axis=1)
        else:
            best_preds = result.test_stack if score_s >= score_b else result.test_blend

        # 2. 抓取測試集的 ID 欄位 (通常叫做 'id' 或 'ID'，依你的資料集而定)
        # 這裡會自動尋找名為 id 的欄位，如果沒有就預設拿第一欄
        id_col = 'id' if 'id' in test_df.columns else ('ID' if 'ID' in test_df.columns else test_df.columns[0])
        test_ids = test_df[id_col].values

        # 3. 呼叫模型組的 generate_submission
        from src.make_submission import generate_submission
        try:
            generate_submission(
                preds=best_preds,
                test_ids=test_ids,
                label_encoder=le, # 💡 run_presplit 裡面已經有定義好的 le (LabelEncoder) 了！
                out_name=f"submission_{dataset_name}.csv",
                target_col=target_col,
                id_col=id_col
            )
        except Exception as e:
            print(f"⚠️ 產生 Submission 時發生錯誤: {e}")

    else:  # regression
        y_tr = np.asarray(y_tr_raw.values, dtype=np.float32).ravel()
        y_te = np.asarray(y_te_raw.values, dtype=np.float32).ravel()
        print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression")

        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
        result = _pt.run_regression(
            X_tr, y_tr, X_te, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            artifacts_dir=art_dir,
            metric=args.reg_metric,
        )
        rmse_b, rmse_s, r2_b, r2_s, _, _, _ = _eval_regression(y_te, result, args.reg_metric)
        elapsed = round(time.time() - t_ds, 1)
        
        if has_test_labels:
            rmse_b, rmse_s, r2_b, r2_s, _, _, _ = _eval_regression(y_te, result, args.reg_metric)
            print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
            print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}")
        else:
            rmse_b, rmse_s, r2_b, r2_s = 0.0, 0.0, 0.0, 0.0
            print(f"\n  [結果] 未知測試集，無法計算誤差，直接產生預測！")
        print(f"  [耗時] {elapsed}s")

        # ---------------------------------------------------------
        # 🚀 加上這段：自動產出 Submission CSV (回歸專用版)
        # ---------------------------------------------------------
        print("\n==================================================")
        print("🏆 產生最終預測結果 (Submission - Regression)")
        print("==================================================")

        # 1. 決定要用哪一個 Ensemble 的結果
        # 如果指標是 r2 (越大越好)，如果是 rmse/mae (越小越好)
        if args.reg_metric == "r2":
            best_preds = result.test_stack if r2_s >= r2_b else result.test_blend
        else:
            best_preds = result.test_stack if rmse_s <= rmse_b else result.test_blend

        # 確保是一維陣列
        best_preds = best_preds.ravel()

        # 2. 抓取測試集的 ID 欄位
        id_col = 'id' if 'id' in test_df.columns else ('ID' if 'ID' in test_df.columns else test_df.columns[0])
        test_ids = test_df[id_col].values

        # 3. 直接用 Pandas 儲存結果 (避開分類專用的 generate_submission)
        out_name = f"submission_{dataset_name}.csv"
        out_path = os.path.join("submissions", out_name)
        os.makedirs("submissions", exist_ok=True) # 確保資料夾存在
        
        df_sub = pd.DataFrame({id_col: test_ids, target_col: best_preds})
        df_sub.to_csv(out_path, index=False)
        
        print(f"  [Submit] Saved → {out_path}  shape={df_sub.shape}")

    print(f"{'='*65}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AutoML Pipeline 執行入口（非時序：分類 + 回歸）\n"
                    "時序資料請改用 run_pipeline_time.py",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    # 通用旗標
    parser.add_argument("--fast",         action="store_true", help="縮減 HPO/NAS 次數")
    parser.add_argument("--skip-tabular", action="store_true", help="跳過傳統模型 HPO")
    parser.add_argument("--skip-dl",      action="store_true", help="跳過深度學習模型")
    parser.add_argument("--no-nas",       action="store_true", help="跳過 NAS，使用預設架構")
    parser.add_argument("--time-limit",   type=float, default=0, help="總時間上限（秒，0=無限）")
    parser.add_argument("--metric",       choices=["f1", "accuracy"], default="f1",
                        help="分類優化指標（預設 f1）")
    parser.add_argument("--reg-metric",   choices=["rmse", "r2", "mae"], default="rmse",
                        help="回歸優化指標（預設 rmse）")

    # 批次模式旗標
    parser.add_argument("--batch",        action="store_true",
                        help="批次模式：評估 openml_dir 前 top-n 個資料集")
    parser.add_argument("--openml-dir",   default="openml_cc18_data",
                        help="OpenML CSV 目錄（批次模式用）")
    parser.add_argument("--top-n",        type=int, default=5,
                        help="取前 N 個資料集（批次模式用）")
    parser.add_argument("--last",         action="store_true",
                        help="取最後 top-n 個（批次模式用）")
    parser.add_argument("--reg-dir",      default="openml_regression_data",
                        help="非時序回歸 CSV 目錄（批次模式用，預設 openml_regression_data）")
    parser.add_argument("--reg-top-n",    type=int, default=5,
                        help="回歸目錄取前 N 個資料集（批次模式用）")

    # 單一 CSV / 預切分模式旗標
    parser.add_argument("--csv",          nargs='+', type=str, default=None,
                        help="單一或多個 CSV 檔案路徑（多檔將自動 Join，例如：--csv train.csv identity.csv）")
    parser.add_argument("--train",        nargs='+', type=str, default=None,
                        help="訓練集 CSV 路徑（支援多檔輸入）")
    parser.add_argument("--test",         nargs='+', type=str, default=None,
                        help="測試集 CSV 路徑（支援多檔輸入）")
    parser.add_argument("--target",       type=str, default=None, # 建議給個預設值
                        help="目標欄位名稱")
    parser.add_argument("--result-file",  default=None,
                        help="附加結果 CSV（含 source 欄，附加模式）")

    args = parser.parse_args()

    if args.train and args.test:
        run_presplit(args)
    elif args.csv:
        run_single(args)
    elif args.batch:
        run_batch(args)
    else:
        parser.print_help()
        print("\n[提示] 請指定 --csv <path>、--train/--test 或 --batch 來執行 Pipeline。")
        print("[提示] 時序資料（UCR）請改用 run_pipeline_time.py")


if __name__ == "__main__":
    main()