"""
run_pipeline_time.py — 時序專用 Pipeline 入口（AutoML 2.0 換心升級版）

支援 AutoML 2.0 雙引擎 (Data Loader + 煉金管線)，
具備記憶體壓縮、MI 特徵降維、自動產出 Submission 預測檔，並支援 Kaggle 盲測。

用法：
    # 單一 TRAIN CSV（自動尋找對應 TEST CSV）支援多檔 Join
    python run_pipeline_time.py --csv train_ts.csv --target label

    # 預切分模式 (支援多檔)
    python run_pipeline_time.py --train train.csv --test test.csv --target label
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

# 🚀 匯入 AutoML 2.0 底層 Data Loader
try:
    from preprocessing.interface import load_and_merge_data
except ImportError:
    print("[警告] 找不到 load_and_merge_data，請確認 preprocessing/interface.py 路徑正確")
    sys.exit(1)


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _find_target_col(df: pd.DataFrame) -> str:
    for c in ("target", "label", "class", "y", "c", "target_feature", "isFraud"):
        if c in df.columns:
            return c
    return df.columns[-1]

def _auto_detect_task(filename: str, y: pd.Series) -> str:
    if os.path.basename(filename).startswith("REG_"):
        return "regression"
    if y.dtype == object or y.dtype == bool:
        return "classification"
    n_unique = y.nunique()
    return "classification" if (n_unique <= 50 and n_unique / len(y) < 0.30) else "regression"


# ── 🚀 AutoML 2.0 預處理引擎橋接器 ──────────────────────────────────────────

def _apply_automl_2(X_tr_raw: pd.DataFrame, y_tr: pd.Series, X_te_raw: pd.DataFrame):
    """通用 AutoML 2.0 預處理掛載器：執行 Phase 0 -> 1 -> 2"""
    print("\n  [AutoML 2.0] 啟動時序專屬煉金引擎 (記憶體壓縮 & 特徵降維)...")
    try:
        from .preprocessing.processors.feature_generator import RobustDataCleaner, MIFeatureSelector
        from .preprocessing.core.router import AutoRouter 
        from .preprocessing.core.assembler import PipelineAssembler
        from sklearn.pipeline import Pipeline
        
        # 1. 探路者 (AutoRouter)
        print("    ↳ [AutoRouter] 正在掃描時序特徵欄位...")
        router = AutoRouter()
        feature_groups = router.fit_transform(X_tr_raw)

        # 2. 組裝廠 (Assembler)
        assembler = PipelineAssembler(feature_groups)
        is_classification = (y_tr.nunique() <= 50)

        # 3. 封裝終極大腦
        fitted_preprocessor = Pipeline([
            ('phase0', RobustDataCleaner()),
            ('phase1', assembler.build()),
            ('phase2_mi_selector', MIFeatureSelector(top_k=300, is_classification=is_classification))
        ])

        # 4. 擬合與轉換
        print("    ↳ [預處理] 正在擬合與轉換訓練集 (Fit_Transform) ...")
        X_tr_clean = fitted_preprocessor.fit_transform(X_tr_raw, y_tr)
        
        print("    ↳ [預處理] 正在轉換測試集 (Transform) ...")
        X_te_clean = fitted_preprocessor.transform(X_te_raw)

        # 轉 float32 節省記憶體並相容後續神經網路
        X_tr_final = X_tr_clean.astype(np.float32)
        X_te_final = X_te_clean.astype(np.float32)
        print(f"    ↳ ✅ [完成] 訓練集維度: {X_tr_final.shape}, 測試集維度: {X_te_final.shape}")

        return X_tr_final, X_te_final, fitted_preprocessor

    except ImportError as e:
        print(f"  [警告] AutoML 2.0 模組匯入失敗 ({e})，退回暴力補 0 模式！")
        X_tr = X_tr_raw.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        X_te = X_te_raw.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        return X_tr, X_te, None


# ── 核心執行邏輯 (單一/預切分) ──────────────────────────────────────────────

def _process_new_ts_one(base_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, args, t_ds: float) -> dict:
    """執行 Pipeline 於 Train/Test 資料夾對，並整合 AutoML 2.0 與 CSV 輸出"""
    task = "regression" if base_name.startswith("REG_") else "classification"
    target_col = args.target if args.target else _find_target_col(train_df)

    if target_col not in train_df.columns:
        raise ValueError(f"找不到目標欄 '{target_col}'")

    # 處理 Train
    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    y_tr_raw = train_df[target_col]
    X_tr_raw = train_df.drop(columns=[target_col])

    # 處理 Test (支援 Kaggle 盲測)
    has_test_labels = target_col in test_df.columns
    if has_test_labels:
        test_df = test_df.dropna(subset=[target_col]).reset_index(drop=True)
        y_te_raw = test_df[target_col]
        X_te_raw = test_df.drop(columns=[target_col])
    else:
        print("  [提示] 偵測到測試集無目標欄位，進入「盲測生成模式 (Inference)」")
        y_te_raw = pd.Series(np.zeros(len(test_df)))
        X_te_raw = test_df.copy()

    # 🚀 呼叫 AutoML 2.0 進行時序特徵清洗與降維
    X_tr, X_te, preprocessor = _apply_automl_2(X_tr_raw, y_tr_raw, X_te_raw)

    if X_tr.shape[1] == 0:
        raise ValueError("經過特徵過濾後無可用特徵！")

    # ── 模型訓練與預測 ──────────────────────────────────────────────────────
    art_dir = os.path.join(ARTIFACTS_DIR, "ts", base_name)
    budget = _pt.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)

    if task == "classification":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_raw.astype(str).values)
        classes_set = set(le.classes_)
        y_te = np.array([le.transform([str(v)])[0] if str(v) in classes_set else 0 for v in y_te_raw], dtype=np.int64)
        n_classes = len(le.classes_)
        print(f"  [Task] Classification  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}")

        import pipeline as _pl
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False # UCR 分類視為樣本獨立

        result = _pt.run_classification(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular, skip_dl=args.skip_dl, no_nas=args.no_nas,
            artifacts_dir=art_dir, metric=args.cls_metric
        )

        from src.metrics import calculate_score
        elapsed = round(time.time() - t_ds, 1)
        if has_test_labels:
            score_b = calculate_score(y_te, result.test_blend, metric=args.cls_metric)
            score_s = calculate_score(y_te, result.test_stack, metric=args.cls_metric)
            print(f"\n  [結果] Blend={score_b:.4f}  Stack={score_s:.4f}  ({elapsed}s)")
        else:
            score_b, score_s = 0.0, 0.0
            print(f"\n  [結果] 盲測模式完成 ({elapsed}s)")

        # 🚀 產生 Submission CSV (分類)
        print("  🏆 產生最終預測結果 (Submission)")
        if len(result.test_stack.shape) > 1 and result.test_stack.shape[1] > 1:
            best_preds = np.argmax(result.test_stack if score_s >= score_b else result.test_blend, axis=1)
        else:
            best_preds = result.test_stack if score_s >= score_b else result.test_blend
        
        id_col = 'id' if 'id' in test_df.columns else ('ID' if 'ID' in test_df.columns else test_df.columns[0])
        test_ids = test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))
        
        from src.make_submission import generate_submission
        try:
            generate_submission(
                preds=best_preds, test_ids=test_ids, label_encoder=le,
                out_name=f"submission_ts_{base_name}.csv", target_col=target_col, id_col=id_col
            )
        except Exception as e:
            print(f"⚠️ CSV 生成錯誤: {e}")

        return {"dataset": base_name, "task": task, "score": max(score_b, score_s)}

    else:
        # 回歸任務
        y_tr = np.asarray(y_tr_raw.values, dtype=np.float32).ravel()
        y_te = np.asarray(y_te_raw.values, dtype=np.float32).ravel()
        print(f"  [Task] Regression  n_train={len(y_tr)}  n_test={len(y_te)}")

        cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
        result = _pt.run_regression(
            X_tr, y_tr, X_te, cfg, budget,
            skip_tabular=args.skip_tabular, skip_dl=args.skip_dl,
            artifacts_dir=art_dir, metric=args.reg_metric
        )

        elapsed = round(time.time() - t_ds, 1)
        if has_test_labels:
            from sklearn.metrics import mean_squared_error, r2_score
            rmse_b = float(np.sqrt(mean_squared_error(y_te, result.test_blend)))
            rmse_s = float(np.sqrt(mean_squared_error(y_te, result.test_stack)))
            print(f"\n  [結果] Blend RMSE={rmse_b:.4f}  Stack RMSE={rmse_s:.4f}  ({elapsed}s)")
        else:
            rmse_b, rmse_s = 0.0, 0.0
            print(f"\n  [結果] 盲測模式完成 ({elapsed}s)")

        # 🚀 產生 Submission CSV (回歸)
        print("  🏆 產生最終預測結果 (Submission - Regression)")
        best_preds = (result.test_stack if rmse_s <= rmse_b else result.test_blend).ravel()
        id_col = 'id' if 'id' in test_df.columns else ('ID' if 'ID' in test_df.columns else test_df.columns[0])
        test_ids = test_df[id_col].values if id_col in test_df.columns else np.arange(len(test_df))
        
        os.makedirs("submissions", exist_ok=True)
        out_path = os.path.join("submissions", f"submission_ts_{base_name}.csv")
        pd.DataFrame({id_col: test_ids, target_col: best_preds}).to_csv(out_path, index=False)
        print(f"  [Submit] Saved → {out_path}")

        return {"dataset": base_name, "task": task, "score": min(rmse_b, rmse_s)}


# ── 控制器：預切分模式 ──────────────────────────────────────────────────────

def run_presplit(args):
    """用戶手動提供 TRAIN / TEST，並支援 nargs='+' 多檔載入"""
    train_paths = args.train
    test_paths  = args.test

    for p in train_paths + test_paths:
        if not os.path.isfile(p):
            print(f"[錯誤] 找不到檔案：{p}"); sys.exit(1)

    base = os.path.splitext(os.path.basename(train_paths[0]))[0]
    base_name = base[:-6] if base.endswith("_TRAIN") else base

    print(f"\n{'='*65}")
    print(f"  Pipeline-Time 預切分模式  ─  {base_name}  |  device={DEVICE}")
    print(f"{'='*65}")
    t_ds = time.time()
    try:
        # 🚀 使用你們強大的多檔載入器
        print("  [Data Loader] 正在讀取並智慧合併 Train 資料...")
        train_df = load_and_merge_data(train_paths)
        print("  [Data Loader] 正在讀取並智慧合併 Test 資料...")
        test_df  = load_and_merge_data(test_paths)
        
        _process_new_ts_one(base_name, train_df, test_df, args, t_ds)
    except Exception:
        traceback.print_exc()


# ── 控制器：單一 CSV 模式 ────────────────────────────────────────────────────

def run_single(args):
    # args.csv 現在是一個 List (支援多檔)
    csv_paths = args.csv 
    for p in csv_paths:
        if not os.path.isfile(p):
            print(f"[錯誤] 找不到檔案：{p}"); sys.exit(1)

    # 若第一個檔案名稱有 _TRAIN，自動尋找對應的 _TEST (單檔配對)
    main_file = csv_paths[0]
    if main_file.endswith("_TRAIN.csv") and len(csv_paths) == 1:
        test_path = main_file[:-10] + "_TEST.csv"
        if os.path.isfile(test_path):
            args.train = [main_file]
            args.test = [test_path]
            return run_presplit(args)

    base_name = os.path.splitext(os.path.basename(main_file))[0]
    print(f"\n{'='*65}")
    print(f"  Pipeline-Time 單檔模式 (自動切分)  ─  {base_name}  |  device={DEVICE}")
    print(f"{'='*65}")
    t_ds = time.time()
    
    try:
        # 讀取並合併
        df = load_and_merge_data(csv_paths)
        
        target_col = args.target if args.target else _find_target_col(df)
        task = _auto_detect_task(main_file, df[target_col])
        
        # 進行時間序列嚴格切割 (80/20) 或分層切割
        if task == "regression":
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
        else:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[target_col])
            
        _process_new_ts_one(base_name, train_df, test_df, args, t_ds)
    except Exception:
        traceback.print_exc()


# ── CLI 入口 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="時序專用 Pipeline 入口（AutoML 2.0 版）")
    
    # 效能與模型開關
    parser.add_argument("--fast", action="store_true", help="縮減 HPO 次數")
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--no-nas", action="store_true", help="跳過 NAS")
    parser.add_argument("--time-limit", type=float, default=0, help="總時間上限（秒）")
    
    # 指標
    parser.add_argument("--cls-metric", choices=["f1", "accuracy"], default="f1")
    parser.add_argument("--reg-metric", choices=["rmse", "r2", "mae"], default="rmse")

    # 🚀 修改為 nargs='+'，全面支援多檔載入 (Kaggle 多表 Join)
    parser.add_argument("--csv",   nargs='+', default=None, help="單一或多個 CSV 路徑")
    parser.add_argument("--train", nargs='+', default=None, help="訓練集 CSV 路徑")
    parser.add_argument("--test",  nargs='+', default=None, help="測試集 CSV 路徑")
    parser.add_argument("--target", type=str, default=None, help="目標欄位名稱")

    args = parser.parse_args()
    
    if args.train and args.test:
        run_presplit(args)
    elif args.csv:
        run_single(args)
    else:
        parser.print_help()
        print("\n[提示] 請指定 --csv 或 --train/--test")

if __name__ == "__main__":
    main()