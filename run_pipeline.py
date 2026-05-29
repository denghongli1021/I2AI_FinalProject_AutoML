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

<<<<<<< HEAD
=======
import joblib # 💾 用來儲存預處理引擎大腦
from sklearn.pipeline import Pipeline

>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.config import DEVICE, SEED, ARTIFACTS_DIR
<<<<<<< HEAD
from src import pipeline as _pl
from src import pipeline_time as _pt
=======
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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e

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
<<<<<<< HEAD
    """批次模式專用：只做精確名稱比對，找不到則報錯（不做啟發式 fallback）。"""
    for cand in ("target", "label", "class", "y", "c"):
        if cand in df.columns:
            return cand
    raise ValueError(
        f"找不到標準目標欄位（target/label/class/y/c）。"
        f"可用欄位：{list(df.columns)}。"
        f"請以 --target 明確指定目標欄位名稱。"
    )


def _prepare_X(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """選取特徵欄並轉為 float32；全為 object 欄時自動 OrdinalEncode。"""
    from sklearn.preprocessing import OrdinalEncoder
    feat_df = df.drop(columns=[target_col])
    X_num = feat_df.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        obj_df = feat_df.select_dtypes(include=["object", "category"])
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_num = pd.DataFrame(
            enc.fit_transform(obj_df.fillna("__missing__")),
            columns=obj_df.columns,
        )
    return X_num.fillna(0).values.astype(np.float32)
=======
    for cand in ("target", "label", "class", "y", "c", "isFraud"):
        if cand in df.columns:
            return cand
    return df.columns[-1]
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e


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


<<<<<<< HEAD
def _get_feature_names(df: pd.DataFrame, target_col: str) -> list:
    """Return numeric column names that _prepare_X will use."""
    feat_df = df.drop(columns=[target_col])
    X_num = feat_df.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        return list(feat_df.select_dtypes(include=["object", "category"]).columns)
    return list(X_num.columns)


def _build_shap_col_names(feature_set: str, orig_names: list, n_transformed: int) -> list:
    """Map transformed feature dimensions to human-readable names."""
    n = len(orig_names)
    if feature_set == "raw":
        return orig_names[:n_transformed]
    if feature_set == "signal":
        base = orig_names[:n]
        return (base + [f"{nm}_l2" for nm in base])[:n_transformed]
    if feature_set in ("pca64", "svd64", "kpca32"):
        return [f"PC{i}" for i in range(n_transformed)]
    # raw_stat / raw_stat_fft / poly2 / ts_tabular — prefix with original names then numbered
    if n_transformed >= n:
        return orig_names + [f"feat_{i}" for i in range(n_transformed - n)]
    return [f"feat_{i}" for i in range(n_transformed)]


# def _smart_prepare(df: pd.DataFrame, target_col: str):
#     """
#     使用 preprocessing 模組進行智慧前處理。
#     AutoRouter 自動偵測 numeric / categorical / text / datetime 欄位，
#     建立雙軌管線（tree 軌道保留原始尺度供樹狀模型使用）。
#     回傳 (X_tr, X_te, y_tr_raw, y_te_raw, feature_names)，X 為 float32 numpy。
#     若 preprocessing 模組不可用，自動回退到基本的 _prepare_X() + train_test_split。
#     """
#     try:
#         from preprocessing.interface import preprocess_for_training
        
#         # 1. 取得雙軌資料字典 (X_dict_tr 裡面包含 "tree" 和 "dl" 的 DataFrame)
#         X_dict_tr, X_dict_te, y_tr_raw, y_te_raw, _ = preprocess_for_training(
#             data_source=df,
#             target_col=target_col,
#             test_size=0.2,
#         )
        
#         # 2. 建立特徵名稱字典 (下游模型在做特徵重要性分析時會用到)
#         feature_names_dict = {
#             "tree": list(X_dict_tr["tree"].columns),
#             "dl": list(X_dict_tr["dl"].columns)
#         }
        
#         # 3. 🚀 直接回傳字典
#         return X_dict_tr, X_dict_te, y_tr_raw, y_te_raw, feature_names_dict
#     except Exception as _e:
#         print(f"  [Preprocess] 智慧前處理失敗（{_e}），回退到基本前處理")
#         X_all = _prepare_X(df, target_col)
#         feat_names = _get_feature_names(df, target_col)
#         y_all_raw = df[target_col]
#         X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(
#             X_all, y_all_raw, test_size=0.2, random_state=SEED
#         )
#         return X_tr, X_te, y_tr_raw, y_te_raw, feat_names
def _smart_prepare(
    train_df: pd.DataFrame, 
    target_col: str, 
    test_df: pd.DataFrame = None,  
    enable_adv_val: bool = True
):
    """
    [Pipeline Wrapper] 使用 preprocessing 模組進行智慧前處理。
    全自動處理對抗驗證、雙軌制分流，並為 HPO 準備好驗證集。
    """
    print("\n" + "="*50)
    print("🚀 [Pipeline] 啟動智慧前處理 (Smart Prepare)...")
    print("="*50)
    
    try:
        from preprocessing.interface import preprocess_for_training
        from sklearn.model_selection import train_test_split
        
        # ==========================================
        # ⚙️ 呼叫底層 interface.py (一鍵包辦清理、對抗驗證、雙軌產出)
        # ==========================================
        # 我們直接把 test_df 傳進去，interface 就會自動做對抗驗證！
        X_out_1, X_out_2, y_1, y_2, preprocessors = preprocess_for_training(
            data_source=train_df,
            target_col=target_col,
            test_data_source=test_df, 
            test_size=0.2,
            enable_adv_val=enable_adv_val
        )
        
        # ==========================================
        # ✂️ 為下游 HPO (超參數優化) 準備 80/20 驗證集
        # ==========================================
        if test_df is not None:
            # 情況 A：有給 test_df，interface 回傳的是 [100% Train] 與 [100% Test]
            # 為了讓 HPO 能調參，我們要把 100% Train (X_out_1) 自己切成 80/20
            print("[Pipeline] 正在為 HPO 訓練切分 80/20 驗證集...")
            
            X_dict_tr, X_dict_val = {}, {}
            stratify = y_1 if (y_1.nunique() <= 20 and y_1.nunique() >= 2) else None
            
            # 使用索引切割，確保 tree 和 dl 兩個軌道的資料是對齊的
            idx_tr, idx_val, y_tr_raw, y_val_raw = train_test_split(
                X_out_1["tree"].index, y_1, test_size=0.2, random_state=42, stratify=stratify
            )
            
            for track in ["tree", "dl"]:
                X_dict_tr[track] = X_out_1[track].loc[idx_tr].reset_index(drop=True)
                X_dict_val[track] = X_out_1[track].loc[idx_val].reset_index(drop=True)
                
            y_tr_raw = y_tr_raw.reset_index(drop=True)
            y_val_raw = y_val_raw.reset_index(drop=True)
            
            # 💡 X_out_2 這裡就是已經處理好的 kaggle test_df (雙軌字典)，
            # 如果你的 Pipeline 最後會直接預測，其實可以直接把它存起來備用！

        else:
            # 情況 B：沒有給 test_df，interface 已經幫我們切好 80/20 了
            X_dict_tr, X_dict_val = X_out_1, X_out_2
            y_tr_raw, y_val_raw = y_1, y_2

        # ==========================================
        # 📦 封裝與回傳
        # ==========================================
        feature_names_dict = {
            "tree": list(X_dict_tr["tree"].columns),
            "dl": list(X_dict_tr["dl"].columns)
        }
        
        print("✅ [Pipeline] 雙軌智慧前處理成功！準備進入 HPO/NAS 分流...")
        
        # 🚀 強烈建議：把 preprocessors 也回傳，這樣你的 Pipeline 才能把它存成 .pkl 給推論期使用！
        return X_dict_tr, X_dict_val, y_tr_raw, y_val_raw, feature_names_dict, preprocessors
        
    except Exception as _e:
        import traceback
        traceback.print_exc() # 印出完整的錯誤 Traceback 方便除錯
        print(f"❌ [Pipeline] 智慧前處理失敗（{_e}），回退到基本前處理...")
        
        # --------- 防呆回退機制 (Fallback) ---------
        # 如果你原本有 _prepare_X 函式，就在這裡保留你的 fallback 邏輯
        # 為了避免 Pipeline 當機，回傳假雙軌
        try:
            X_all = _prepare_X(train_df, target_col) 
            feat_names = _get_feature_names(train_df, target_col)
            y_all_raw = train_df[target_col]

            X_tr, X_val, y_tr_raw, y_val_raw = train_test_split(
                X_all, y_all_raw, test_size=0.2, random_state=42
            )
            
            X_dict_tr = {"tree": X_tr, "dl": X_tr}
            X_dict_val = {"tree": X_val, "dl": X_val}
            feature_names_dict = {"tree": feat_names, "dl": feat_names}
            
            return X_dict_tr, X_dict_val, y_tr_raw, y_val_raw, feature_names_dict, None
        except:
            print("🚨 嚴重錯誤：Fallback 基本前處理也失敗了！")
            raise

def _run_shap_visualization(artifacts_dir: str, X_test_raw: np.ndarray,
                             feature_names: list, dataset_name: str = ""):
    """Load best tabular model + FeatureBuilder from artifacts_dir and generate SHAP plots."""
    try:
        import sys as _sys
        _viz_path = os.path.join(HERE, "visualization")
        if _viz_path not in _sys.path:
            _sys.path.insert(0, _viz_path)
        from visualizer import AutoMLVisualizer
        import joblib as _jl

        model_path = os.path.join(artifacts_dir, "best_tabular_model.pkl")
        fb_path    = os.path.join(artifacts_dir, "best_tabular_model_fb.pkl")

        if not os.path.exists(model_path):
            print("[Viz] best_tabular_model.pkl 不存在，跳過視覺化"); return
        if not os.path.exists(fb_path):
            print("[Viz] best_tabular_model_fb.pkl 不存在，跳過視覺化"); return

        model = _jl.load(model_path)
        fb    = _jl.load(fb_path)

        X_transformed = fb.transform(X_test_raw)
        col_names = _build_shap_col_names(fb.feature_set, feature_names, X_transformed.shape[1])

        X_df = pd.DataFrame(X_transformed, columns=col_names)
        viz_output = os.path.join(artifacts_dir, "shap_plots")
        viz = AutoMLVisualizer(model=model, X_test=X_df, output_dir=viz_output)

        shap_mat = viz._get_shap_matrix()
        top_feat = col_names[int(np.abs(shap_mat).mean(0).argmax())]
        viz.generate_all_plots(
            sample_index=0,
            target_feature=top_feat,
            prefix=dataset_name or "pipeline",
        )
        print(f"\n[Viz] SHAP 圖表已輸出 → {viz_output}")
    except Exception as _e:
        print(f"\n[Viz] 視覺化跳過（{_e}）")


# ── 批次評估模式 ──────────────────────────────────────────────────────────────
from preprocessing.data_loader import load_and_merge_data
def run_batch(args, datasets_override=None):
    """
    [共用核心] 執行資料集前處理與模型訓練。
    - 如果沒有 datasets_override，就是正常的 Batch 模式 (去掃描資料夾)。
    - 如果有 datasets_override，就是 Single 模式 (只跑指定的那一個檔案)。
    """
    # 判斷當前是單檔模式還是批次模式
    is_single_mode = datasets_override is not None
    datasets = []

    if is_single_mode:
        datasets = datasets_override
    else:
        openml_dir = os.path.join(HERE, getattr(args, "openml_dir", ""))
        reg_dir    = os.path.join(HERE, getattr(args, "reg_dir", ""))
        if os.path.isdir(openml_dir):
            datasets += _find_datasets(openml_dir, args.top_n, last=args.last)
        else:
            print(f"[警告] 找不到 OpenML 目錄：{openml_dir}")
        if os.path.isdir(reg_dir):
            datasets += _find_datasets(reg_dir, args.reg_top_n, last=args.last, force_task="regression")
        if not datasets:
            print("[錯誤] 無可用資料集"); sys.exit(1)

    mode_name = "單資料集" if is_single_mode else "批次"
    print(f"\n{'='*65}")
    print(f"  Pipeline {mode_name}模式（非時序）  ─  {len(datasets)} 個資料集  fast={args.fast}")
    print(f"{'='*65}")
    
    results = []
    out_file_name = "pipeline_single_results.csv" if is_single_mode else "pipeline_batch_results.csv"
    out_path = os.path.join(HERE, out_file_name)
=======
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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e

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
<<<<<<< HEAD
            # 🚀 統一使用 DataLoader 進行記憶體壓縮
            df = load_and_merge_data(csv_path)
            
            # 🚀 動態決定 target_col：單檔模式吃 args.target，批次模式自動偵測
            if is_single_mode and getattr(args, "target", None):
                target_col = args.target
                if target_col not in df.columns:
                    print(f"[錯誤] 找不到目標欄位 '{target_col}'，可用欄位：{list(df.columns)}")
                    sys.exit(1)
            else:
                target_col = _find_target_col(df)
                
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
            y_raw = df[target_col]
            task = force_task if force_task else _auto_detect_task(y_raw)
            
            # 呼叫你新版的智慧前處理
            X_dict_tr, X_dict_te, y_tr_raw, y_te_raw, feature_names_dict, preprocessors = _smart_prepare(df, target_col)
            
            # 區分單檔或批次的存放路徑
            artifacts_dir = os.path.join(ARTIFACTS_DIR, "single" if is_single_mode else "batch", dataset_name)

            if task == "classification":
                le = LabelEncoder()
                le.fit(y_raw.astype(str).values)
                y_tr = le.transform(y_tr_raw.astype(str).values)
                y_te = le.transform(y_te_raw.astype(str).values)
                n_classes = len(le.classes_)
                print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  split=Random Stratified")
=======
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
            
            # 直接將字典裡的 DataFrame 轉換為 Numpy Array，並保持字典格式
            X_tr = {
                "tree": X_tr_dict["tree"].values.astype(np.float32),
                "dl": X_tr_dict["dl"].values.astype(np.float32)
            }
            
            X_te = {
                "tree": X_te_dict["tree"].values.astype(np.float32),
                "dl": X_te_dict["dl"].values.astype(np.float32)
            }
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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e

                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
                cfg["is_timeseries"] = False
<<<<<<< HEAD
                
                result = _pl.run(
                    X_dict_tr, y_tr, X_dict_te, n_classes, cfg, budget,
=======
                result = _pl.run(
                    X_tr, y_tr, X_te, n_classes, cfg, budget,
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
                    skip_tabular=args.skip_tabular,
                    skip_dl=args.skip_dl,
                    no_nas=args.no_nas,
                    is_ts=False,
<<<<<<< HEAD
                    artifacts_dir=artifacts_dir,
=======
                    artifacts_dir=art_dir,
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
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
<<<<<<< HEAD
                
                if args.viz:
                    _run_shap_visualization(artifacts_dir, X_dict_te["tree"], feature_names_dict["tree"], dataset_name)
                    
=======
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
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
<<<<<<< HEAD
                print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression  split=Random")

                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
                
                result = _pt.run_regression(
                    X_dict_tr, y_tr, X_dict_te, cfg, budget,
                    skip_tabular=args.skip_tabular,
                    skip_dl=args.skip_dl,
                    artifacts_dir=artifacts_dir,
                    metric=args.reg_metric,
                )
                rmse_b, rmse_s, r2_b, r2_s, best_rmse, best_r2, primary_score = \
                    _eval_regression(y_te, result, args.reg_metric)
                elapsed = round(time.time() - t_ds, 1)
                
=======
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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
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
<<<<<<< HEAD
    print(f"  SUMMARY — Pipeline（非時序 {mode_name}模式）")
=======
    print("  BATCH SUMMARY — Pipeline（非時序）")
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
    print(f"{'='*65}")
    print(pd.DataFrame(results).to_string(index=False))
    _flush()
    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


<<<<<<< HEAD
# ── 單一 CSV 模式 (轉接器) ────────────────────────────────────────────────────────

def run_single(args):
    """
    接收單一 CSV 的參數，將其包裝成陣列後，直接轉接給 run_batch 處理。
    """
    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"[錯誤] 找不到檔案：{csv_path}"); sys.exit(1)
        
    if not getattr(args, "target", None):
        print("[錯誤] --csv 模式需明確指定 --target 目標欄位名稱。")
        sys.exit(1)

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # 🚀 將單一資料集包裝成 batch 引擎看得懂的格式 [(name, path, force_task)]
    # force_task 設為 None，讓引擎自己去判斷
    single_dataset_list = [(dataset_name, csv_path, None)]
    
    # 🚀 直接轉接！
    run_batch(args, datasets_override=single_dataset_list)

# ── 預切分模式 ────────────────────────────────────────────────────────────────

def run_presplit(args):
    """用戶手動提供 TRAIN / TEST 兩個 CSV，直接使用不再自行切分。"""
    train_path = os.path.abspath(args.train)
    test_path  = os.path.abspath(args.test)
    for p, label in [(train_path, "TRAIN"), (test_path, "TEST")]:
        if not os.path.isfile(p):
            print(f"[錯誤] 找不到{label}檔案：{p}"); sys.exit(1)

    base = os.path.splitext(os.path.basename(train_path))[0]
    dataset_name = base[:-6] if base.endswith("_TRAIN") else base

    print(f"\n{'='*65}")
    print(f"  Pipeline 預切分模式（非時序）  ─  {dataset_name}  |  device={DEVICE}")
    print(f"{'='*65}")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    if not args.target:
        print(f"[錯誤] --train/--test 模式需以 --target 明確指定目標欄位名稱。可用欄位：{list(train_df.columns)}")
        sys.exit(1)
    target_col = args.target
    if target_col not in train_df.columns:
        print(f"[錯誤] 找不到目標欄位 '{target_col}'，可用欄位：{list(train_df.columns)}")
        sys.exit(1)

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=[target_col]).reset_index(drop=True)

    y_tr_raw = train_df[target_col]
    y_te_raw = test_df[target_col]
    task = _auto_detect_task(y_tr_raw)
    feature_names = _get_feature_names(train_df, target_col)

    X_tr = _prepare_X(train_df, target_col)
    X_te = _prepare_X(test_df, target_col)
    min_cols = min(X_tr.shape[1], X_te.shape[1])
    X_tr = X_tr[:, :min_cols]
    X_te = X_te[:, :min_cols]

    t_ds = time.time()

=======
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
    print("  [AutoML 2.0] 呼叫 Data_loader 與雙軌預處理引擎...")
    
    # 這裡的變數名稱改成 dict 和 fitted_preprocessors(加s) 以符合雙軌制
    X_tr_dict, X_te_dict, y_tr_raw, y_te_raw, fitted_preprocessors = preprocess_for_training(
        data_source=csv_paths, 
        target_col=args.target,
        test_size=0.2
    )

    task = _auto_detect_task(y_tr_raw)
    
    # ==========================================
    # 🚀 雙軌制資料轉換 (將字典中的 DataFrame 分別轉為 float32)
    # ==========================================
    X_tr_tree = X_tr_dict["tree"].values.astype(np.float32)
    X_te_tree = X_te_dict["tree"].values.astype(np.float32)
    X_tr_dl = X_tr_dict["dl"].values.astype(np.float32)
    X_te_dl = X_te_dict["dl"].values.astype(np.float32)
    
    X_tr = {"tree": X_tr_tree, "dl": X_tr_dl}
    X_te = {"tree": X_te_tree, "dl": X_te_dl}
    # ==========================================

    # 💾 儲存預處理大腦 (注意變數名稱改為 fitted_preprocessors)
    art_dir = os.path.join(ARTIFACTS_DIR, "single", dataset_name)
    os.makedirs(art_dir, exist_ok=True)
    joblib.dump(fitted_preprocessors, os.path.join(art_dir, "fitted_preprocessors.pkl"))

>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
    if task == "classification":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_raw.astype(str).values)
        classes_set = set(le.classes_)
<<<<<<< HEAD
        y_te = np.array(
            [le.transform([str(v)])[0] if str(v) in classes_set else 0
             for v in y_te_raw], dtype=np.int64)
        n_classes = len(le.classes_)
        print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  task=classification")

        _presplit_artifacts = os.path.join(ARTIFACTS_DIR, "single", dataset_name)
=======
        y_te = np.array([le.transform([str(v)])[0] if str(v) in classes_set else 0 for v in y_te_raw], dtype=np.int64)
        n_classes = len(le.classes_)
        print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}")

>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False
        result = _pl.run(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            is_ts=False,
<<<<<<< HEAD
            artifacts_dir=_presplit_artifacts,
=======
            artifacts_dir=art_dir,
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
            metric=args.metric,
        )
        from src.metrics import calculate_score, get_metric_name
        score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
        score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → {get_metric_name(args.metric)}={score_b:.4f}")
        print(f"  [結果] Stack → {get_metric_name(args.metric)}={score_s:.4f}")
        print(f"  [耗時] {elapsed}s")
<<<<<<< HEAD
        if args.viz:
            _run_shap_visualization(_presplit_artifacts, X_te, feature_names, dataset_name)
=======

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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e

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
<<<<<<< HEAD
            artifacts_dir=os.path.join(ARTIFACTS_DIR, "single", dataset_name),
=======
            artifacts_dir=art_dir,
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
            metric=args.reg_metric,
        )
        rmse_b, rmse_s, r2_b, r2_s, _, _, _ = _eval_regression(y_te, result, args.reg_metric)
        elapsed = round(time.time() - t_ds, 1)
<<<<<<< HEAD
        print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
        print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}")
        print(f"  [耗時] {elapsed}s")

=======
        
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

>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
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
<<<<<<< HEAD
    parser.add_argument("--csv",          type=str, default=None,
                        help="單一 CSV 檔案路徑（80/20 random split）")
    parser.add_argument("--train",        type=str, default=None,
                        help="訓練集 CSV 路徑（搭配 --test 使用預切分模式）")
    parser.add_argument("--test",         type=str, default=None,
                        help="測試集 CSV 路徑（搭配 --train 使用預切分模式）")
    parser.add_argument("--target",       type=str, default=None,
                        help="目標欄位名稱（--csv / --train/--test 模式必填）")
    parser.add_argument("--result-file",  default=None,
                        help="附加結果 CSV（含 source 欄，附加模式）")
    parser.add_argument("--viz",          action="store_true",
                        help="訓練完成後自動產生 SHAP 視覺化圖表（需要 shap + plotly + kaleido）")
=======
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
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e

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
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
