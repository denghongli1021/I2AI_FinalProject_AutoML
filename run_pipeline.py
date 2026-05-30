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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.config import DEVICE, SEED, ARTIFACTS_DIR
from src import pipeline as _pl
from src import pipeline_time as _pt

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

    # 🛡️ 防禦 1: 強制關閉 Pandas 輸出模式
    # 避免 OrdinalEncoder 輸出 DataFrame 導致 Cython 無法讀取而當機
    import sklearn
    sklearn.set_config(transform_output="default")

    # 🛡️ 防禦 2: 深度淨化資料型態 (Deep Sanitize)
    # 剝除所有會讓 Cython 崩潰的 PyArrow 與 Pandas Extension Types (Int8, UInt8 等)
    def _sanitize_df(df):
        if df is None: return None
        df = df.copy()
        for col in df.columns:
            dtype_str = str(df[col].dtype).lower()
            
            # 1. 處理字串與類別 (Category, string[pyarrow], string) -> 轉回標準 Python object
            if isinstance(df[col].dtype, pd.CategoricalDtype) or "string" in dtype_str:
                df[col] = df[col].astype(object)
                
            # 2. 處理 Pandas Nullable 數值 (Int8, UInt8, Float32 等) -> 轉為 Float64
            elif pd.api.types.is_extension_array_dtype(df[col]):
                # 再次確認它真的是數值，避免把奇怪的擴充型態誤轉
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].astype(np.float64)
                else:
                    df[col] = df[col].astype(object)
                    
        return df

    print("[防爆裝甲] 正在淨化訓練集資料型態，保護 Scikit-Learn 底層引擎...")
    train_df = _sanitize_df(train_df)
    if test_df is not None:
        test_df = _sanitize_df(test_df)
    
    # 🛡️ 防禦 3: 確保 Target 是最乾淨的 Numpy 型態
    if pd.api.types.is_numeric_dtype(train_df[target_col]):
        train_df[target_col] = train_df[target_col].astype(np.float64)
    else:
        train_df[target_col] = train_df[target_col].astype(str)
    
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
# ── 批次評估模式 ──────────────────────────────────────────────────────────────
from preprocessing.data_loader import load_and_merge_data

def run_batch(args, datasets_override=None):
    """
    [共用核心] 執行資料集前處理與模型訓練。
    - 如果沒有 datasets_override，就是正常的 Batch 模式 (去掃描資料夾)。
    - 如果有 datasets_override，就是 Single 模式 (只跑指定的那一個檔案)。
    """
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
            # 🚀 統一使用 DataLoader 進行記憶體壓縮 (支援字串或列表)
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
            
            artifacts_dir = os.path.join(ARTIFACTS_DIR, "single" if is_single_mode else "batch", dataset_name)

            if task == "classification":
                le = LabelEncoder()
                le.fit(y_raw.astype(str).values)
                y_tr = le.transform(y_tr_raw.astype(str).values)
                y_te = le.transform(y_te_raw.astype(str).values)
                n_classes = len(le.classes_)
                print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  split=Random Stratified")

                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
                cfg["is_timeseries"] = False
                
                # 分類引擎有 Router，可以直接傳字典
                result = _pl.run(
                    X_dict_tr, y_tr, X_dict_te, n_classes, cfg, budget,
                    skip_tabular=args.skip_tabular, skip_dl=args.skip_dl,
                    no_nas=args.no_nas, is_ts=False, artifacts_dir=artifacts_dir, metric=args.metric,
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
                
                if args.viz:
                    # SHAP 統一使用 tree 軌道畫圖
                    _run_shap_visualization(artifacts_dir, X_dict_te.get("tree", X_dict_te), feature_names_dict.get("tree", []), dataset_name)
                    
                results.append({
                    "dataset": dataset_name, "type": "Tab", "task": task,
                    "n_train": len(y_tr), "n_test": len(y_te),
                    "accuracy": acc, "f1_macro": f1, "rmse": None, "r2": None,
                    "score": best_score, "elapsed_s": elapsed,
                })
                

            else:  # regression
                y_tr = np.asarray(y_tr_raw.values, dtype=np.float32).ravel()
                y_te = np.asarray(y_te_raw.values, dtype=np.float32).ravel()
                print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression  split=Random")

                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
                
                # 🛡️ 迴歸引擎專用防呆：因為 pipeline_time 可能沒有 router，我們把 tree 拆出來傳
                X_tr_pass = X_dict_tr["tree"] if isinstance(X_dict_tr, dict) else X_dict_tr
                X_te_pass = X_dict_te["tree"] if isinstance(X_dict_te, dict) else X_dict_te
                
                # 將神經網路需要的資料藏在 cfg 裡面帶進去
                if isinstance(X_dict_tr, dict):
                    cfg["X_train_dl"] = X_dict_tr.get("dl", X_tr_pass)
                    cfg["X_test_dl"]  = X_dict_te.get("dl", X_te_pass)
                
                result = _pt.run_regression(
                    X_tr_pass, y_tr, X_te_pass, cfg, budget,
                    skip_tabular=args.skip_tabular, skip_dl=args.skip_dl,
                    artifacts_dir=artifacts_dir, metric=args.reg_metric,
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
                "accuracy": None, "f1_macro": None, "rmse": None, "r2": None,
                "score": None, "elapsed_s": round(time.time() - t_ds, 1),
            })
            _flush()
            _flush_result()

    print(f"\n{'='*65}")
    print(f"  SUMMARY — Pipeline（非時序 {mode_name}模式）")
    print(f"{'='*65}")
    print(pd.DataFrame(results).to_string(index=False))
    _flush()
    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


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
    single_dataset_list = [(dataset_name, csv_path, None)]
    run_batch(args, datasets_override=single_dataset_list)

# ── 預切分模式 (支援多檔列表 + 雙軌前處理) ────────────────────────────────────────────────

def run_presplit(args):
    """用戶手動提供 TRAIN / TEST 檔案 (支援清單)，直接進入雙軌前處理並訓練。"""
    
    # 1. 處理訓練集與測試集路徑 (支援 args.train 為列表)
    train_paths = [os.path.abspath(p) for p in args.train] if isinstance(args.train, list) else [os.path.abspath(args.train)]
    test_path = os.path.abspath(args.test) if args.test else None
    
    # 2. 檢查檔案是否存在
    for p in train_paths + ([test_path] if test_path else []):
        if not p or not os.path.isfile(p):
            print(f"[錯誤] 找不到檔案：{p}"); sys.exit(1)

    base = os.path.splitext(os.path.basename(train_paths[0]))[0]
    dataset_name = base[:-6] if base.endswith("_TRAIN") else base

    print(f"\n{'='*65}")
    print(f"  Pipeline 預切分模式（非時序）  ─  {dataset_name}  |  device={DEVICE}")
    print(f"{'='*65}")

    # 3. 使用 DataLoader 自動壓縮與合併
    train_df = load_and_merge_data(train_paths)

    if test_path:
        # 將 test.csv 當主表，並把 train_paths 裡的 4 個副表接在後面一起送去合併
        test_paths = [test_path] + train_paths[1:]
        test_df = load_and_merge_data(test_paths)

    if not getattr(args, "target", None) or args.target not in train_df.columns:
        print(f"[錯誤] --train/--test 模式需明確指定 --target，可用欄位：{list(train_df.columns)}")
        sys.exit(1)
    target_col = args.target

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    if test_df is not None and target_col in test_df.columns:
        test_df = test_df.dropna(subset=[target_col]).reset_index(drop=True)
    
    y_tr_raw = train_df[target_col]
    task = _auto_detect_task(y_tr_raw)

    # 🚀 4. 呼叫新版雙軌智慧前處理！(傳入 test_df 會自動啟動對抗驗證)
    X_dict_tr, X_dict_te, y_tr_raw, y_te_raw, feature_names_dict, _ = _smart_prepare(
        train_df, target_col, test_df=test_df
    )

    t_ds = time.time()
    artifacts_dir = os.path.join(ARTIFACTS_DIR, "single", dataset_name)

    # 5. 分類或迴歸任務分流
    if task == "classification":
        le = LabelEncoder()
        le.fit(y_tr_raw.astype(str).values)
        y_tr = le.transform(y_tr_raw.astype(str).values)
        
        # 安全轉換測試集標籤
        classes_set = set(le.classes_)
        y_te = np.array([le.transform([str(v)])[0] if str(v) in classes_set else 0 for v in y_te_raw], dtype=np.int64)
        n_classes = len(le.classes_)
        print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  task=classification")

        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False
        
        result = _pl.run(
            X_dict_tr, y_tr, X_dict_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular, skip_dl=args.skip_dl,
            no_nas=args.no_nas, is_ts=False, artifacts_dir=artifacts_dir, metric=args.metric,
        )
        from src.metrics import calculate_score, get_metric_name
        score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
        score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → {get_metric_name(args.metric)}={score_b:.4f}")
        print(f"  [結果] Stack → {get_metric_name(args.metric)}={score_s:.4f}")
        print(f"  [耗時] {elapsed}s")
        if args.viz:
            _run_shap_visualization(artifacts_dir, X_dict_te.get("tree", X_dict_te), feature_names_dict.get("tree", []), dataset_name)

    else:  # regression
        y_tr = np.asarray(y_tr_raw.values, dtype=np.float32).ravel()
        y_te = np.asarray(y_te_raw.values, dtype=np.float32).ravel()
        print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression")

        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
        
        # 🛡️ 迴歸防呆處理
        X_tr_pass = X_dict_tr["tree"] if isinstance(X_dict_tr, dict) else X_dict_tr
        X_te_pass = X_dict_te["tree"] if isinstance(X_dict_te, dict) else X_dict_te
        if isinstance(X_dict_tr, dict):
            cfg["X_train_dl"] = X_dict_tr.get("dl", X_tr_pass)
            cfg["X_test_dl"]  = X_dict_te.get("dl", X_te_pass)

        result = _pt.run_regression(
            X_tr_pass, y_tr, X_te_pass, cfg, budget,
            skip_tabular=args.skip_tabular, skip_dl=args.skip_dl,
            artifacts_dir=artifacts_dir, metric=args.reg_metric,
        )
        rmse_b, rmse_s, r2_b, r2_s, _, _, _ = _eval_regression(y_te, result, args.reg_metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
        print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}")
        print(f"  [耗時] {elapsed}s")


    # ==========================================
    # 🚀 在 run_presplit 的最後，加入 Kaggle 提交檔強制輸出模組
    # ==========================================
    print("\n🚀 正在生成 Kaggle 專用提交檔 (submission.csv)...")
    try:
        # 1. 讀取原始 test.csv 拿 ID
        test_raw = pd.read_csv(args.test) 
        
        # 2. 獲取預測結果
        if hasattr(_pl, "predict_proba"):
            test_proba = _pl.predict_proba(X_dict_te)
        elif hasattr(result, "predict_proba"):
            test_proba = result.predict_proba(X_dict_te)
        else:
            # 備案：如果沒有提供機率介面，直接拿 Blend 的預測結果
            test_proba = getattr(result, "test_blend_proba", result.test_blend)
        
        # 🛡️ 關鍵防呆裝甲：如果 test_proba 只有 1 維（代表拿到的是類別 0, 1, 2）
        if len(test_proba.shape) == 1:
            print("  ⚠️ 提示：捕捉到 1D 硬標籤，自動轉換為格式要求的 One-Hot 機率矩陣。")
            n_samples = len(test_proba)
            # 建立一個形狀為 (N, 3) 的全零矩陣
            one_hot_proba = np.zeros((n_samples, 3))
            # 將預測類別對應的位置設為 1.0
            for i, p_class in enumerate(test_proba):
                one_hot_proba[i, int(p_class)] = 1.0
            test_proba = one_hot_proba  # 成功升級為 2D 機率矩陣！
        
        # 3. 如果是分類任務，組裝成 Telstra 比賽格式
        if task == "classification":
            sub = pd.DataFrame({
                'id': test_raw['id'],
                'predict_0': test_proba[:, 0],
                'predict_1': test_proba[:, 1],
                'predict_2': test_proba[:, 2]
            })
        else:
            sub = pd.DataFrame({'id': test_raw['id'], 'predict': test_proba.flatten()})
        
        # 4. 存檔到 Kaggle 的工作區
        out_csv_path = "submission.csv"
        sub.to_csv(out_csv_path, index=False)
        print(f"✅ 成功！已儲存至 /kaggle/working/I2AI_FinalProject_AutoML/{out_csv_path}")
        
    except Exception as e:
        print(f"⚠️ 生成提交檔失敗，錯誤原因: {e}")
        import traceback
        traceback.print_exc()

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
    parser.add_argument("--csv",          type=str, default=None,
                        help="單一 CSV 檔案路徑（80/20 random split）")
    
    # 🚀 這裡加上了 nargs='+'，讓 --train 可以接收多個檔案變成 List
    parser.add_argument("--train",        nargs='+', default=None,
                        help="訓練集 CSV 路徑清單（支援多檔，搭配 --test 使用預切分模式）")
    
    parser.add_argument("--test",         type=str, default=None,
                        help="測試集 CSV 路徑（搭配 --train 使用預切分模式）")
    parser.add_argument("--target",       type=str, default=None,
                        help="目標欄位名稱（--csv / --train/--test 模式必填）")
    parser.add_argument("--result-file",  default=None,
                        help="附加結果 CSV（含 source 欄，附加模式）")
    parser.add_argument("--viz",          action="store_true",
                        help="訓練完成後自動產生 SHAP 視覺化圖表（需要 shap + plotly + kaleido）")

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
