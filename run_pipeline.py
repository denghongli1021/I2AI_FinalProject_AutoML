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
    # 1. 降維特徵：因為特徵已混淆，直接用 PC/SVD 命名
    if any(k in feature_set for k in ("pca", "svd", "kpca")):
        return [f"{feature_set}_{i}" for i in range(n_transformed)]
    # 2. 完美對齊：如果轉換後的維度與原始特徵數一致，直接還原！
    if n_transformed == n:
        return orig_names.copy()
        
    # 3. Signal 相關特徵 (支援模糊匹配如 "ts_signal")
    if "signal" in feature_set:
        base = orig_names.copy()
        extended = base + [f"{nm}_l2" for nm in base]
        
        if n_transformed <= len(extended):
            return extended[:n_transformed]
        else:
            # 萬一維度比預期還大，後面再用 feature_set 名稱補齊
            return extended + [f"{feature_set}_{i}" for i in range(n_transformed - len(extended))]
            
    # 4. 擴增特徵 (如 raw_stat 增加了統計量)
    if n_transformed > n:
        return orig_names + [f"{feature_set}_{i}" for i in range(n_transformed - n)]
        
    # 5. 【關鍵修復 Fallback】：如果維度變小 (例如前處理自動剔除了無效/常數欄位)
    # 絕對不能變成 feat_N！至少保留對應數量的真實名稱。
    return orig_names[:n_transformed]


def _smart_prepare(df: pd.DataFrame, target_col: str):
    """
    使用 preprocessing 模組進行智慧前處理。
    AutoRouter 自動偵測 numeric / categorical / text / datetime 欄位，
    建立雙軌管線（tree 軌道保留原始尺度供樹狀模型使用）。
    回傳 (X_tr, X_te, y_tr_raw, y_te_raw, feature_names)，X 為 float32 numpy。
    若 preprocessing 模組不可用，自動回退到基本的 _prepare_X() + train_test_split。
    """
    try:
        from preprocessing.interface import preprocess_for_training
        X_dict_tr, X_dict_te, y_tr_raw, y_te_raw, _ = preprocess_for_training(
            data_source=df,
            target_col=target_col,
            test_size=0.2,
        )
        X_tr = X_dict_tr["tree"].values.astype(np.float32)
        X_te = X_dict_te["tree"].values.astype(np.float32)
        feature_names = list(X_dict_tr["tree"].columns)
        return X_tr, X_te, y_tr_raw, y_te_raw, feature_names
    except Exception as _e:
        print(f"  [Preprocess] 智慧前處理失敗（{_e}），回退到基本前處理")
        X_all = _prepare_X(df, target_col)
        feat_names = _get_feature_names(df, target_col)
        y_all_raw = df[target_col]
        X_tr, X_te, y_tr_raw, y_te_raw = train_test_split(
            X_all, y_all_raw, test_size=0.2, random_state=SEED
        )
        return X_tr, X_te, y_tr_raw, y_te_raw, feat_names


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


def _run_dl_shap_visualization(artifacts_dir: str, X_train_raw: np.ndarray,
                               X_test_raw: np.ndarray, feature_names: list,
                               dataset_name: str = "", max_test_samples: int = 200):
    """Load best DL model (.pt) from artifacts_dir and generate SHAP plots via PermutationExplainer."""
    try:
        import torch
        import sys as _sys
        import os
        import pandas as pd
        import numpy as np
        
        _viz_path = os.path.join(HERE, "visualization")
        if _viz_path not in _sys.path:
            _sys.path.insert(0, _viz_path)
            
        from visualizer import AutoMLVisualizer
        from src.train import _build_dl_model
        from src.preprocess import FeatureBuilder
        from src.config import SEED as _SEED

        pt_path = os.path.join(artifacts_dir, "best_dl_model.pt")
        if not os.path.exists(pt_path):
            print("[Viz-DL] best_dl_model.pt 不存在，跳過 DL 視覺化"); return

        ckpt        = torch.load(pt_path, map_location="cpu")
        config      = ckpt["config"]
        in_features = ckpt["in_features"]
        n_classes   = ckpt["n_classes"]
        model_name  = config["model_name"]
        feature_set = config["feature_set"]

        _model_obj = _build_dl_model(model_name, config["arch_params"], in_features, n_classes)
        _model_obj.load_state_dict(ckpt["state_dict"])
        _model_obj.eval()
        print(f"[Viz-DL] 載入 {model_name}  feature_set={feature_set}"
              f"  in_features={in_features}  n_classes={n_classes}")

        class _Predictor:
            def __init__(self, m):
                self._m = m
            def predict_proba(self, X):
                with torch.no_grad():
                    if isinstance(X, pd.DataFrame):
                        X = X.values
                    return torch.softmax(self._m(torch.FloatTensor(X)), dim=1).numpy()
            def predict(self, X):
                return self.predict_proba(X).argmax(axis=1)

        predictor = _Predictor(_model_obj)

        _gcfg_candidates = [
            {"use_kmeans": True,  "use_kpca": False},
            {"use_kmeans": False, "use_kpca": False},
            {"use_kmeans": True,  "use_kpca": True},
            {"use_kmeans": False, "use_kpca": True},
        ]
        
        fb = None
        for _gcfg in _gcfg_candidates:
            _fb_try = FeatureBuilder(feature_set=feature_set, global_cfg=_gcfg).fit(X_train_raw)
            if _fb_try.transform(X_train_raw[:1]).shape[1] == in_features:
                fb = _fb_try
                print(f"[Viz-DL] FeatureBuilder(feature_set={feature_set},"
                      f" use_kmeans={_gcfg['use_kmeans']}) → {in_features} 維")
                break
                
        if fb is None:
            print(f"[Viz-DL] 警告：無法自動匹配 in_features={in_features}，使用預設 global_cfg")
            fb = FeatureBuilder(feature_set=feature_set).fit(X_train_raw)

        X_tf = fb.transform(X_test_raw)
        
        # 🚀 [修改點] 生成 SHAP 欄位名稱
        col_names = _build_shap_col_names(feature_set, feature_names, X_tf.shape[1])
        
        # 建立 DataFrame
        X_df = pd.DataFrame(X_tf, columns=col_names)

        if len(X_df) > max_test_samples:
            rng = np.random.default_rng(_SEED)
            idx = rng.choice(len(X_df), size=max_test_samples, replace=False)
            X_df = X_df.iloc[idx].reset_index(drop=True)
            print(f"[Viz-DL] X_test 取樣 {max_test_samples}/{X_tf.shape[0]} 筆"
                  f"（可用 --dl-shap-samples 調整）")

        viz_output = os.path.join(artifacts_dir, "shap_plots_dl")
        print(f"[Viz-DL] 執行 SHAP（{X_df.shape[0]} 筆 × {X_df.shape[1]} 特徵）...")
        viz = AutoMLVisualizer(model=predictor, X_test=X_df, output_dir=viz_output)

        shap_mat = viz._get_shap_matrix()
        top_feat = col_names[int(np.abs(shap_mat).mean(0).argmax())]
        viz.generate_all_plots(
            sample_index=0,
            target_feature=top_feat,
            prefix=f"{dataset_name}_dl",
        )
        print(f"\n[Viz-DL] SHAP 圖表已輸出 → {viz_output}")
    except Exception as _e:
        print(f"\n[Viz-DL] DL 視覺化跳過（{_e}）")



# ── 批次評估模式 ──────────────────────────────────────────────────────────────

def run_batch(args):
    openml_dir = os.path.join(HERE, args.openml_dir)
    reg_dir    = os.path.join(HERE, args.reg_dir)

    datasets = []
    if os.path.isdir(openml_dir):
        datasets += _find_datasets(openml_dir, args.top_n, last=args.last)
    else:
        print(f"[警告] 找不到 OpenML 目錄：{openml_dir}")
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
            df = df.dropna(subset=[target_col]).reset_index(drop=True)
            y_raw = df[target_col]
            task = force_task if force_task else _auto_detect_task(y_raw)
            X_tr, X_te, y_tr_raw, y_te_raw, feature_names = _smart_prepare(df, target_col)

            if task == "classification":
                le = LabelEncoder()
                le.fit(y_raw.astype(str).values)
                y_tr = le.transform(y_tr_raw.astype(str).values)
                y_te = le.transform(y_te_raw.astype(str).values)
                n_classes = len(le.classes_)
                print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  split=Random Stratified")

                _batch_artifacts = os.path.join(ARTIFACTS_DIR, "batch", dataset_name)
                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
                cfg["is_timeseries"] = False
                result = _pl.run(
                    X_tr, y_tr, X_te, n_classes, cfg, budget,
                    skip_tabular=args.skip_tabular,
                    skip_dl=args.skip_dl,
                    no_nas=args.no_nas,
                    is_ts=False,
                    artifacts_dir=_batch_artifacts,
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
                if args.viz:
                    _run_shap_visualization(_batch_artifacts, X_te, feature_names, dataset_name)
                    _run_dl_shap_visualization(_batch_artifacts, X_tr, X_te, feature_names, dataset_name, args.dl_shap_samples)
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
                print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  task=regression  split=Random")

                budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
                cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))
                result = _pt.run_regression(
                    X_tr, y_tr, X_te, cfg, budget,
                    skip_tabular=args.skip_tabular,
                    skip_dl=args.skip_dl,
                    artifacts_dir=os.path.join(ARTIFACTS_DIR, "batch", dataset_name),
                    metric=args.reg_metric,
                )
                rmse_b, rmse_s, r2_b, r2_s, best_rmse, best_r2, primary_score = \
                    _eval_regression(y_te, result, args.reg_metric)
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
    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"[錯誤] 找不到檔案：{csv_path}"); sys.exit(1)

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n{'='*65}")
    print(f"  Pipeline 單資料集模式（非時序）  ─  {dataset_name}")
    print(f"{'='*65}")
    t_ds = time.time()

    df = pd.read_csv(csv_path)
    if not getattr(args, "target", None):
        print(f"[錯誤] --csv 模式需以 --target 明確指定目標欄位名稱。可用欄位：{list(df.columns)}")
        sys.exit(1)
    target_col = args.target
    if target_col not in df.columns:
        print(f"[錯誤] 找不到目標欄位 '{target_col}'，可用欄位：{list(df.columns)}")
        sys.exit(1)

    y_raw = df[target_col]
    task = _auto_detect_task(y_raw)
    X_tr, X_te, y_tr_raw, y_te_raw, feature_names = _smart_prepare(df, target_col)

    if task == "classification":
        le = LabelEncoder()
        le.fit(y_raw.astype(str).values)
        y_tr = le.transform(y_tr_raw.astype(str).values)
        y_te = le.transform(y_te_raw.astype(str).values)
        n_classes = len(le.classes_)
        print(f"  target={target_col}  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}")

        _single_artifacts = os.path.join(ARTIFACTS_DIR, "single", dataset_name)
        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False
        result = _pl.run(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            is_ts=False,
            artifacts_dir=_single_artifacts,
            metric=args.metric,
        )
        from src.metrics import calculate_score, get_metric_name
        score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
        score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → {get_metric_name(args.metric)}={score_b:.4f}")
        print(f"  [結果] Stack → {get_metric_name(args.metric)}={score_s:.4f}")
        print(f"  [耗時] {elapsed}s")
        if args.viz:
            _run_shap_visualization(_single_artifacts, X_te, feature_names, dataset_name)
            _run_dl_shap_visualization(_single_artifacts, X_tr, X_te, feature_names, dataset_name, args.dl_shap_samples)

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
            artifacts_dir=os.path.join(ARTIFACTS_DIR, "single", dataset_name),
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

    if task == "classification":
        le = LabelEncoder()
        y_tr = le.fit_transform(y_tr_raw.astype(str).values)
        classes_set = set(le.classes_)
        y_te = np.array(
            [le.transform([str(v)])[0] if str(v) in classes_set else 0
             for v in y_te_raw], dtype=np.int64)
        n_classes = len(le.classes_)
        print(f"  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}  task=classification")

        _presplit_artifacts = os.path.join(ARTIFACTS_DIR, "single", dataset_name)
        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False
        result = _pl.run(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            is_ts=False,
            artifacts_dir=_presplit_artifacts,
            metric=args.metric,
        )
        from src.metrics import calculate_score, get_metric_name
        score_b = calculate_score(y_te, result.test_blend, metric=args.metric)
        score_s = calculate_score(y_te, result.test_stack, metric=args.metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → {get_metric_name(args.metric)}={score_b:.4f}")
        print(f"  [結果] Stack → {get_metric_name(args.metric)}={score_s:.4f}")
        print(f"  [耗時] {elapsed}s")
        if args.viz:
            _run_shap_visualization(_presplit_artifacts, X_te, feature_names, dataset_name)
            _run_dl_shap_visualization(_presplit_artifacts, X_tr, X_te, feature_names, dataset_name, args.dl_shap_samples)

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
            artifacts_dir=os.path.join(ARTIFACTS_DIR, "single", dataset_name),
            metric=args.reg_metric,
        )
        rmse_b, rmse_s, r2_b, r2_s, _, _, _ = _eval_regression(y_te, result, args.reg_metric)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
        print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}")
        print(f"  [耗時] {elapsed}s")

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
    parser.add_argument("--dl-shap-samples", type=int, default=200,
                        help="DL SHAP 使用的 X_test 樣本數上限（預設 200，控制 PermutationExplainer 耗時）")

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
