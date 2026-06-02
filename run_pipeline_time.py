"""
run_pipeline_time.py — 時序專用 Pipeline 入口（v1）

僅掃 ucr_ts_80_new(時序資料)/（預切分格式：*_TRAIN.csv + *_TEST.csv），
每個資料集依前綴自動偵測 task：
  - REG_* 前綴   → 回歸（用 pipeline_time.run_regression，Walk-forward 時序切分）
  - 其他（含 CLS_*）→ 分類（用 pipeline_time.run_classification，即 pipeline.run(is_ts=True)）

用法：
    # 批次（時序資料夾的前 N 個 / 後 N 個）
    python run_pipeline_time.py --batch --top-n 10
    python run_pipeline_time.py --batch --last --top-n 10
    python run_pipeline_time.py --batch --fast

    # 單一 TRAIN CSV（自動尋找對應 TEST CSV）
    python run_pipeline_time.py --csv "ucr_ts_80_new(時序資料)/REG_VentilatorPressure_TRAIN.csv"

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
sys.path.insert(0, os.path.join(HERE, "visualization"))

from src.config import ARTIFACTS_DIR, DEVICE, SEED
from src import pipeline_time as _pt


# ── 工具 ─────────────────────────────────────────────────────────────────────

def _find_target_col(df: pd.DataFrame) -> str:
    """批次模式專用：只做精確名稱比對，找不到則報錯。"""
    for c in ("target", "label", "class", "y", "c"):
        if c in df.columns:
            return c
    raise ValueError(
        f"找不到標準目標欄位（target/label/class/y/c）。"
        f"可用欄位：{list(df.columns)}。"
        f"請以 --target 明確指定目標欄位名稱。"
    )


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


def _build_shap_col_names(feature_set: str, orig_names: list, n_transformed: int) -> list:
    n = len(orig_names)
    if feature_set == "raw":
        return orig_names[:n_transformed]
    if feature_set == "signal":
        base = orig_names[:n]
        return (base + [f"{nm}_l2" for nm in base])[:n_transformed]
    if feature_set in ("pca64", "svd64", "kpca32"):
        return [f"PC{i}" for i in range(n_transformed)]
    extra = [f"feat_{i}" for i in range(max(0, n_transformed - n))]
    return (orig_names + extra)[:n_transformed]


def _run_shap_visualization(artifacts_dir: str, X_test_raw: np.ndarray,
                             feature_names: list, dataset_name: str = ""):
    try:
        from visualizer import AutoMLVisualizer
        import joblib as _jl
        model_path = os.path.join(artifacts_dir, "best_tabular_model.pkl")
        fb_path    = os.path.join(artifacts_dir, "best_tabular_model_fb.pkl")
        if not os.path.exists(model_path):
            print(f"\n[Viz] 找不到 {model_path}，跳過"); return
        model = _jl.load(model_path)
        fb    = _jl.load(fb_path)
        X_tf  = fb.transform(X_test_raw)
        col_names = _build_shap_col_names(fb.feature_set, feature_names, X_tf.shape[1])
        X_df  = pd.DataFrame(X_tf, columns=col_names)
        out_dir = os.path.join(artifacts_dir, "shap_plots")
        viz = AutoMLVisualizer(model=model, X_test=X_df, output_dir=out_dir)
        shap_mat = viz._get_shap_matrix()
        top_feat = col_names[int(np.abs(shap_mat).mean(0).argmax())]
        viz.generate_all_plots(sample_index=0, target_feature=top_feat,
                               prefix=dataset_name or "pipeline_time")
        print(f"\n[Viz] SHAP 圖表已輸出 → {out_dir}")
    except Exception as _e:
        print(f"\n[Viz] 視覺化跳過（{_e}）")


def _ts_datasets(ts_dir: str, top_n: int, last: bool) -> list:
    """回傳 (base_name, train_path, test_path) 列表（ucr_ts_80_new 預切分格式）。"""
    trains = sorted(f for f in os.listdir(ts_dir) if f.endswith("_TRAIN.csv"))
    chosen = trains[-top_n:] if last else trains[:top_n]
    result = []
    for f in chosen:
        base = f[:-10]  # strip "_TRAIN.csv"
        result.append((base,
                       os.path.join(ts_dir, f),
                       os.path.join(ts_dir, base + "_TEST.csv")))
    return result

def _run_dl_shap_visualization(artifacts_dir: str, X_train_raw: np.ndarray,
                               X_test_raw: np.ndarray, feature_names: list,
                               dataset_name: str = "", max_test_samples: int = 200,
                               task: str = "classification"):
    """載入最佳深度學習模型 (.pt) 並透過自適應 3D 轉換產生時序 DL SHAP 圖表"""
    try:
        import torch
        import sys as _sys
        import os
        import pandas as pd
        import numpy as np
        
        from visualizer import AutoMLVisualizer
        from src.train import _build_dl_model
        from src.preprocess import FeatureBuilder
        from src.config import SEED as _SEED

        pt_path = os.path.join(artifacts_dir, "best_dl_model.pt")
        if not os.path.exists(pt_path):
            print("[Viz-DL] best_dl_model.pt 不存在，跳過 DL 視覺化"); return

        # 讀取權重與配置
        ckpt        = torch.load(pt_path, map_location="cpu")
        config      = ckpt["config"]
        in_features = ckpt["in_features"]
        n_classes   = ckpt.get("n_classes", 1)  # 回歸任務可能無種類數，預設為 1
        model_name  = config["model_name"]
        feature_set = config["feature_set"]

        # 重建 DL 模型架構
        _model_obj = _build_dl_model(model_name, config["arch_params"], in_features, n_classes)
        _model_obj.load_state_dict(ckpt["state_dict"])
        _model_obj.eval()
        print(f"[Viz-DL] 載入 {model_name}  feature_set={feature_set}"
              f"  in_features={in_features}  task={task}")

        # 🚀 【步驟 2 核心實作】支援時序 3D Tensor 的 Predictor 包裝器
        class _Predictor:
            def __init__(self, m, task_type):
                self._m = m
                self.task_type = task_type
                
            def predict_proba(self, X):
                with torch.no_grad():
                    if isinstance(X, pd.DataFrame):
                        X = X.values
                    x_tensor = torch.FloatTensor(X)
                    
                    # 💡 試探法：自動依據時序模型的維度要求進行 2D -> 3D 重排
                    try:
                        out = self._m(x_tensor)  # 嘗試原本的 2D [batch, features]
                    except Exception:
                        try:
                            # 嘗試 1D-CNN 常見格式: [batch, 1, seq_len]
                            out = self._m(x_tensor.unsqueeze(1))
                        except Exception:
                            # 嘗試 Transformer/RNN 常見格式: [batch, seq_len, 1]
                            out = self._m(x_tensor.unsqueeze(-1))
                    
                    if self.task_type == "classification":
                        return torch.softmax(out, dim=1).numpy()
                    else:
                        return out.numpy().reshape(len(X), -1)

            def predict(self, X):
                res = self.predict_proba(X)
                if self.task_type == "classification":
                    return res.argmax(axis=1)
                return res

        predictor = _Predictor(_model_obj, task)

        # 自動匹配特徵轉換器
        _gcfg_candidates = [
            {"use_kmeans": True,  "use_kpca": False},
            {"use_kmeans": False, "use_kpca": False},
            {"use_kmeans": True,  "use_kpca": True},
            {"use_kmeans": False, "use_kpca": True},
        ]
        fb = None
        for _gcfg in _gcfg_candidates:
            try:
                _fb_try = FeatureBuilder(feature_set=feature_set, global_cfg=_gcfg).fit(X_train_raw)
                if _fb_try.transform(X_train_raw[:1]).shape[1] == in_features:
                    fb = _fb_try
                    break
            except Exception:
                continue
                
        if fb is None:
            fb = FeatureBuilder(feature_set=feature_set).fit(X_train_raw)

        X_tf = fb.transform(X_test_raw)
        col_names = _build_shap_col_names(feature_set, feature_names, X_tf.shape[1])
        X_df = pd.DataFrame(X_tf, columns=col_names)

        # 樣本數過大時進行採樣，避免 SHAP 算太久
        if len(X_df) > max_test_samples:
            rng = np.random.default_rng(_SEED)
            idx = rng.choice(len(X_df), size=max_test_samples, replace=False)
            X_df = X_df.iloc[idx].reset_index(drop=True)
            print(f"[Viz-DL] X_test 取樣 {max_test_samples}/{X_tf.shape[0]} 筆進行解釋")

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
        print(f"\n[Viz-DL] 時序 DL SHAP 圖表已輸出 → {viz_output}")
    except Exception as _e:
        print(f"\n[Viz-DL] DL 視覺化跳過（{_e}）")
# ── 單一資料集執行 ───────────────────────────────────────────────────────────

def _process_one(csv_path: str, args, t_ds: float) -> dict:
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)
    if not args.target:
        raise ValueError(
            f"--csv 模式需以 --target 明確指定目標欄位名稱。可用欄位：{list(df.columns)}"
        )
    target_col = args.target
    if target_col not in df.columns:
        raise ValueError(f"找不到目標欄 '{target_col}'，可用：{list(df.columns)}")

    n_before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(df) < n_before:
        print(f"  [info] dropped {n_before - len(df)} rows with NaN target")

    y_raw = df[target_col]
    task = _auto_detect_task(csv_path, y_raw)
    print(f"  [Task] {task}  target={target_col}")
    _arts_dir = os.path.join(ARTIFACTS_DIR, "batch_time", dataset_name)

    # 特徵：僅取數值欄、填 0
    feature_names = list(df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns)
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
        from src import pipeline as _pl
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

        if getattr(args, "viz", False):
            _run_shap_visualization(_arts_dir, X_te, feature_names, dataset_name)

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

    if getattr(args, "viz", False):
        _run_shap_visualization(_arts_dir, X_te, feature_names, dataset_name)

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


# ── time_results.csv 工具 ────────────────────────────────────────────────────

_TIME_RESULT_COLS = [
    "source", "dataset", "type", "task", "n_train", "n_test",
    "accuracy", "f1_macro", "rmse", "r2", "score", "elapsed_s", "fast",
]


def _append_time_result(out_path: str, row: dict):
    """Append one result row; write header only when file is new."""
    df = pd.DataFrame([{c: row.get(c) for c in _TIME_RESULT_COLS}])
    df.to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)


# ── 新TS批次（ucr_ts_80_new）────────────────────────────────────────────────

def _process_new_ts_one(base_name: str, train_df: pd.DataFrame,
                        test_df: pd.DataFrame, args, t_ds: float,
                        out_path: str = None) -> dict:
    """Run pipeline on one pre-split TRAIN/TEST dataset pair."""
    task = "regression" if base_name.startswith("REG_") else "classification"
    target_col = _find_target_col(train_df)
    _arts_dir = os.path.join(ARTIFACTS_DIR, "batch_new_ts", base_name)

    train_df = train_df.dropna(subset=[target_col]).reset_index(drop=True)
    test_df  = test_df.dropna(subset=[target_col]).reset_index(drop=True)

    y_tr_raw = train_df[target_col]
    y_te_raw = test_df[target_col]

    _feat_cols = list(train_df.drop(columns=[target_col])
                      .select_dtypes(include=[np.number]).columns)
    X_tr = (train_df.drop(columns=[target_col])
              .select_dtypes(include=[np.number])
              .fillna(0).values.astype(np.float32))
    X_te = (test_df.drop(columns=[target_col])
              .select_dtypes(include=[np.number])
              .fillna(0).values.astype(np.float32))

    if X_tr.shape[1] == 0:
        raise ValueError("無可用數值特徵（select_dtypes 後 0 欄）")

    # Truncate to the minimum feature count if TRAIN/TEST column counts differ
    min_cols = min(X_tr.shape[1], X_te.shape[1])
    X_tr = X_tr[:, :min_cols]
    X_te = X_te[:, :min_cols]
    feature_names = _feat_cols[:min_cols]

    if task == "classification":
        from sklearn.preprocessing import LabelEncoder as _LE
        le = _LE()
        y_tr = le.fit_transform(y_tr_raw.astype(str).values)
        # Map unseen test labels to class 0 rather than crashing
        classes_set = set(le.classes_)
        y_te = np.array(
            [le.transform([str(v)])[0] if str(v) in classes_set else 0
             for v in y_te_raw],
            dtype=np.int64,
        )
        n_classes = len(le.classes_)
        print(f"  [Task] classification  n_train={len(y_tr)}  n_test={len(y_te)}  n_classes={n_classes}")

        budget = _pt.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
        from src import pipeline as _pl
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = False

        result = _pt.run_classification(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            artifacts_dir=os.path.join(ARTIFACTS_DIR, "batch_new_ts", base_name),
            metric=args.cls_metric,
        )
        from src.metrics import calculate_score
        score_b = calculate_score(y_te, result.test_blend, metric=args.cls_metric)
        score_s = calculate_score(y_te, result.test_stack, metric=args.cls_metric)
        best_preds = result.test_stack if score_s >= score_b else result.test_blend
        acc = round(calculate_score(y_te, best_preds, metric="accuracy"), 4)
        f1  = round(calculate_score(y_te, best_preds, metric="f1"), 4)
        elapsed = round(time.time() - t_ds, 1)
        print(f"\n  [結果] Blend={score_b:.4f}  Stack={score_s:.4f}  ({elapsed}s)")

        row = {
            "source": "pipeline",
            "dataset": base_name, "type": "TS", "task": task,
            "n_train": len(y_tr), "n_test": len(y_te),
            "accuracy": acc, "f1_macro": f1,
            "rmse": None, "r2": None,
            "score": round(max(score_b, score_s), 4),
            "elapsed_s": elapsed,
            "fast": args.fast,
        }
        if out_path:
            _append_time_result(out_path, row)
            row["_already_saved"] = True

        if getattr(args, "viz", False):
            _run_shap_visualization(_arts_dir, X_te, feature_names, base_name)
            if not getattr(args, "skip_dl", False):
                _run_dl_shap_visualization(_arts_dir, X_tr, X_te, feature_names, base_name,
                                           max_test_samples=getattr(args, "dl_shap_samples", 200),
                                           task="classification")

        return row

    # ── 回歸 ────────────────────────────────────────────────────────────────
    y_tr = np.asarray(y_tr_raw.values, dtype=np.float32).ravel()
    y_te = np.asarray(y_te_raw.values, dtype=np.float32).ravel()
    print(f"  [Task] regression  n_train={len(y_tr)}  n_test={len(y_te)}")

    budget = _pt.TimeBudget(limit_sec=args.time_limit, t_start=t_ds)
    cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))

    result = _pt.run_regression(
        X_tr, y_tr, X_te, cfg, budget,
        skip_tabular=args.skip_tabular,
        skip_dl=args.skip_dl,
        artifacts_dir=os.path.join(ARTIFACTS_DIR, "batch_new_ts", base_name),
        metric=args.reg_metric,
    )

    from sklearn.metrics import mean_squared_error, r2_score
    rmse_b = float(np.sqrt(mean_squared_error(y_te, result.test_blend)))
    rmse_s = float(np.sqrt(mean_squared_error(y_te, result.test_stack)))
    r2_b   = float(r2_score(y_te, result.test_blend))
    r2_s   = float(r2_score(y_te, result.test_stack))

    if args.reg_metric == "r2":
        best_is_stack = r2_s >= r2_b
        primary_score = max(r2_b, r2_s)
    else:
        best_is_stack = rmse_s <= rmse_b
        primary_score = min(rmse_b, rmse_s)

    best_rmse = rmse_s if best_is_stack else rmse_b
    best_r2   = r2_s   if best_is_stack else r2_b
    elapsed = round(time.time() - t_ds, 1)
    print(f"\n  [結果] Blend → RMSE={rmse_b:.4f}  R2={r2_b:.4f}")
    print(f"  [結果] Stack → RMSE={rmse_s:.4f}  R2={r2_s:.4f}  ({elapsed}s)")

    row = {
        "source": "pipeline",
        "dataset": base_name, "type": "TS", "task": task,
        "n_train": len(y_tr), "n_test": len(y_te),
        "accuracy": None, "f1_macro": None,
        "rmse": round(best_rmse, 4), "r2": round(best_r2, 4),
        "score": round(primary_score, 4),
        "elapsed_s": elapsed,
        "fast": args.fast,
    }
    if out_path:
        _append_time_result(out_path, row)
        row["_already_saved"] = True

    if getattr(args, "viz", False):
        _run_shap_visualization(_arts_dir, X_te, feature_names, base_name)
        if not getattr(args, "skip_dl", False):
            _run_dl_shap_visualization(_arts_dir, X_tr, X_te, feature_names, base_name,
                                       max_test_samples=getattr(args, "dl_shap_samples", 200),
                                       task="regression")

    return row


def run_new_ts_batch_pipeline(args):
    """Pipeline-Time evaluation on 3 CLS + 3 REG from ucr_ts_80_new(時序資料)."""
    new_ts_dir = os.path.join(HERE, "ucr_ts_80_new(時序資料)")
    if not os.path.isdir(new_ts_dir):
        print(f"[錯誤] 找不到目錄：{new_ts_dir}"); sys.exit(1)

    train_files = sorted(f for f in os.listdir(new_ts_dir) if f.endswith("_TRAIN.csv"))
    cls_top    = getattr(args, "cls_top_n",   3)
    reg_top    = getattr(args, "reg_top_n",   3)
    cls_offset = getattr(args, "cls_offset",  0)
    reg_offset = getattr(args, "reg_offset",  0)
    cls_bases = [f[:-10] for f in train_files if f.startswith("CLS_")][cls_offset:cls_offset + cls_top]
    reg_bases = [f[:-10] for f in train_files if f.startswith("REG_")][reg_offset:reg_offset + reg_top]
    selected = cls_bases + reg_bases

    result_file = getattr(args, "result_file", None)
    out_path = os.path.join(HERE, result_file) if result_file else os.path.join(HERE, "time_results.csv")

    # 斷點續跑：只跳過 source==pipeline 且 fast 值相同的記錄（區分 fast/full 兩次跑）
    done_datasets = set()
    if os.path.exists(out_path):
        try:
            _existing = pd.read_csv(out_path)
            if "source" in _existing.columns and "fast" in _existing.columns:
                _mask = (_existing["source"] == "pipeline") & \
                        (_existing["fast"].fillna(False).astype(bool) == bool(args.fast))
                done_datasets = set(_existing.loc[_mask, "dataset"].tolist())
            else:
                done_datasets = set(_existing["dataset"].tolist())
            if done_datasets:
                print(f"  [Resume] 已完成 {len(done_datasets)} 個，將跳過: {sorted(done_datasets)}")
        except Exception:
            pass

    print(f"\n{'='*65}")
    print(f"  Pipeline-Time 新TS批次  ─  {len(selected)} 個資料集  (source=pipeline)")
    print(f"{'='*65}")

    for base_name in selected:
        if base_name in done_datasets:
            print(f"\n  [Skip] {base_name}（已有結果，跳過）")
            continue
        train_path = os.path.join(new_ts_dir, base_name + "_TRAIN.csv")
        test_path  = os.path.join(new_ts_dir, base_name + "_TEST.csv")
        task = "regression" if base_name.startswith("REG_") else "classification"
        print(f"\n{'─'*65}")
        print(f"  [TS] {base_name}  |  device={DEVICE}")
        print(f"{'─'*65}")
        t_ds = time.time()
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            row = _process_new_ts_one(base_name, train_df, test_df, args, t_ds,
                                       out_path=out_path)
        except Exception:
            traceback.print_exc()
            row = {
                "source": "pipeline",
                "dataset": base_name, "type": "TS", "task": task,
                "n_train": None, "n_test": None,
                "accuracy": None, "f1_macro": None,
                "rmse": None, "r2": None, "score": None,
                "elapsed_s": round(time.time() - t_ds, 1),
                "fast": args.fast,
            }
        finally:
            import gc as _gc
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
        if not row.get("_already_saved"):
            _append_time_result(out_path, row)

    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


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

    for base_name, train_path, test_path in datasets:
        if base_name in done_datasets:
            print(f"\n  [Skip] {base_name}（已有結果，跳過）")
            continue
        print(f"\n{'─'*65}")
        print(f"  [TS] {base_name}  |  device={DEVICE}")
        print(f"{'─'*65}")
        t_ds = time.time()
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            row = _process_new_ts_one(base_name, train_df, test_df, args, t_ds)
            results.append(row)
        except Exception:
            traceback.print_exc()
            results.append({
                "dataset": base_name, "type": "TS", "task": "?",
                "n_train": None, "n_test": None,
                "accuracy": None, "f1_macro": None,
                "rmse": None, "r2": None, "score": None,
                "elapsed_s": round(time.time() - t_ds, 1),
                "fast": args.fast,
            })
        finally:
            import gc as _gc
            _gc.collect()
            try:
                import torch as _torch
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
        _flush()

    print(f"\n{'='*65}")
    print("  BATCH SUMMARY — Pipeline Time")
    print(f"{'='*65}")
    summary = pd.DataFrame(results)
    print(summary.to_string(index=False))
    _flush()
    print(f"\n  結果已儲存 → {out_path}")
    print(f"{'='*65}\n")


# ── 預切分單一模式 ────────────────────────────────────────────────────────────

def run_presplit(args):
    """用戶手動提供 TRAIN / TEST 兩個 CSV，直接使用不再自行切分。"""
    train_path = os.path.abspath(args.train)
    test_path  = os.path.abspath(args.test)
    for p, label in [(train_path, "TRAIN"), (test_path, "TEST")]:
        if not os.path.isfile(p):
            print(f"[錯誤] 找不到{label}檔案：{p}")
            sys.exit(1)

    base = os.path.splitext(os.path.basename(train_path))[0]
    base_name = base[:-6] if base.endswith("_TRAIN") else base

    print(f"\n{'='*65}")
    print(f"  Pipeline-Time 預切分模式  ─  {base_name}  |  device={DEVICE}")
    print(f"{'='*65}")
    t_ds = time.time()
    try:
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
        row = _process_new_ts_one(base_name, train_df, test_df, args, t_ds)
        print("\n  [完成]", row)
    except Exception:
        traceback.print_exc()


# ── 單一 CSV 模式 ────────────────────────────────────────────────────────────

def run_single(args):
    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"[錯誤] 找不到檔案：{csv_path}"); sys.exit(1)

    # 若傳入 _TRAIN.csv，自動尋找對應 _TEST.csv 並使用預切分模式
    if csv_path.endswith("_TRAIN.csv"):
        test_path = csv_path[:-10] + "_TEST.csv"
        if not os.path.isfile(test_path):
            print(f"[錯誤] 找不到對應 TEST 檔案：{test_path}"); sys.exit(1)
        base_name = os.path.basename(csv_path)[:-10]
        print(f"\n{'='*65}")
        print(f"  Pipeline-Time 單檔  ─  {base_name}  |  device={DEVICE}")
        print(f"{'='*65}")
        t_ds = time.time()
        result_file = getattr(args, "result_file", None)
        out_path = os.path.join(HERE, result_file) if result_file else None
        try:
            train_df = pd.read_csv(csv_path)
            test_df  = pd.read_csv(test_path)
            row = _process_new_ts_one(base_name, train_df, test_df, args, t_ds,
                                      out_path=out_path)
            print("\n  [完成]", row)
        except Exception:
            traceback.print_exc()
        return

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
    parser.add_argument("--ts-dir", default="ucr_ts_80_new(時序資料)",
                        help="UCR 時序目錄（預切分格式：含 _TRAIN.csv / _TEST.csv）")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--last", action="store_true",
                        help="取每目錄最後 top-n 個")
    parser.add_argument("--out", default="pipeline_time_batch_results.csv",
                        help="批次模式輸出 CSV")
    parser.add_argument("--new-ts-batch", action="store_true",
                        help="新TS批次：讀 ucr_ts_80_new，各取N個CLS+REG，寫 time_results.csv")
    parser.add_argument("--cls-top-n",   type=int, default=3,
                        help="--new-ts-batch 模式：取 N 個 CLS 資料集（預設 3）")
    parser.add_argument("--reg-top-n",   type=int, default=3,
                        help="--new-ts-batch 模式：取 N 個 REG 資料集（預設 3）")
    parser.add_argument("--cls-offset",  type=int, default=0,
                        help="--new-ts-batch 模式：CLS 資料集起始偏移（預設 0，即從第 1 個開始）")
    parser.add_argument("--reg-offset",  type=int, default=0,
                        help="--new-ts-batch 模式：REG 資料集起始偏移（預設 0，即從第 1 個開始）")
    parser.add_argument("--result-file", default=None,
                        help="附加結果 CSV（含 source 欄，附加模式）")
    parser.add_argument("--viz", action="store_true",
                        help="訓練完成後自動產生 SHAP 視覺化圖表（需要 shap + plotly + kaleido）")

    parser.add_argument("--csv",    default=None, help="單一 CSV 路徑（或 _TRAIN.csv 自動找 _TEST.csv）")
    parser.add_argument("--train",  default=None, help="訓練集 CSV 路徑（搭配 --test 使用預切分模式）")
    parser.add_argument("--test",   default=None, help="測試集 CSV 路徑（搭配 --train 使用預切分模式）")
    parser.add_argument("--target", default=None,
                        help="目標欄位名稱（--csv 模式必填）")

    args = parser.parse_args()
    if args.new_ts_batch:
        run_new_ts_batch_pipeline(args)
    elif args.train and args.test:
        run_presplit(args)
    elif args.csv:
        run_single(args)
    elif args.batch:
        run_batch(args)
    else:
        parser.print_help()
        print("\n[提示] 請指定 --csv、--train/--test、--batch 或 --new-ts-batch。")


if __name__ == "__main__":
    main()
