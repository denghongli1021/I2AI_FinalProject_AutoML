"""
generate_shap.py — 從現有 artifacts 直接產生 SHAP 視覺化圖表

不需要重新訓練，直接載入已存在的模型檔並生成 SHAP 圖表。

使用方式（從專案根目錄執行）：
    # 使用 tabular 最佳模型（預設）
    python generate_shap.py --dataset test/dataset.csv --target RiskPerformance

    # 使用 DL 最佳模型（SignalTransformer）
    python generate_shap.py --dataset test/dataset.csv --target RiskPerformance --model dl

    # 同時產生兩種模型的 SHAP 圖
    python generate_shap.py --dataset test/dataset.csv --target RiskPerformance --model both

    # 指定不同的 artifacts 目錄
    python generate_shap.py --dataset test/dataset.csv --target RiskPerformance \\
        --artifacts artifacts/batch/37_diabetes

    # 覆蓋 tabular 模型的 feature-set（當 best_tabular_model_fb.pkl 不存在時使用）
    python generate_shap.py --dataset test/dataset.csv --target RiskPerformance --feature-set raw_stat
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "visualization"))

from src.config import SEED
from src.preprocess import FeatureBuilder


# ── 資料輔助 ──────────────────────────────────────────────────────────────────

def _auto_detect_task(y: pd.Series) -> str:
    if y.dtype == object or y.dtype == bool:
        return "classification"
    n = y.nunique()
    return "classification" if (n <= 50 and n / len(y) < 0.30) else "regression"


def _prepare_X_raw(df: pd.DataFrame, target_col: str) -> tuple[np.ndarray, list]:
    """回傳 (X_raw float32, feature_names)，不做任何標準化。"""
    feat_df = df.drop(columns=[target_col])
    X_num = feat_df.select_dtypes(include=[np.number])
    if X_num.shape[1] == 0:
        obj_df = feat_df.select_dtypes(include=["object", "category"])
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_num = pd.DataFrame(
            enc.fit_transform(obj_df.fillna("__missing__")),
            columns=obj_df.columns,
        )
    return X_num.fillna(0).values.astype(np.float32), list(X_num.columns)


def _build_col_names(feature_set: str, orig_names: list, n_transformed: int) -> list:
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


# ── Tabular 視覺化 ────────────────────────────────────────────────────────────

def run_tabular_viz(artifacts_dir: str, X_tr_raw: np.ndarray, X_te_raw: np.ndarray,
                    feature_names: list, dataset_name: str, feature_set_override: str | None):
    import joblib
    from visualizer import AutoMLVisualizer

    model_path = os.path.join(artifacts_dir, "best_tabular_model.pkl")
    fb_path    = os.path.join(artifacts_dir, "best_tabular_model_fb.pkl")

    if not os.path.exists(model_path):
        print(f"[Tabular] 找不到 {model_path}，跳過"); return

    model = joblib.load(model_path)
    print(f"[Tabular] 載入模型：{type(model).__name__}")

    # 載入或重建 FeatureBuilder
    if os.path.exists(fb_path):
        fb = joblib.load(fb_path)
        print(f"[Tabular] 載入 FeatureBuilder（feature_set={fb.feature_set}）")
    else:
        fs = feature_set_override or "raw"
        print(f"[Tabular] best_tabular_model_fb.pkl 不存在，重新 fit FeatureBuilder(feature_set={fs})")
        fb = FeatureBuilder(feature_set=fs).fit(X_tr_raw)

    X_tf = fb.transform(X_te_raw)
    col_names = _build_col_names(fb.feature_set, feature_names, X_tf.shape[1])

    X_df = pd.DataFrame(X_tf, columns=col_names)
    out_dir = os.path.join(artifacts_dir, "shap_plots_tabular")
    print(f"[Tabular] 執行 SHAP（{X_df.shape[0]} 筆 × {X_df.shape[1]} 特徵）...")

    viz = AutoMLVisualizer(model=model, X_test=X_df, output_dir=out_dir)
    shap_mat  = viz._get_shap_matrix()
    top_feat  = col_names[int(np.abs(shap_mat).mean(0).argmax())]

    viz.generate_all_plots(
        sample_index=0,
        target_feature=top_feat,
        prefix=f"{dataset_name}_tab",
    )
    print(f"[Tabular] 圖表已輸出 → {out_dir}")


# ── DL 視覺化 ─────────────────────────────────────────────────────────────────

class _DLPredictor:
    """把 PyTorch DL 模型包裝成 sklearn-compatible predict / predict_proba。"""

    def __init__(self, pt_path: str):
        import torch
        from src.train import _build_dl_model

        ckpt        = torch.load(pt_path, map_location="cpu")
        config      = ckpt["config"]
        in_features = ckpt["in_features"]
        n_classes   = ckpt["n_classes"]

        self.model_name  = config["model_name"]
        self.feature_set = config["feature_set"]
        self.in_features = in_features
        self.n_classes   = n_classes

        self._model = _build_dl_model(self.model_name, config["arch_params"],
                                      in_features, n_classes)
        self._model.load_state_dict(ckpt["state_dict"])
        self._model.eval()

        print(f"[DL] 載入模型：{self.model_name}"
              f"  feature_set={self.feature_set}"
              f"  in_features={in_features}  n_classes={n_classes}")
        print(f"     arch_params={config['arch_params']}")

    def predict_proba(self, X) -> np.ndarray:
        import torch
        if isinstance(X, pd.DataFrame):
            X = X.values
        with torch.no_grad():
            t = torch.FloatTensor(X)
            probs = torch.softmax(self._model(t), dim=1).numpy()
        return probs

    def predict(self, X) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def run_dl_viz(artifacts_dir: str, X_tr_raw: np.ndarray, X_te_raw: np.ndarray,
               feature_names: list, dataset_name: str,
               max_bg: int = 100, max_test_samples: int = 200):
    from visualizer import AutoMLVisualizer

    pt_path = os.path.join(artifacts_dir, "best_dl_model.pt")
    if not os.path.exists(pt_path):
        print(f"[DL] 找不到 {pt_path}，跳過"); return

    predictor = _DLPredictor(pt_path)

    # 重新 fit FeatureBuilder（DL 的 feature_set 從 .pt config 讀取）
    # 自動偵測訓練時的 global_cfg：依序嘗試常見的 use_kmeans 組合，
    # 找到與 checkpoint in_features 一致的設定即停止（避免 KMeans 維度不符）
    _target_dim = predictor.in_features
    _gcfg_candidates = [
        {"use_kmeans": True,  "use_kpca": False},
        {"use_kmeans": False, "use_kpca": False},
        {"use_kmeans": True,  "use_kpca": True},
        {"use_kmeans": False, "use_kpca": True},
    ]
    fb = None
    for _gcfg in _gcfg_candidates:
        _fb_try = FeatureBuilder(feature_set=predictor.feature_set,
                                  global_cfg=_gcfg).fit(X_tr_raw)
        if _fb_try.transform(X_tr_raw[:1]).shape[1] == _target_dim:
            fb = _fb_try
            print(f"[DL] FeatureBuilder(feature_set={predictor.feature_set}, "
                  f"use_kmeans={_gcfg['use_kmeans']}) → {_target_dim} 維（與 checkpoint 吻合）")
            break
    if fb is None:
        print(f"[DL] 警告：無法自動匹配 in_features={_target_dim}，"
              f"使用預設 global_cfg（維度可能不符，SHAP 結果僅供參考）")
        fb = FeatureBuilder(feature_set=predictor.feature_set).fit(X_tr_raw)

    X_tf  = fb.transform(X_te_raw)
    col_names = _build_col_names(predictor.feature_set, feature_names, X_tf.shape[1])
    X_df  = pd.DataFrame(X_tf, columns=col_names)

    # PermutationExplainer 跑全量測試集很慢 (O(n × features))，
    # 取隨機子集以控制時間：每筆約 2~3s，200 筆 ≈ 10 min
    if len(X_df) > max_test_samples:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_df), size=max_test_samples, replace=False)
        X_df = X_df.iloc[idx].reset_index(drop=True)
        print(f"[DL] X_test 已取樣 {max_test_samples}/{X_tf.shape[0]} 筆"
              f"（PermutationExplainer 速度限制，可用 --dl-test-samples 調整）")

    out_dir = os.path.join(artifacts_dir, "shap_plots_dl")
    print(f"[DL] 執行 SHAP（{X_df.shape[0]} 筆 × {X_df.shape[1]} 特徵，"
          f"background={min(max_bg, len(X_df))} 筆）...")

    viz = AutoMLVisualizer(model=predictor, X_test=X_df, output_dir=out_dir)
    shap_mat  = viz._get_shap_matrix()
    top_feat  = col_names[int(np.abs(shap_mat).mean(0).argmax())]

    viz.generate_all_plots(
        sample_index=0,
        target_feature=top_feat,
        prefix=f"{dataset_name}_dl",
    )
    print(f"[DL] 圖表已輸出 → {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="從現有 artifacts 產生 SHAP 視覺化（不重新訓練）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset",       required=True,
                        help="資料集 CSV 路徑（用於重建 train/test split）")
    parser.add_argument("--target",        required=True,
                        help="目標欄位名稱")
    parser.add_argument("--model",         choices=["tabular", "dl", "both"],
                        default="tabular",
                        help="使用哪個模型產生 SHAP（預設 tabular）")
    parser.add_argument("--artifacts",     default=None,
                        help="artifacts 目錄（預設 artifacts/submission）")
    parser.add_argument("--test-size",     type=float, default=0.2)
    parser.add_argument("--split-seed",    type=int, default=SEED)
    parser.add_argument("--feature-set",   default=None,
                        help="tabular 模型的 feature-set（當 _fb.pkl 不存在時使用，預設 raw）")
    parser.add_argument("--dl-bg",          type=int, default=100,
                        help="DL PermutationExplainer 的 background 樣本數（預設 100）")
    parser.add_argument("--dl-test-samples", type=int, default=200,
                        help="DL SHAP 使用的 X_test 樣本數上限（預設 200，避免 PermutationExplainer 耗時過長）")
    args = parser.parse_args()

    # ── 載入資料 ──────────────────────────────────────────────────────────────
    ds_path = (args.dataset if os.path.isabs(args.dataset)
               else os.path.join(HERE, args.dataset))
    if not os.path.exists(ds_path):
        print(f"[Error] 找不到資料集：{ds_path}"); sys.exit(1)

    df = pd.read_csv(ds_path)
    if args.target not in df.columns:
        print(f"[Error] 找不到目標欄 '{args.target}'，可用欄位：{list(df.columns)}"); sys.exit(1)

    task = _auto_detect_task(df[args.target])
    if task != "classification":
        print(f"[Warning] 偵測到回歸任務（task={task}）。SHAP 圖表目前只針對分類最佳化，仍繼續執行。")

    y = df[args.target].values
    # 優先使用與 pipeline 相同的雙軌前處理，確保特徵數與訓練時一致
    _adv_ok = False
    try:
        from preprocessing.interface import preprocess_for_training
        _Xd_tr, _Xd_te, _, _, _ = preprocess_for_training(
            df, args.target, test_size=args.test_size
        )
        X_tr = _Xd_tr["tree"].values.astype(np.float32)
        X_te = _Xd_te["tree"].values.astype(np.float32)
        feature_names = list(_Xd_tr["tree"].columns)
        _adv_ok = True
    except Exception as _adv_err:
        print(f"[Preprocess] 雙軌前處理失敗: {_adv_err}")

    if not _adv_ok:
        X_raw, feature_names = _prepare_X_raw(df, args.target)
        try:
            X_tr, X_te, _, _ = train_test_split(
                X_raw, y, test_size=args.test_size,
                random_state=args.split_seed, stratify=y)
        except ValueError:
            X_tr, X_te, _, _ = train_test_split(
                X_raw, y, test_size=args.test_size, random_state=args.split_seed)

    print(f"[Load] shape={df.shape}  target={args.target}  task={task}")
    print(f"       features={len(feature_names)}  → {feature_names[:5]}{'...' if len(feature_names)>5 else ''}")
    print(f"[Split] train={len(X_tr)}  test={len(X_te)}  seed={args.split_seed}")

    artifacts_dir = (args.artifacts if args.artifacts
                     else os.path.join(HERE, "artifacts", "submission"))
    dataset_name  = os.path.splitext(os.path.basename(ds_path))[0]

    # ── 產生圖表 ──────────────────────────────────────────────────────────────
    if args.model in ("tabular", "both"):
        print(f"\n{'─'*55}\n  Tabular SHAP\n{'─'*55}")
        run_tabular_viz(artifacts_dir, X_tr, X_te, feature_names,
                        dataset_name, args.feature_set)

    if args.model in ("dl", "both"):
        print(f"\n{'─'*55}\n  DL SHAP（Transformer）\n{'─'*55}")
        run_dl_viz(artifacts_dir, X_tr, X_te, feature_names,
                   dataset_name, max_bg=args.dl_bg,
                   max_test_samples=args.dl_test_samples)

    print(f"\n{'='*55}")
    print("  完成！")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
