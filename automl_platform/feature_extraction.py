"""
特徵工程模組（GPU 加速版）
涵蓋以下技術：
    - MI 特徵選擇（sklearn 計算分數，GPU 排序）
    - 多項式交互特徵（PyTorch 廣播，無 sklearn 依賴）
    - 群組聚合特徵（PyTorch scatter_reduce GPU 聚合）
    - 時序統計特徵（全程 PyTorch GPU 運算）
    - 動態數值縮放（Z-score，PyTorch GPU 運算）
    - SHAP 重要性剪枝（XGBoost GPU 優先，fallback sklearn CPU）
"""
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import shap

warnings.filterwarnings("ignore")

# ── GPU 設備偵測 ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[FeatureExtraction] Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[FeatureExtraction] GPU: {torch.cuda.get_device_name(0)}")


def _to_tensor(data, dtype=torch.float32) -> torch.Tensor:
    """將 numpy array / DataFrame / Series 轉為 GPU Tensor。"""
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.values
    return torch.as_tensor(np.asarray(data), dtype=dtype, device=DEVICE)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


# ── 各子模組 ──────────────────────────────────────────────────────────────────

class MIFeatureSelector:
    """以互資訊（Mutual Information）排名，保留分數前 k 名的欄位。
    分數計算使用 sklearn（CPU），argsort 移至 GPU 加速。"""

    def __init__(self, k=50, task="classification"):
        self.k = k
        self.task = task
        self.selected_cols_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MIFeatureSelector":
        fn = mutual_info_classif if self.task == "classification" else mutual_info_regression
        scores = fn(X.fillna(0), y, random_state=42)
        k = min(self.k, X.shape[1])
        idx = _to_numpy(torch.argsort(_to_tensor(scores), descending=True)[:k]).astype(int)
        self.selected_cols_ = X.columns[idx].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_]


class PolynomialInteractionGenerator:
    """對最重要前 N 欄，用 PyTorch 廣播一次算出所有兩兩交互項。
    取代 sklearn PolynomialFeatures，計算全程在 GPU 上完成。"""

    def __init__(self, max_input_cols=20):
        self.max_input_cols = max_input_cols
        self.input_cols_ = None
        self.feature_names_ = None
        self._i_idx = None  # triu 索引，fit 時計算一次，transform 時重複使用
        self._j_idx = None

    def fit(self, X: pd.DataFrame) -> "PolynomialInteractionGenerator":
        cols = X.columns[: self.max_input_cols].tolist()
        self.input_cols_ = cols
        C = len(cols)
        self._i_idx, self._j_idx = torch.triu_indices(C, C, offset=1, device=DEVICE)
        i_np = _to_numpy(self._i_idx).astype(int)
        j_np = _to_numpy(self._j_idx).astype(int)
        self.feature_names_ = [f"poly_{cols[i]}_{cols[j]}" for i, j in zip(i_np, j_np)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = _to_tensor(X[self.input_cols_].fillna(0))          # [N, C]
        # [N, C, 1] * [N, 1, C] → [N, C, C]，取上三角 (i < j)
        outer = arr.unsqueeze(2) * arr.unsqueeze(1)
        result = outer[:, self._i_idx, self._j_idx]              # [N, num_pairs]
        return pd.DataFrame(_to_numpy(result), columns=self.feature_names_, index=X.index)


class GroupAggregationFeatures:
    """對類別欄依群組做 mean / std / min / max 聚合。
    使用 torch.scatter_add_ 與 scatter_reduce_ 在 GPU 上完成聚合。"""

    def __init__(self, max_unique_ratio=0.05, agg_funcs=("mean", "std", "min", "max")):
        self.max_unique_ratio = max_unique_ratio
        self.agg_funcs = agg_funcs
        self.group_cols_ = []
        self.numeric_cols_ = []
        self.agg_stats_ = {}   # {group_col: {numeric_col: {fn: pd.Series}}}

    def fit(self, X: pd.DataFrame, y=None) -> "GroupAggregationFeatures":
        n = len(X)
        self.group_cols_ = [
            c for c in X.select_dtypes(include=["object", "category"]).columns
            if 1 < X[c].nunique() <= max(2, int(n * self.max_unique_ratio))
        ]
        self.numeric_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        for gc in self.group_cols_:
            cat = X[gc].astype("category")
            cat_codes = torch.tensor(cat.cat.codes.values, dtype=torch.long, device=DEVICE)
            n_cats = int(cat_codes.max().item()) + 1
            cat_names = cat.cat.categories.tolist()
            stats = {}

            for nc in self.numeric_cols_[:10]:
                vals = _to_tensor(X[nc].fillna(0).values)
                nc_stats = {}

                # count 與 sum（std 也依賴 mean，故無論 agg_funcs 為何都先算）
                counts = torch.zeros(n_cats, device=DEVICE)
                counts.scatter_add_(0, cat_codes, torch.ones(len(vals), device=DEVICE))
                safe_counts = counts.clamp(min=1)

                sums = torch.zeros(n_cats, device=DEVICE)
                sums.scatter_add_(0, cat_codes, vals)
                means_gpu = sums / safe_counts

                if "mean" in self.agg_funcs:
                    nc_stats["mean"] = pd.Series(_to_numpy(means_gpu), index=cat_names)

                if "std" in self.agg_funcs:
                    sq_diff = (vals - means_gpu[cat_codes]) ** 2
                    var_sum = torch.zeros(n_cats, device=DEVICE)
                    var_sum.scatter_add_(0, cat_codes, sq_diff)
                    nc_stats["std"] = pd.Series(
                        _to_numpy(torch.sqrt(var_sum / safe_counts)), index=cat_names
                    )

                if "min" in self.agg_funcs:
                    mins = torch.full((n_cats,), float("inf"), device=DEVICE)
                    mins.scatter_reduce_(0, cat_codes, vals, reduce="amin", include_self=True)
                    nc_stats["min"] = pd.Series(_to_numpy(mins), index=cat_names)

                if "max" in self.agg_funcs:
                    maxs = torch.full((n_cats,), float("-inf"), device=DEVICE)
                    maxs.scatter_reduce_(0, cat_codes, vals, reduce="amax", include_self=True)
                    nc_stats["max"] = pd.Series(_to_numpy(maxs), index=cat_names)

                stats[nc] = nc_stats
            self.agg_stats_[gc] = stats
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new_cols = {}
        for gc in self.group_cols_:
            for nc, nc_stats in self.agg_stats_[gc].items():
                for fn, stat_series in nc_stats.items():
                    new_cols[f"grp_{gc}_{nc}_{fn}"] = X[gc].map(stat_series)
        if new_cols:
            return pd.DataFrame(new_cols, index=X.index)
        return pd.DataFrame(index=X.index)


class TimeSeriesFeatureExtractor:
    """針對「每列即為一條完整時序」的資料集，全程在 GPU 上抽取統計特徵。"""

    def fit(self, X: pd.DataFrame) -> "TimeSeriesFeatureExtractor":  # noqa: ARG002
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        arr = _to_tensor(X)   # [N, T]
        features = {}

        mean = arr.mean(dim=1)
        std  = arr.std(dim=1) + 1e-8

        features["ts_mean"]  = _to_numpy(mean)
        features["ts_std"]   = _to_numpy(std)
        features["ts_min"]   = _to_numpy(arr.min(dim=1).values)
        features["ts_max"]   = _to_numpy(arr.max(dim=1).values)
        features["ts_range"] = features["ts_max"] - features["ts_min"]

        q25 = torch.quantile(arr, 0.25, dim=1)
        q75 = torch.quantile(arr, 0.75, dim=1)
        features["ts_q25"] = _to_numpy(q25)
        features["ts_q75"] = _to_numpy(q75)
        features["ts_iqr"]  = _to_numpy(q75 - q25)

        centered = arr - mean.unsqueeze(1)
        features["ts_skew"] = _to_numpy((centered ** 3).mean(dim=1) / (std ** 3))
        features["ts_kurt"] = _to_numpy((centered ** 4).mean(dim=1) / (std ** 4) - 3)

        diff = torch.diff(arr, dim=1)
        features["ts_diff_mean"] = _to_numpy(diff.abs().mean(dim=1))
        features["ts_diff_std"]  = _to_numpy(diff.std(dim=1))
        features["ts_diff_max"]  = _to_numpy(diff.abs().max(dim=1).values)

        if arr.shape[1] > 2:
            a = arr[:, :-1] - mean.unsqueeze(1)
            b = arr[:, 1:]  - mean.unsqueeze(1)
            features["ts_autocorr_lag1"] = _to_numpy((a * b).mean(dim=1) / (std ** 2))

        n_pts   = arr.shape[1]
        quarter = n_pts // 4
        if quarter > 0:
            for q in range(4):
                seg = arr[:, q * quarter : (q + 1) * quarter]
                features[f"ts_q{q+1}_mean"] = _to_numpy(seg.mean(dim=1))

        return pd.DataFrame(features, index=X.index)


class DynamicNumericalScaler:
    """Z-score 標準化，mean / std 以 GPU Tensor 儲存，transform 全在 GPU 上完成。"""

    def __init__(self, use_quantile=True):
        self.use_quantile = use_quantile   # 保留介面相容，目前套用 Z-score
        self.mean_  = None
        self.std_   = None
        self.cols_  = None

    def fit(self, X: pd.DataFrame) -> "DynamicNumericalScaler":
        self.cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        arr = _to_tensor(X[self.cols_].fillna(0))
        self.mean_ = arr.mean(dim=0)
        self.std_  = arr.std(dim=0).clamp(min=1e-8)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        arr = _to_tensor(X[self.cols_].fillna(0))
        result[self.cols_] = _to_numpy((arr - self.mean_) / self.std_)
        return result


class SHAPFeaturePruner:
    """用樹模型 SHAP 重要性剪枝。
    優先使用 XGBoost GPU（device='cuda'），若未安裝或失敗則 fallback 至 sklearn GBDT（CPU）。"""

    def __init__(self, task="classification", min_features=5, importance_threshold=0.001):
        self.task = task
        self.min_features = min_features
        self.importance_threshold = importance_threshold
        self.selected_cols_ = None

    def _build_model(self):
        try:
            import xgboost as xgb
            kwargs = {"device": "cuda"} if DEVICE.type == "cuda" else {}
            if self.task == "classification":
                return xgb.XGBClassifier(
                    n_estimators=50, max_depth=3, random_state=42,
                    eval_metric="logloss", verbosity=0, **kwargs
                )
            return xgb.XGBRegressor(
                n_estimators=50, max_depth=3, random_state=42, verbosity=0, **kwargs
            )
        except Exception:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            if self.task == "classification":
                return GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
            return GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SHAPFeaturePruner":
        cols  = X.columns.tolist()
        X_arr = X.fillna(0).values
        model = self._build_model()
        model.fit(X_arr, y)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_arr[: min(200, len(X_arr))])

        if isinstance(shap_vals, list):
            importance = np.abs(np.array(shap_vals)).mean(axis=(0, 1))
        else:
            importance = np.abs(shap_vals).mean(axis=0)

        # GPU 上做正規化與排序
        imp_t = _to_tensor(importance)
        if imp_t.max() > 0:
            imp_t = imp_t / imp_t.max()

        importance_norm = _to_numpy(imp_t)
        keep_mask = importance_norm >= self.importance_threshold

        if keep_mask.sum() < self.min_features:
            top_idx = _to_numpy(
                torch.argsort(imp_t, descending=True)[: self.min_features]
            ).astype(int)
            keep_mask = np.zeros(len(cols), dtype=bool)
            keep_mask[top_idx] = True

        self.selected_cols_ = [c for c, k in zip(cols, keep_mask) if k]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_cols_]


# ── 主流程協調器 ──────────────────────────────────────────────────────────────

class FeatureExtractionPipeline:
    """
    特徵工程主流程協調器（GPU 加速版）。
    一般表格資料：MI 選擇 → 多項式交互 → 群組聚合 → 動態縮放 → SHAP 剪枝
    時序資料：           TS 統計特徵 →            動態縮放 → SHAP 剪枝（可關閉）
    """

    def __init__(self, task="classification", is_timeseries=False,
                 mi_k=50, poly_max_cols=15, use_shap_pruning=True):
        self.task = task
        self.is_timeseries = is_timeseries
        self.mi_k = mi_k
        self.poly_max_cols = poly_max_cols
        self.use_shap_pruning = use_shap_pruning

        self.mi_selector_  = None
        self.poly_gen_     = None
        self.group_agg_    = None
        self.ts_extractor_ = None
        self.scaler_       = None
        self.shap_pruner_  = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        print(f"  [FE] Input shape: {X.shape}")

        if self.is_timeseries:
            self.ts_extractor_ = TimeSeriesFeatureExtractor()
            X = self.ts_extractor_.transform(X)
            print(f"  [FE] After TS feature extraction: {X.shape}")
        else:
            numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
            self.mi_selector_ = MIFeatureSelector(k=self.mi_k, task=self.task)
            self.mi_selector_.fit(numeric_X, y)
            X_mi = self.mi_selector_.transform(numeric_X)
            print(f"  [FE] After MI selection: {X_mi.shape}")

            self.group_agg_ = GroupAggregationFeatures()
            self.group_agg_.fit(X, y)
            X_grp = self.group_agg_.transform(X)

            self.poly_gen_ = PolynomialInteractionGenerator(max_input_cols=self.poly_max_cols)
            self.poly_gen_.fit(X_mi)
            X_poly = self.poly_gen_.transform(X_mi)

            parts = [X_mi, X_poly]
            if not X_grp.empty:
                parts.append(X_grp)
            X = pd.concat(parts, axis=1)
            print(f"  [FE] After poly + group aggregation: {X.shape}")

        self.scaler_ = DynamicNumericalScaler()
        self.scaler_.fit(X)
        X = self.scaler_.transform(X)

        if self.use_shap_pruning and X.shape[1] > self.mi_k:
            try:
                self.shap_pruner_ = SHAPFeaturePruner(task=self.task)
                self.shap_pruner_.fit(X, y)
                X = self.shap_pruner_.transform(X)
                print(f"  [FE] After SHAP pruning: {X.shape}")
            except Exception as e:
                print(f"  [FE] SHAP pruning skipped: {e}")
                self.shap_pruner_ = None

        print(f"  [FE] Final feature shape: {X.shape}")
        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.is_timeseries:
            X = self.ts_extractor_.transform(X)
        else:
            numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
            X_mi   = self.mi_selector_.transform(numeric_X)
            X_grp  = self.group_agg_.transform(X)
            X_poly = self.poly_gen_.transform(X_mi)
            parts  = [X_mi, X_poly]
            if not X_grp.empty:
                parts.append(X_grp)
            X = pd.concat(parts, axis=1)

        X = self.scaler_.transform(X)
        if self.shap_pruner_ is not None:
            X = self.shap_pruner_.transform(X)
        return X
