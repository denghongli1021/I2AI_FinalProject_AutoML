"""
特徵工程模組（FeatureBuilder + robust_clean_dataframe）。

支援 6 種特徵集：raw / signal / pca64 / svd64 / raw_stat / raw_stat_fft
嚴格遵守「只在 train fold 上 fit，再 transform val/test」原則，杜絕資料洩漏。
所有前處理數值參數（IQR 倍數、降維維度）皆由外部傳入，不人為固定。

robust_clean_dataframe：在進入 FeatureBuilder 前的防呆清理。
  - 自動偵測 datetime 欄位並拆解為數值特徵
  - 修復混入字串的數值欄位（pd.to_numeric coerce）
  - 刪除常數欄（zero variance）
  - 讓 Pipeline「絕不因為資料格式不完美而崩潰」
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA
from sklearn.cluster import MiniBatchKMeans


# ── 防呆特徵推斷（Phase 1 資料清理）────────────────────────────────────────────

def robust_clean_dataframe(df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
    """
    在進入特徵工程前對 DataFrame 做強健清理，確保不因髒資料崩潰。

    處理順序：
      1. object 欄位嘗試 datetime 解析 → 拆解為 _year/_month/_dayofweek/_dayofyear
      2. 殘餘 object 欄位嘗試 to_numeric(coerce)；多數不可轉換則 Label Encode
      3. 刪除常數欄（nunique==1 或全 NaN）
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c != target_col]

    derived: dict = {}
    drop_cols: list = []

    for col in feature_cols:
        ser = df[col]

        # 非 object 欄位不需要額外推斷
        if ser.dtype != object:
            continue

        # 嘗試 datetime 解析（成功率 > 50% 才視為日期欄）
        try:
            parsed = pd.to_datetime(ser, infer_datetime_format=True, errors="raise")
            derived[f"{col}_year"] = parsed.dt.year.astype("float32")
            derived[f"{col}_month"] = parsed.dt.month.astype("float32")
            derived[f"{col}_dayofweek"] = parsed.dt.dayofweek.astype("float32")
            derived[f"{col}_dayofyear"] = parsed.dt.dayofyear.astype("float32")
            drop_cols.append(col)
            continue
        except Exception:
            pass

        # 嘗試數值轉換（NaN 表示無法轉換）
        numeric = pd.to_numeric(ser, errors="coerce")
        if numeric.notna().mean() >= 0.5:
            df[col] = numeric
        else:
            # Label encode：-1 保留給未知類別
            df[col] = pd.Categorical(ser).codes.astype("float32").replace(-1, np.nan)

    # 插入日期衍生欄
    for new_col, values in derived.items():
        df[new_col] = values

    # 刪除原始 datetime 欄
    df = df.drop(columns=drop_cols, errors="ignore")

    # 刪除常數欄（不包含目標欄）
    const_cols = [
        c for c in df.columns
        if c != target_col and df[c].nunique(dropna=False) <= 1
    ]
    if const_cols:
        df = df.drop(columns=const_cols)

    return df

FEATURE_SETS = ["raw", "signal", "pca64", "svd64", "kpca32", "raw_stat", "raw_stat_fft", "poly2",
                "ts_tabular", "ts_tabular_fft"]

# 各模型類型可使用的特徵集候選（供 HPO 用）
# poly2: Top-30 特徵的 degree-2 交互項，補捉非線性關係；移除對表格資料意義不大的 raw_stat_fft
TABULAR_FEATURE_SETS = ["raw", "raw_stat", "poly2"]
# CatBoost 對稱樹 + CTR 已自動學特徵交互，poly2 是重複計算且寬特徵下會 bad_alloc
CATBOOST_FEATURE_SETS = ["raw", "raw_stat"]
LINEAR_FEATURE_SETS = ["pca64", "svd64", "kpca32"]
MLP_FEATURE_SETS = ["raw", "raw_stat", "raw_stat_fft", "signal"]
CNN_FEATURE_SETS = ["raw", "signal"]
TRANSFORMER_FEATURE_SETS = ["raw", "raw_stat_fft", "signal"]

# 時序模式：per-row 視窗特徵（差分 + Lag + Rolling），DL 模型直接使用 raw/signal
TS_TABULAR_FEATURE_SETS = ["ts_tabular", "ts_tabular_fft"]  # 包含 per-row 時序視窗特徵
TS_DL_FEATURE_SETS = ["raw", "signal"]


class FeatureBuilder:
    """
    前處理 + 特徵工程。
    fit 僅使用訓練資料，transform 套用相同統計量到 val/test。

    Parameters
    ----------
    feature_set : str
        六種特徵集之一。
    n_components : int
        PCA/SVD 目標維度（僅 pca64/svd64 使用）。
    iqr_factor : float
        IQR 截斷倍數（clip = Q1 - k*IQR … Q3 + k*IQR）。
    n_segments : int
        raw_stat 局部統計的分段數。
    """

    def __init__(
        self,
        feature_set: str = "raw_stat_fft",
        n_components: int = 64,
        iqr_factor: float = 3.0,
        n_segments: int = 8,
        global_cfg: dict = None,
    ):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.feature_set = feature_set
        self.n_components = n_components
        self.iqr_factor = iqr_factor
        self.n_segments = n_segments
        self.global_cfg = global_cfg or {}

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FeatureBuilder":
        X = np.asarray(X, dtype=np.float32)

        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X)

        q1 = np.percentile(X_imp, 25, axis=0)
        q3 = np.percentile(X_imp, 75, axis=0)
        iqr = q3 - q1
        self.clip_low_ = q1 - self.iqr_factor * iqr
        self.clip_high_ = q3 + self.iqr_factor * iqr
        X_clipped = np.clip(X_imp, self.clip_low_, self.clip_high_)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_clipped)

        if self.feature_set == "pca64":
            nc = min(self.n_components, X_scaled.shape[1], X_scaled.shape[0] - 1)
            self.reducer_ = PCA(n_components=nc, random_state=42)
            self.reducer_.fit(X_scaled)
        elif self.feature_set == "svd64":
            nc = min(self.n_components, X_scaled.shape[1])
            self.reducer_ = TruncatedSVD(n_components=nc, random_state=42)
            self.reducer_.fit(X_scaled)
        elif self.feature_set == "kpca32" and self.global_cfg.get("use_kpca", False):
            nc = min(32, X_scaled.shape[1])
            self.reducer_ = KernelPCA(n_components=nc, kernel="rbf", random_state=42, n_jobs=-1)
            self.reducer_.fit(X_scaled)
        elif self.feature_set == "poly2":
            from sklearn.preprocessing import PolynomialFeatures
            # 自適應上限：確保 (nf + interactions) × n_samples ≤ 50M 個元素（約 200 MB/fold）
            # 先扣掉原始特徵佔的配額，剩餘才給交互項
            n_top = min(30, X_scaled.shape[1])
            budget_per_sample = max(10, int(50_000_000 / max(1, X_scaled.shape[0])))
            max_inter = max(5, budget_per_sample - X_scaled.shape[1])
            while n_top > 2 and n_top * (n_top - 1) // 2 > max_inter:
                n_top -= 1
            self._poly2_idx = np.argsort(X_scaled.var(axis=0))[-n_top:]
            self._poly2 = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            self._poly2.fit(X_scaled[:, self._poly2_idx])
            self.reducer_ = None
        else:
            self.reducer_ = None

        if self.global_cfg.get("use_kmeans", False) and self.feature_set in ["raw_stat", "raw_stat_fft"]:
<<<<<<< HEAD
            n_samples = X_scaled.shape[0]
            valid_ks = sorted(set(min(k, n_samples) for k in [15, 30]))
            self.kmeans_ = [
                MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init="auto")
                for k in valid_ks
=======
            self.kmeans_ = [
                MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, n_init="auto")
                for k in [15, 30]
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
            ]
            for km in self.kmeans_:
                km.fit(X_scaled)
        else:
            self.kmeans_ = []

        self.in_features_ = X_scaled.shape[1]
        return self

    # ------------------------------------------------------------------
    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_imp = self.imputer_.transform(X)
        X_clipped = np.clip(X_imp, self.clip_low_, self.clip_high_)
        X_scaled = self.scaler_.transform(X_clipped)

        fs = self.feature_set
        if fs == "raw":
            return X_scaled.astype(np.float32)
        if fs == "signal":
            row_norm = X_scaled / (
                np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8
            )
            return np.hstack([X_scaled, row_norm]).astype(np.float32)
        if fs == "pca64":
            return self.reducer_.transform(X_scaled).astype(np.float32)
        if fs == "svd64":
            return self.reducer_.transform(X_scaled).astype(np.float32)
        if fs == "kpca32":
            if self.reducer_ is None:
                return X_scaled.astype(np.float32)
            return self.reducer_.transform(X_scaled).astype(np.float32)
        if fs == "poly2":
            X_top = X_scaled[:, self._poly2_idx]
            inter_all = self._poly2.transform(X_top)
            # PolynomialFeatures(interaction_only) 輸出 = [original_top, cross_terms]
            # 只保留交互項部分，原始特徵已在 X_scaled 中
            interactions = inter_all[:, X_top.shape[1]:]
            return np.hstack([X_scaled, interactions]).astype(np.float32)

        if fs == "ts_tabular":
            ts_win = self._ts_window_features(X_scaled)
            stat   = self._stat_features(X_scaled)
            return np.hstack([X_scaled, ts_win, stat]).astype(np.float32)
        if fs == "ts_tabular_fft":
            ts_win = self._ts_window_features(X_scaled)
            fft    = self._fft_features(X_scaled)
            stat   = self._stat_features(X_scaled)
            return np.hstack([X_scaled, ts_win, fft, stat]).astype(np.float32)

        kmeans_feats = []
        for km in getattr(self, "kmeans_", []):
            kmeans_feats.append(km.transform(X_scaled))
        km_feat = np.hstack(kmeans_feats) if kmeans_feats else np.zeros((X_scaled.shape[0], 0))

        if fs == "raw_stat":
            return np.hstack(
                [X_scaled, self._stat_features(X_scaled), km_feat]
            ).astype(np.float32)
        if fs == "raw_stat_fft":
            stat = self._stat_features(X_scaled)
            fft = self._fft_features(X_scaled)
            return np.hstack([X_scaled, stat, fft, km_feat]).astype(np.float32)
        raise ValueError(f"Unknown feature_set: {fs}")

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    def _stat_features(self, X: np.ndarray) -> np.ndarray:
        """全域統計 + n_segments*2 維局部統計。"""
        zcr = ((X[:, :-1] * X[:, 1:]) < 0).sum(axis=1) / max(1, X.shape[1])
        energy = (X**2).sum(axis=1)
        rms = np.sqrt(energy / max(1, X.shape[1]))
        mean_abs_diff = np.abs(np.diff(X, axis=1)).mean(axis=1)

        global_stats = np.column_stack([
            X.mean(axis=1),
            X.std(axis=1),
            np.percentile(X, 5, axis=1),
            np.percentile(X, 25, axis=1),
            np.percentile(X, 75, axis=1),
            np.percentile(X, 95, axis=1),
            X.min(axis=1),
            X.max(axis=1),
            stats.skew(X, axis=1),
            stats.kurtosis(X, axis=1),
            np.median(X, axis=1),
            np.abs(X).mean(axis=1),
            np.abs(X).max(axis=1),
            (X > 0).mean(axis=1).astype(np.float32),
            np.ptp(X, axis=1),
            np.var(X, axis=1),
            energy,
            rms,
            mean_abs_diff,
            zcr,
        ])

        n = X.shape[1]
        seg_size = max(1, n // self.n_segments)
        local_means, local_stds, local_maxs, local_mins = [], [], [], []
        for i in range(self.n_segments):
            seg = X[:, i * seg_size: (i + 1) * seg_size]
            if seg.shape[1] > 0:
                local_means.append(seg.mean(axis=1, keepdims=True))
                local_stds.append(seg.std(axis=1, keepdims=True))
                local_maxs.append(seg.max(axis=1, keepdims=True))
                local_mins.append(seg.min(axis=1, keepdims=True))

        local_stats = np.hstack(local_means + local_stds + local_maxs + local_mins)  # n_segments*4 特徵
        return np.hstack([global_stats, local_stats])

    # ------------------------------------------------------------------
    def _ts_window_features(self, X: np.ndarray) -> np.ndarray:
        """Per-row 時序視窗特徵（因果特徵，無未來洩漏）。

        每一列（row）視為一條時序序列，計算：
          - 一階差分（X[t] - X[t-1]，第 0 列填 0）
          - Lag=1 特徵（前移一步，首列用 0 填充）
          - Lag=2 特徵（前移二步，前兩列用 0 填充）
          - Rolling Mean window=3, 5（前向平均，邊界用已有資料）
          - Rolling Std  window=3, 5（前向標準差，邊界用已有資料）

        Parameters
        ----------
        X : np.ndarray, shape [N, F]
            已 scale 的特徵矩陣（每列為一個樣本/時間步的特徵向量）。

        Returns
        -------
        np.ndarray, shape [N, F * 13]
            diff + lag1 + lag2 + (mean/std/max/min) × 2 windows + EMA × 2 = 13 × F
        """
        X = np.asarray(X, dtype=np.float32)
        n, f = X.shape
        parts = []

        # 一階差分（首列填 0）
        diff = np.zeros_like(X)
        diff[1:] = X[1:] - X[:-1]
        parts.append(diff)

        # Lag=1（前移一步，首列填 0）
        lag1 = np.zeros_like(X)
        lag1[1:] = X[:-1]
        parts.append(lag1)

        # Lag=2（前移兩步，前兩列填 0）
        lag2 = np.zeros_like(X)
        lag2[2:] = X[:-2]
        parts.append(lag2)

        # Rolling Mean / Std / Max / Min（window=3, 5）
        df_tmp = pd.DataFrame(X)
        for w in (3, 5):
            roll = df_tmp.rolling(window=w, min_periods=1)
            parts.append(roll.mean().values.astype(np.float32))
            parts.append(roll.std(ddof=0).fillna(0.0).values.astype(np.float32))
            parts.append(roll.max().values.astype(np.float32))
            parts.append(roll.min().values.astype(np.float32))

        # EMA（span=3, 5）
        for span in (3, 5):
            parts.append(df_tmp.ewm(span=span, adjust=False).mean().values.astype(np.float32))

        return np.hstack(parts)

    # ------------------------------------------------------------------
    def _fft_features(self, X: np.ndarray) -> np.ndarray:
        """FFT 頻域特徵。"""
        fft_abs = np.abs(np.fft.rfft(X, axis=1)).astype(np.float64)
        n_fft = fft_abs.shape[1]
        total_energy = (fft_abs ** 2).sum(axis=1, keepdims=True) + 1e-8

        feat = [
            fft_abs.mean(axis=1),
            fft_abs.std(axis=1),
            fft_abs.max(axis=1),
            fft_abs.argmax(axis=1).astype(float),
        ]

        # 8 頻帶能量占比
        band_size = max(1, n_fft // 8)
        for i in range(8):
            band = fft_abs[:, i * band_size: (i + 1) * band_size]
            if band.shape[1] > 0:
                feat.append((band ** 2).sum(axis=1) / total_energy.squeeze())

        # 頻譜重心 / 展寬 / 滾降點
        freqs = np.arange(n_fft, dtype=np.float64)
        power = fft_abs ** 2 + 1e-8
        total_p = power.sum(axis=1, keepdims=True) + 1e-8
        centroid = (power * freqs).sum(axis=1) / total_p.squeeze()
        spread = np.sqrt(
            (power * (freqs - centroid[:, None]) ** 2).sum(axis=1) / total_p.squeeze()
        )
        cumulative = np.cumsum(power, axis=1)
        rolloff = np.argmax(
            cumulative >= 0.85 * cumulative[:, -1:], axis=1
        ).astype(float)
        feat += [centroid, spread, rolloff]

        # 頻譜熵
        prob = power / total_p
        entropy = -(prob * np.log(prob + 1e-10)).sum(axis=1)
        feat.append(entropy)

        # Top-20 最大振幅值（捕捉主要頻率成分）
        top_k = min(20, n_fft)
        top20_vals = np.sort(fft_abs, axis=1)[:, -top_k:]
        feat.append(np.log1p(top20_vals).astype(np.float32))

        result = np.column_stack(feat).astype(np.float32)
        return result


# ── 時序特徵引擎（Phase 2 擴充）────────────────────────────────────────────────

class TSFeatureBuilder:
    """
    時序專屬特徵引擎：為「列即時間步」的資料集生成因果（Causal）特徵。

    生成特徵類別：
      - 原始特徵（Raw）
      - 滯後特徵（Lag）：X[t-1], X[t-2], X[t-3], X[t-7]
      - 滾動統計（Rolling）：均值、標準差（Window = 3, 5, 10, 20）
      - 動能指標（Momentum）：短期均線 − 長期均線（黃金/死亡交叉訊號）
      - 一階差分（First Diff）：X[t] − X[t-1]

    使用方式（須在 CV 切割前呼叫 compute_features，確保 lag 特徵有正確上下文）：
        X_ts_all = TSFeatureBuilder().compute_features(np.vstack([X_train, X_test]))
        X_ts     = X_ts_all[:len(X_train)]
        X_ts_test= X_ts_all[len(X_train):]
    """

    def __init__(
        self,
        lags: tuple = (1, 2, 3, 7),
        windows: tuple = (3, 5, 10, 20),
    ):
        self.lags = lags
        self.windows = windows

    def compute_features(self, X: np.ndarray) -> np.ndarray:
        """
        在完整有序陣列 X（列為時間步）上計算因果特徵。
        每列第 t 的特徵只使用 t 及 t 之前的資料，無未來洩漏。

        Parameters
        ----------
        X : np.ndarray, shape [N, F]
            完整有序資料集（訓練 + 測試拼接）

        Returns
        -------
        np.ndarray, shape [N, F * (1 + len(lags) + 2*len(windows) + 2)]
        """
        X = np.asarray(X, dtype=np.float32)
        n, f = X.shape
        parts = [X]

        # 滯後特徵：X[t-lag]（前 lag 列用第一列填充）
        for lag in self.lags:
            lagged = np.empty_like(X)
            lagged[:lag] = X[0:1]       # forward fill
            lagged[lag:] = X[:-lag]
            parts.append(lagged)

        # 滾動統計（使用 pandas 高效計算）
        df = pd.DataFrame(X)
        for w in self.windows:
            roll = df.rolling(window=w, min_periods=1)
            roll_mean = roll.mean().values.astype(np.float32)
            roll_std  = roll.std(ddof=0).fillna(0.0).values.astype(np.float32)
            parts.append(roll_mean)
            parts.append(roll_std)

        # 動能指標：短期均線 − 長期均線
        if len(self.windows) >= 2:
            short_w = self.windows[0]
            long_w  = self.windows[-1]
            short_ma = df.rolling(window=short_w, min_periods=1).mean().values.astype(np.float32)
            long_ma  = df.rolling(window=long_w,  min_periods=1).mean().values.astype(np.float32)
            momentum = short_ma - long_ma
            parts.append(momentum)

        # 一階差分（X[t] − X[t-1]，第 0 列填 0）
        diff = np.zeros_like(X)
        diff[1:] = X[1:] - X[:-1]
        parts.append(diff)

        return np.hstack(parts)
