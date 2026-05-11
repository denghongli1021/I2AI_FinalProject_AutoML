"""
特徵工程模組（FeatureBuilder）。

支援 6 種特徵集：raw / signal / pca64 / svd64 / raw_stat / raw_stat_fft
嚴格遵守「只在 train fold 上 fit，再 transform val/test」原則，杜絕資料洩漏。
所有前處理數值參數（IQR 倍數、降維維度）皆由外部傳入，不人為固定。
"""
import numpy as np
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD

FEATURE_SETS = ["raw", "signal", "pca64", "svd64", "raw_stat", "raw_stat_fft"]

# 各模型類型可使用的特徵集候選（供 HPO 用）
TABULAR_FEATURE_SETS = ["raw", "raw_stat", "raw_stat_fft"]
LINEAR_FEATURE_SETS = ["pca64", "svd64"]
MLP_FEATURE_SETS = ["raw", "raw_stat", "raw_stat_fft", "signal"]
CNN_FEATURE_SETS = ["raw", "signal"]
TRANSFORMER_FEATURE_SETS = ["raw", "raw_stat_fft", "signal"]


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
    ):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {FEATURE_SETS}")
        self.feature_set = feature_set
        self.n_components = n_components
        self.iqr_factor = iqr_factor
        self.n_segments = n_segments

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
        else:
            self.reducer_ = None

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
        if fs == "raw_stat":
            return np.hstack(
                [X_scaled, self._stat_features(X_scaled)]
            ).astype(np.float32)
        if fs == "raw_stat_fft":
            stat = self._stat_features(X_scaled)
            fft = self._fft_features(X_scaled)
            return np.hstack([X_scaled, stat, fft]).astype(np.float32)
        raise ValueError(f"Unknown feature_set: {fs}")

    # ------------------------------------------------------------------
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    def _stat_features(self, X: np.ndarray) -> np.ndarray:
        """16 維全域統計 + n_segments*2 維局部統計。"""
        global_stats = np.column_stack([
            X.mean(axis=1),
            X.std(axis=1),
            np.percentile(X, 25, axis=1),
            np.percentile(X, 75, axis=1),
            X.min(axis=1),
            X.max(axis=1),
            stats.skew(X, axis=1),
            stats.kurtosis(X, axis=1),
            np.median(X, axis=1),
            np.abs(X).mean(axis=1),
            np.abs(X).max(axis=1),
            (X > 0).mean(axis=1).astype(np.float32),
            np.percentile(X, 10, axis=1),
            np.percentile(X, 90, axis=1),
            np.ptp(X, axis=1),
            np.var(X, axis=1),
        ])  # 16 特徵

        n = X.shape[1]
        seg_size = max(1, n // self.n_segments)
        local_means, local_stds = [], []
        for i in range(self.n_segments):
            seg = X[:, i * seg_size: (i + 1) * seg_size]
            if seg.shape[1] > 0:
                local_means.append(seg.mean(axis=1, keepdims=True))
                local_stds.append(seg.std(axis=1, keepdims=True))

        local_stats = np.hstack(local_means + local_stds)  # n_segments*2 特徵
        return np.hstack([global_stats, local_stats])

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

        result = np.column_stack(feat).astype(np.float32)
        return result
