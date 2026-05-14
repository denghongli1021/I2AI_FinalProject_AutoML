"""
train 模組
==========================
對外只需要實作 `run()`。可以在這個資料夾內任意新增其他 .py。

本檔以 sklearn / xgboost 取代原本 js/ml-engine.js 的純 JS 手刻演算法,
產出格式刻意對齊原本 frontend renderExperimentResults 期望的欄位:
  m.metrics.testR2 / testRMSE / testMAE / testAccuracy / f1 / precision / testScore
  m.featureNames / featureImportance / testTrue / testPred / trainTime / name / type

------------------------------------------------------------
Input:
  df         : pandas.DataFrame
  target     : str
  features   : list[str] | None    # 可包含原始欄位名 + 日期衍生 (例如 "date_month_sin")
  algorithms : list[str]
  options    : dict                # { taskType, timeSeries, testSize }

Output: list[dict]  (一筆 = 一個訓練好的模型)
------------------------------------------------------------
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, Ridge, Lasso,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


# ============================================================
# Algorithm registry — 對應 js MLEngine.ALGORITHMS
# ============================================================
def _algo_factory(key: str, task_type: str, random_state: int = 42):
    # random_state 傳給所有有隨機性的模型,固定種子才能重現結果
    if key == "linear_regression": return LinearRegression()
    if key == "ridge":              return Ridge(alpha=1.0)
    if key == "lasso":              return Lasso(alpha=0.1, max_iter=2000)
    if key == "logistic":           return LogisticRegression(max_iter=300, random_state=random_state)
    if key == "naive_bayes":        return GaussianNB()
    if key == "knn_3":  return KNeighborsClassifier(3) if task_type == "classification" else KNeighborsRegressor(3)
    if key == "knn_5":  return KNeighborsClassifier(5) if task_type == "classification" else KNeighborsRegressor(5)
    if key == "knn_7":  return KNeighborsClassifier(7) if task_type == "classification" else KNeighborsRegressor(7)
    if key == "decision_tree":
        return DecisionTreeClassifier(max_depth=6, random_state=random_state) if task_type == "classification" else DecisionTreeRegressor(max_depth=6, random_state=random_state)
    if key == "random_forest":
        return RandomForestClassifier(n_estimators=20, max_depth=5, n_jobs=-1, random_state=random_state) if task_type == "classification" else RandomForestRegressor(n_estimators=20, max_depth=5, n_jobs=-1, random_state=random_state)
    if key == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=30, max_depth=3, random_state=random_state) if task_type == "classification" else GradientBoostingRegressor(n_estimators=30, max_depth=3, random_state=random_state)
    if key == "xgboost":
        if not _HAS_XGB:
            raise RuntimeError("xgboost 套件未安裝,請執行 pip install xgboost")
        return XGBClassifier(n_estimators=40, max_depth=4, learning_rate=0.1, eval_metric="logloss", verbosity=0, random_state=random_state) if task_type == "classification" else XGBRegressor(n_estimators=40, max_depth=4, learning_rate=0.1, verbosity=0, random_state=random_state)
    if key == "svr":
        if task_type != "regression":
            raise RuntimeError("SVR 僅支援回歸任務")
        return SVR(kernel="rbf", C=1.0, epsilon=0.1)
    raise ValueError(f"未知演算法: {key}")


_ALGO_LABELS = {
    "linear_regression": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "logistic": "Logistic Regression",
    "naive_bayes": "Naive Bayes",
    "knn_3": "KNN (k=3)",
    "knn_5": "KNN (k=5)",
    "knn_7": "KNN (k=7)",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest (20 trees)",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "svr": "SVR",
}

_REGRESSION_ONLY = {"linear_regression", "ridge", "lasso", "svr"}
_CLASSIFICATION_ONLY = {"logistic", "naive_bayes"}


# ============================================================
# Date feature extraction — 對應 js MLEngine._extractDateFeatures
# ============================================================
_DATE_SUFFIXES = ["_year", "_month", "_day_of_year", "_month_sin", "_month_cos", "_day_sin", "_day_cos"]


def _date_feature_value(parsed: pd.Series, suffix: str) -> pd.Series:
    if suffix == "_year":         return parsed.dt.year
    if suffix == "_month":        return parsed.dt.month
    if suffix == "_day_of_year":  return parsed.dt.dayofyear
    if suffix == "_month_sin":    return np.sin(2 * np.pi * parsed.dt.month / 12)
    if suffix == "_month_cos":    return np.cos(2 * np.pi * parsed.dt.month / 12)
    if suffix == "_day_sin":      return np.sin(2 * np.pi * parsed.dt.dayofyear / 365)
    if suffix == "_day_cos":      return np.cos(2 * np.pi * parsed.dt.dayofyear / 365)
    raise ValueError(suffix)


# ============================================================
# Public entry
# ============================================================
def run(
    df: pd.DataFrame,
    target: str,
    features: list[str] | None,
    algorithms: list[str],
    options: dict[str, Any],
    on_progress: Callable[[dict], None] | None = None,
) -> list[tuple[dict[str, Any], Any, StandardScaler, pd.DataFrame]]:
    """
    從「原始 DataFrame」訓練 — 自己做特徵建構 / 切分。
    回傳 list of (bundle, estimator, scaler, X_test_df)。

    on_progress: 可選回呼,接收 {"type": "log"/"progress", ...} 事件,給 SSE streaming 用。
    """
    def _emit(ev):
        if on_progress:
            try: on_progress(ev)
            except Exception: pass
    is_time_series = bool(options.get("timeSeries", False))
    test_size = float(options.get("testSize", 0.2))
    random_state = int(options.get("randomState", 42))
    task_override = options.get("taskType")

    # 1. Build feature matrix (handle date-derived features)
    X_df, feature_names = _build_features(df, target, features)
    y_series = df[target]

    # 2. Drop rows with NaN in X or y
    valid_mask = X_df.notna().all(axis=1) & y_series.notna()
    X_df = X_df[valid_mask].reset_index(drop=True)
    y_series = y_series[valid_mask].reset_index(drop=True)

    if len(X_df) < 10:
        raise ValueError("有效數據不足 10 筆,無法訓練")

    # 3. Detect task type
    if task_override and task_override != "auto":
        task_type = task_override
    else:
        unique_y = y_series.unique()
        task_type = "classification" if len(unique_y) <= 10 else "regression"

    # For classification, ensure y is hashable (cast to str if not numeric)
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y_series):
        y_series = y_series.astype(str)

    # 4. Train/test split
    X = X_df.to_numpy(dtype=float)
    y = y_series.to_numpy()
    if is_time_series:
        split = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, random_state=random_state,
        )

    _emit({"type": "log", "msg": f"資料準備完成: {len(X)} 筆有效數據, {X.shape[1]} 個特徵", "level": "success"})
    _emit({"type": "log", "msg": f"任務類型: {'回歸' if task_type == 'regression' else '分類'}{' (時間序列)' if is_time_series else ''}", "level": "info"})
    _emit({"type": "log", "msg": f"訓練集: {len(y_train)} 筆, 測試集: {len(y_test)} 筆", "level": "info"})

    # 5-8. Standardize + train + sort (與 run_prepared 共用)
    return _train_all(X_train, X_test, y_train, y_test, feature_names, target,
                      task_type, algorithms, _emit, random_state)


def run_prepared(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    y_train_series: pd.Series,
    y_test_series: pd.Series,
    target: str,
    algorithms: list[str],
    options: dict[str, Any],
    on_progress: Callable[[dict], None] | None = None,
) -> list[tuple[dict[str, Any], Any, StandardScaler, pd.DataFrame]]:
    """
    從「已預處理好的 train/test」訓練 — 特徵建構與切分都已由 preprocessing 模組做完。
    X_train_df / X_test_df 已是最終特徵矩陣 (含 OneHot / TF-IDF 展開)。
    回傳格式同 run()。
    """
    def _emit(ev):
        if on_progress:
            try: on_progress(ev)
            except Exception: pass

    feature_names = list(X_train_df.columns)
    X_train = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)
    y_train = y_train_series.to_numpy()
    y_test = y_test_series.to_numpy()

    if len(X_train) < 5:
        raise ValueError("訓練集不足 5 筆,無法訓練")

    # Detect task type
    task_override = options.get("taskType")
    if task_override and task_override != "auto":
        task_type = task_override
    else:
        uniq = set(list(y_train) + list(y_test))
        task_type = "classification" if len(uniq) <= 10 else "regression"

    if task_type == "classification" and not np.issubdtype(np.asarray(y_train).dtype, np.number):
        y_train = np.asarray(y_train).astype(str)
        y_test = np.asarray(y_test).astype(str)

    random_state = int(options.get("randomState", 42))

    _emit({"type": "log", "msg": f"使用已預處理資料: 訓練 {len(X_train)} 筆 / 測試 {len(X_test)} 筆, {X_train.shape[1]} 個特徵", "level": "success"})
    _emit({"type": "log", "msg": f"任務類型: {'回歸' if task_type == 'regression' else '分類'}", "level": "info"})

    return _train_all(X_train, X_test, y_train, y_test, feature_names, target,
                      task_type, algorithms, _emit, random_state)


# ============================================================
# Shared: standardize + train each algorithm + sort  (run / run_prepared 共用)
# ============================================================
def _train_all(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    target: str,
    task_type: str,
    algorithms: list[str],
    _emit: Callable[[dict], None],
    random_state: int = 42,
) -> list[tuple[dict[str, Any], Any, StandardScaler, pd.DataFrame]]:
    # 5. Standardize
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # 6. Per-feature stats for What-If sliders (raw scale)
    X_all = np.vstack([X_train, X_test])
    feature_stats = [
        {"min": float(X_all[:, j].min()), "max": float(X_all[:, j].max()), "mean": float(X_all[:, j].mean())}
        for j in range(X_all.shape[1])
    ]

    # 7. Train each algorithm
    valid_algos = [
        k for k in algorithms
        if k in _ALGO_LABELS
        and not (task_type == "regression" and k in _CLASSIFICATION_ONLY)
        and not (task_type == "classification" and k in _REGRESSION_ONLY)
    ]

    # X_test 標準化後 DataFrame — 給 SHAP visualizer 用 (跟所有模型共用)
    X_test_df = pd.DataFrame(X_test_norm, columns=feature_names)

    results: list[tuple[dict[str, Any], Any, StandardScaler, pd.DataFrame]] = []
    for i, key in enumerate(valid_algos):
        _emit({"type": "progress", "pct": int(i / max(len(valid_algos), 1) * 100),
               "step": f"訓練 {_ALGO_LABELS[key]}"})
        _emit({"type": "log", "msg": f"開始訓練 {_ALGO_LABELS[key]}...", "level": "info"})
        try:
            bundle, estimator = _train_one(
                key, task_type, X_train_norm, X_test_norm, y_train, y_test,
                feature_names, target, feature_stats, scaler, random_state,
            )
            results.append((bundle, estimator, scaler, X_test_df))
            score = bundle["metrics"].get("testScore", 0.0)
            score_label = "R²" if task_type == "regression" else "Acc"
            _emit({"type": "log",
                   "msg": f"✓ {_ALGO_LABELS[key]} 完成 — {score_label}={score:.4f}, {bundle['trainTime']:.0f}ms",
                   "level": "success" if score > 0.5 else "warning"})
        except Exception as e:
            bundle = _failed_bundle(key, task_type, target, feature_names, str(e))
            results.append((bundle, None, scaler, X_test_df))
            _emit({"type": "log", "msg": f"✗ {_ALGO_LABELS[key]} 失敗: {e}", "level": "error"})

    # 8. Sort by testScore desc
    results.sort(key=lambda r: r[0]["metrics"].get("testScore", 0.0), reverse=True)
    _emit({"type": "progress", "pct": 100, "step": "完成"})
    return results


# ============================================================
# Train a single algorithm
# ============================================================
def _train_one(
    key: str,
    task_type: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    target: str,
    feature_stats: list[dict[str, float]],
    scaler: StandardScaler,
    random_state: int = 42,
) -> tuple[dict[str, Any], Any]:
    t0 = time.perf_counter()
    estimator = _algo_factory(key, task_type, random_state)
    estimator.fit(X_train, y_train)

    train_pred = estimator.predict(X_train)
    # 量測測試集推論時間 → 換算每筆樣本的平均延遲 (ms)
    t_pred = time.perf_counter()
    test_pred = estimator.predict(X_test)
    pred_elapsed_ms = (time.perf_counter() - t_pred) * 1000
    train_time_ms = (time.perf_counter() - t0) * 1000
    infer_latency_ms = pred_elapsed_ms / max(len(X_test), 1)

    metrics = (
        _regression_metrics(y_test, test_pred, y_train, train_pred)
        if task_type == "regression"
        else _classification_metrics(y_test, test_pred, y_train, train_pred)
    )

    importance = _feature_importance(estimator, X_test, y_test, len(feature_names))

    bundle = {
        "type": key,
        "name": _ALGO_LABELS[key],
        "taskType": task_type,
        "targetName": target,
        "featureNames": feature_names,
        "metrics": metrics,
        "featureImportance": importance,
        "testTrue": _to_jsonable_list(y_test),
        "testPred": _to_jsonable_list(test_pred),
        "trainTime": round(train_time_ms, 2),
        "inferLatency": round(infer_latency_ms, 4),  # 每筆樣本平均推論延遲 (ms)
        "trainSize": int(len(y_train)),
        "testSize": int(len(y_test)),
        "means": scaler.mean_.tolist(),
        "stds": scaler.scale_.tolist(),
        "featureStats": feature_stats,
    }
    return bundle, estimator


def _failed_bundle(key: str, task_type: str, target: str, feature_names: list[str], err: str) -> dict[str, Any]:
    zero = (
        {"taskType": "regression", "testR2": 0.0, "testRMSE": 0.0, "testMAE": 0.0,
         "testMSE": 0.0, "trainR2": 0.0, "trainMSE": 0.0, "testScore": 0.0}
        if task_type == "regression" else
        {"taskType": "classification", "testAccuracy": 0.0, "trainAccuracy": 0.0,
         "precision": 0.0, "recall": 0.0, "f1": 0.0, "classes": [], "testScore": 0.0}
    )
    return {
        "type": key, "name": _ALGO_LABELS.get(key, key),
        "taskType": task_type, "targetName": target, "featureNames": feature_names,
        "metrics": zero, "featureImportance": [0.0] * len(feature_names),
        "testTrue": [], "testPred": [], "trainTime": 0.0, "inferLatency": 0.0,
        "trainSize": 0, "testSize": 0, "means": [], "stds": [], "featureStats": [],
        "error": err,
    }


# ============================================================
# Build feature matrix (numeric + date-derived)
# ============================================================
def _build_features(
    df: pd.DataFrame, target: str, requested: list[str] | None,
) -> tuple[pd.DataFrame, list[str]]:
    # Auto 模式才需要主動偵測 datetime 欄位、自動產生衍生 feature 名單。
    # Manual 模式 (前端有送 requested) 直接相信前端送什麼就解什麼,
    # 避免後端 datetime 偵測比 preprocess 嚴格時把合法的衍生 feature 拒掉。
    if requested is None:
        date_cols = _detect_date_columns(df, target)
        feature_names = [
            c for c in df.columns
            if c != target and pd.api.types.is_numeric_dtype(df[c])
        ]
        for dc in date_cols:
            feature_names.extend(dc + suf for suf in _DATE_SUFFIXES)
    else:
        feature_names = list(requested)

    if not feature_names:
        raise ValueError("沒有可用的特徵欄位")

    cols: dict[str, pd.Series] = {}
    for fn in feature_names:
        # 1. 直接是欄位名 → 當數值欄位處理
        if fn in df.columns and fn != target:
            cols[fn] = pd.to_numeric(df[fn], errors="coerce")
            continue

        # 2. 否則嘗試解成日期衍生 feature: fn = "{col_name}{suffix}"
        matched = False
        for suf in _DATE_SUFFIXES:
            if fn.endswith(suf):
                col_name = fn[: -len(suf)]
                if col_name in df.columns and col_name != target:
                    parsed = pd.to_datetime(df[col_name], errors="coerce")
                    cols[fn] = _date_feature_value(parsed, suf).astype(float)
                    matched = True
                    break
        if matched:
            continue

        raise ValueError(f"未知特徵: {fn}")

    return pd.DataFrame(cols), feature_names


def _detect_date_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Auto 模式專用 — 抓出看起來像 datetime 的欄位。"""
    out: list[str] = []
    for c in df.columns:
        if c == target:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
            continue
        if df[c].dtype == object:
            sample = df[c].dropna().head(50)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() / len(sample) > 0.7:
                out.append(c)
    return out


# ============================================================
# Metrics — 對應 js _regressionMetrics / _classificationMetrics
# ============================================================
def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_true: np.ndarray, y_train_pred: np.ndarray) -> dict[str, Any]:
    def _mse(yt, yp): return float(np.mean((yt - yp) ** 2))
    def _mae(yt, yp): return float(np.mean(np.abs(yt - yp)))
    def _r2(yt, yp):
        m = float(np.mean(yt))
        tot = float(np.sum((yt - m) ** 2))
        res = float(np.sum((yt - yp) ** 2))
        return 1 - res / tot if tot > 0 else 0.0

    test_r2 = _r2(y_true, y_pred)
    test_mse = _mse(y_true, y_pred)
    return {
        "taskType": "regression",
        "testMSE": test_mse,
        "testMAE": _mae(y_true, y_pred),
        "testRMSE": float(np.sqrt(test_mse)),
        "testR2": test_r2,
        "trainR2": _r2(y_train_true, y_train_pred),
        "trainMSE": _mse(y_train_true, y_train_pred),
        "testScore": test_r2,
    }


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_train_true: np.ndarray, y_train_pred: np.ndarray) -> dict[str, Any]:
    test_acc = float((y_pred == y_true).mean()) if len(y_true) else 0.0
    train_acc = float((y_train_pred == y_train_true).mean()) if len(y_train_true) else 0.0
    classes = sorted(set(list(y_true) + list(y_train_true)), key=lambda x: str(x))

    precision = recall = f1 = test_acc
    if len(classes) == 2:
        pos = classes[1]
        tp = int(((y_pred == pos) & (y_true == pos)).sum())
        fp = int(((y_pred == pos) & (y_true != pos)).sum())
        fn = int(((y_pred != pos) & (y_true == pos)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "taskType": "classification",
        "testAccuracy": test_acc,
        "trainAccuracy": train_acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classes": [str(c) for c in classes],
        "testScore": test_acc,
    }


# ============================================================
# Feature importance — coef_ / feature_importances_ / permutation
# ============================================================
def _feature_importance(estimator, X_test: np.ndarray, y_test: np.ndarray, n_features: int) -> list[float]:
    raw: np.ndarray | None = None

    # Tree-based: feature_importances_
    if hasattr(estimator, "feature_importances_"):
        raw = np.asarray(estimator.feature_importances_, dtype=float)
    # Linear: coef_
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        raw = np.abs(coef[0]) if coef.ndim == 2 else np.abs(coef)
    # Else use permutation importance
    else:
        try:
            r = permutation_importance(estimator, X_test, y_test, n_repeats=3, random_state=42, n_jobs=-1)
            raw = np.maximum(0.0, r.importances_mean)
        except Exception:
            raw = np.full(n_features, 1.0 / max(n_features, 1))

    if raw is None or raw.size != n_features:
        return [1.0 / max(n_features, 1)] * n_features

    s = float(raw.sum())
    if s <= 0:
        return [1.0 / max(n_features, 1)] * n_features
    return (raw / s).tolist()


def _to_jsonable_list(arr) -> list:
    out = []
    for v in np.asarray(arr).tolist():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            out.append(None)
        else:
            out.append(v)
    return out
