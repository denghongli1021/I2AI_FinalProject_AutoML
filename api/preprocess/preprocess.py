"""
preprocess 模組 
================================
對外只需要實作 `run()`。可以在這個資料夾內任意新增其他 .py。

對應原本 js/data-engine.js 的功能:
  - analyzeColumns / detectColumnType / computeNumericStats / computeHistogram
  - countOutliers / computeValueCounts
  - computeCorrelationMatrix
  - computeHealthScore
  - generateProcessingLog

------------------------------------------------------------
Input:
  file_bytes : bytes
  filename   : str

Output: (df, response)
  df       : pandas.DataFrame
  response : dict — 對齊 DataEngine.currentDataset 的形狀
------------------------------------------------------------
"""

from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

# ============================================================
# Public entry
# ============================================================
MAX_BYTES = 50 * 1024 * 1024  # 50 MB 上限,避免大檔案直接 OOM


def run(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    # 檔案大小檢查 (#25)
    if len(file_bytes) > MAX_BYTES:
        raise ValueError(f"檔案過大 ({len(file_bytes) / 1024 / 1024:.1f} MB),上限 50 MB")

    # sep=None + engine='python' 讓 pandas 自動偵測分隔符 (CSV/TSV/管道符皆可) (#22)
    df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python')
    analysis = _analyze_columns(df)
    response = {
        "fileName": filename,
        "rowCount": len(df),
        "colCount": len(df.columns),
        "headers": list(df.columns),
        "data": df.fillna("").astype(str).values.tolist(),
        "analysis": analysis,
        "correlation": _correlation_matrix(df, analysis),
        "healthScore": _health_score(analysis, len(df)),
        "processingLog": _processing_log(analysis),
    }
    return df, response


# ============================================================
# Column analysis
# ============================================================
_MISSING_TOKENS = {"na", "nan", "null", "n/a", "none", "missing", "undefined", "?", "-", "--", "."}


def _is_missing(v: Any) -> bool:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == "":
        return True
    if isinstance(v, str):
        return v.strip().lower() in _MISSING_TOKENS
    return False


def _analyze_columns(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in df.columns:
        raw = df[name]
        total = len(raw)
        valid_mask = raw.apply(lambda v: not _is_missing(v))
        valid = raw[valid_mask]

        missing_count = int(total - valid_mask.sum())
        missing_pct = round((missing_count / total * 100), 1) if total else 0.0
        unique_count = int(valid.nunique())

        col_type, dtype = _detect_column_type(valid, raw)

        info: dict[str, Any] = {
            "name": name,
            "type": col_type,
            "dtype": dtype,
            "totalCount": total,
            "missingCount": missing_count,
            "missingPct": missing_pct,
            "uniqueCount": unique_count,
            "uniqueValues": [],
            "outlierCount": 0,
            "stats": None,
            "distribution": [],
            "topValues": [],
        }

        if col_type == "numeric":
            nums = pd.to_numeric(valid, errors="coerce").dropna().to_numpy(dtype=float)
            if len(nums) > 0:
                info["stats"] = _numeric_stats(nums)
                info["distribution"] = _histogram(nums, 15)
                info["outlierCount"] = _count_outliers(nums, info["stats"])
        elif col_type in ("categorical", "boolean", "text"):
            info["topValues"] = _value_counts(valid, 10)
            info["distribution"] = [v["count"] for v in info["topValues"]]
        elif col_type == "datetime":
            parsed = pd.to_datetime(valid, errors="coerce").dropna().sort_values()
            if len(parsed) > 0:
                info["stats"] = {
                    "min": parsed.iloc[0].strftime("%Y-%m-%d"),
                    "max": parsed.iloc[-1].strftime("%Y-%m-%d"),
                    "count": int(len(parsed)),
                }
                month_keys = parsed.dt.strftime("%Y-%m")
                month_counts = month_keys.value_counts().sort_index()
                info["distribution"] = [
                    {"label": str(k), "count": int(v)} for k, v in month_counts.items()
                ]

        out.append(info)
    return out


def _detect_column_type(valid: pd.Series, raw: pd.Series) -> tuple[str, str]:
    """Mirror data-engine.js detectColumnType."""
    if len(valid) == 0:
        return "unknown", "unknown"

    sample = valid.head(200)

    # Pandas dtype shortcuts
    if pd.api.types.is_bool_dtype(raw):
        return "boolean", "bool"
    if pd.api.types.is_numeric_dtype(raw):
        has_decimal = pd.api.types.is_float_dtype(raw)
        return "numeric", "float64" if has_decimal else "int64"
    if pd.api.types.is_datetime64_any_dtype(raw):
        return "datetime", "datetime"

    sample_str = sample.astype(str)

    # Boolean
    bool_set = set(sample_str.str.lower().str.strip().unique())
    if len(bool_set) <= 3 and bool_set <= {"true", "false", "yes", "no", "0", "1", "t", "f", "y", "n"}:
        return "boolean", "bool"

    # Numeric (>85% parse as number)
    numeric_count = pd.to_numeric(sample_str, errors="coerce").notna().sum()
    if len(sample) > 0 and numeric_count / len(sample) > 0.85:
        has_decimal = sample_str.str.contains(r"\.", regex=True, na=False).any()
        return "numeric", "float64" if has_decimal else "int64"

    # Datetime
    parsed = pd.to_datetime(sample, errors="coerce")
    if len(sample) > 0 and parsed.notna().sum() / len(sample) > 0.7:
        return "datetime", "datetime"

    # Categorical
    full_unique = valid.nunique()
    unique_ratio = len(set(sample)) / len(sample)
    if unique_ratio < 0.3 or full_unique <= 20:
        return "categorical", "category"

    return "text", "string"


def _numeric_stats(nums: np.ndarray) -> dict[str, Any]:
    sorted_arr = np.sort(nums)
    n = len(sorted_arr)
    mean = float(np.mean(sorted_arr))
    std = float(np.std(sorted_arr))
    q1 = float(sorted_arr[int(n * 0.25)])
    median = float(sorted_arr[int(n * 0.5)])
    q3 = float(sorted_arr[int(n * 0.75)])
    iqr = q3 - q1
    if std > 0:
        skewness = float(np.mean(((sorted_arr - mean) / std) ** 3))
    else:
        skewness = 0.0
    return {
        "count": int(n),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": float(sorted_arr[0]),
        "q1": round(q1, 4),
        "median": round(median, 4),
        "q3": round(q3, 4),
        "max": float(sorted_arr[-1]),
        "iqr": round(iqr, 4),
        "skewness": round(skewness, 4),
    }


def _histogram(nums: np.ndarray, bins: int) -> list[dict[str, Any]]:
    if len(nums) == 0:
        return []
    lo, hi = float(np.min(nums)), float(np.max(nums))
    if lo == hi:
        return [{"min": lo, "max": hi, "count": int(len(nums)), "label": f"{lo:.1f}"}]
    width = (hi - lo) / bins
    out = []
    for i in range(bins):
        left = lo + i * width
        right = left + width
        if i == bins - 1:
            count = int(((nums >= left) & (nums <= right)).sum())
        else:
            count = int(((nums >= left) & (nums < right)).sum())
        out.append({
            "min": round(left, 4),
            "max": round(right, 4),
            "count": count,
            "label": f"{left:.1f}-{right:.1f}",
        })
    return out


def _count_outliers(nums: np.ndarray, stats: dict[str, Any]) -> int:
    if not stats:
        return 0
    lower = stats["q1"] - 1.5 * stats["iqr"]
    upper = stats["q3"] + 1.5 * stats["iqr"]
    return int(((nums < lower) | (nums > upper)).sum())


def _value_counts(values: pd.Series, top_n: int) -> list[dict[str, Any]]:
    counts = values.astype(str).value_counts().head(top_n)
    total = len(values)
    return [
        {
            "value": str(k),
            "count": int(v),
            "pct": round((v / total) * 100, 1) if total else 0.0,
        }
        for k, v in counts.items()
    ]


# ============================================================
# Correlation matrix (Pearson on numeric columns)
# ============================================================
def _correlation_matrix(df: pd.DataFrame, analysis: list[dict[str, Any]]) -> dict[str, Any] | None:
    numeric_cols = [a["name"] for a in analysis if a["type"] == "numeric"]
    if len(numeric_cols) < 2:
        return None
    sub = df[numeric_cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 2:
        return None
    corr = sub.corr(method="pearson").fillna(0.0)
    matrix: list[list[Any]] = []
    for i, _ in enumerate(numeric_cols):
        for j, _ in enumerate(numeric_cols):
            matrix.append([i, j, round(float(corr.iloc[i, j]), 3)])
    return {"names": numeric_cols, "matrix": matrix}


# ============================================================
# Data health score
# ============================================================
def _health_score(analysis: list[dict[str, Any]], row_count: int) -> dict[str, int]:
    if not analysis:
        return {"completeness": 0, "consistency": 0, "outlierScore": 0,
                "balanceScore": 0, "featureQuality": 0, "overall": 0}

    avg_missing = sum(c["missingPct"] for c in analysis) / len(analysis)
    completeness = max(0, round(100 - avg_missing))

    typed = [c for c in analysis if c["type"] not in ("unknown", "text")]
    consistency = round(len(typed) / len(analysis) * 100)

    total_outliers = sum(c["outlierCount"] for c in analysis)
    outlier_score = max(0, round(100 - (total_outliers / row_count) * 100 * 5)) if row_count else 0

    cat_cols = [c for c in analysis if c["type"] in ("categorical", "boolean")]
    balance_score = 85
    for c in cat_cols:
        if c["topValues"]:
            top_pct = c["topValues"][0]["pct"]
            if top_pct > 90:
                balance_score = min(balance_score, 40)
            elif top_pct > 80:
                balance_score = min(balance_score, 60)
            elif top_pct > 70:
                balance_score = min(balance_score, 75)

    def _is_good(c: dict[str, Any]) -> bool:
        if c["type"] == "numeric":
            return bool(c["stats"]) and c["stats"]["std"] > 0
        if c["type"] == "categorical":
            return 1 < c["uniqueCount"] < c["totalCount"] * 0.5
        return True

    feature_quality = round(sum(1 for c in analysis if _is_good(c)) / len(analysis) * 100)

    overall = round((completeness + consistency + outlier_score + balance_score + feature_quality) / 5)
    return {
        "completeness": completeness,
        "consistency": consistency,
        "outlierScore": outlier_score,
        "balanceScore": balance_score,
        "featureQuality": feature_quality,
        "overall": overall,
    }


# ============================================================
# Auto-processing recommendation log
# ============================================================
def _processing_log(analysis: list[dict[str, Any]]) -> list[dict[str, Any]]:
    logs: list[dict[str, Any]] = []
    time_str = datetime.now().strftime("%H:%M")

    for col in analysis:
        name = col["name"]
        if 0 < col["missingPct"] < 50:
            method = "median" if col["type"] == "numeric" else "mode"
            method_label = "中位數" if col["type"] == "numeric" else "眾數"
            logs.append({
                "type": "success", "action": "fill_missing", "colName": name, "time": time_str,
                "text": f'已自動填補 <strong class="text-accent-400">{name}</strong> 欄位 {col["missingPct"]}% 的缺失值（使用{method_label}填補）',
                "selectedMethod": method,
            })
        elif col["missingPct"] >= 50:
            logs.append({
                "type": "warning", "action": "drop_column", "colName": name, "time": time_str,
                "text": f'<strong class="text-warning-400">{name}</strong> 欄位缺失率高達 {col["missingPct"]}%，建議考慮移除此欄位',
                "selectedMethod": "drop_column",
            })

        if col["outlierCount"] > 0:
            logs.append({
                "type": "warning" if col["outlierCount"] > 10 else "success",
                "action": "handle_outlier", "colName": name, "time": time_str,
                "text": f'偵測到 <strong class="{"text-warning-400" if col["outlierCount"] > 10 else "text-accent-400"}">{name}</strong> 欄位有 {col["outlierCount"]} 筆異常值（IQR 法）',
                "selectedMethod": "keep",
            })

        if col["type"] == "categorical" and col["uniqueCount"] <= 10:
            logs.append({
                "type": "success", "action": "encode", "colName": name, "time": time_str,
                "text": f'已自動將 <strong class="text-accent-400">{name}</strong> 欄位進行 One-Hot 編碼（{col["uniqueCount"]} 個類別）',
                "selectedMethod": "onehot",
            })

        if col["type"] == "datetime":
            logs.append({
                "type": "success", "action": "decompose_date", "colName": name, "time": time_str,
                "text": f'已將 <strong class="text-accent-400">{name}</strong> 拆解為年、月、星期、季度特徵',
                "selectedMethod": "decompose",
            })

    num_cols = [c for c in analysis if c["type"] == "numeric" and c["stats"] and c["stats"]["std"] > 0]
    if len(num_cols) >= 2:
        a, b = num_cols[0]["name"], num_cols[1]["name"]
        logs.append({
            "type": "success", "action": "interaction", "colName": f"{a} × {b}", "time": time_str,
            "text": f'已自動產生交互特徵: <strong class="text-accent-400">{a} × {b}</strong>',
            "selectedMethod": "multiply",
        })
    return logs
