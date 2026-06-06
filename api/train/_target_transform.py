"""
回歸 target 自動轉換 — 平台層通用,三引擎(sklearn / daniel / autogluon)共用。

動機:
  右偏正值 target (銷量、瀏覽數、計數、房價...) 直接用 MSE 訓練,
  樹模型自然偏向高值,小值段位被當「不重要」→ MAPE 爆炸。
  自動偵測偏度後 log1p,推論期 expm1 + clip(≥0) 取回原尺度。

設計:
  · 訓練時:呼叫 decide_target_transform(y_train, task) → 拿 info dict
  · 訓練前:apply_forward(y_train, info)        — log1p (若需要)
  · 推論後:apply_inverse(preds, info)           — expm1 + clip (若需要)
  · 內部評估指標(R²/RMSE/MAE)在 inverse 後計算,給使用者看的是原尺度數字
  · bundle["targetTransform"] = info 持久化,推論期 (main.py) 集中還原
"""
from __future__ import annotations
from typing import Any
import numpy as np


def _safe_skew(y: np.ndarray) -> float:
    """不依賴 scipy 的 skew(rho_3) 計算。"""
    y = y.astype(np.float64)
    mu = float(y.mean())
    sd = float(y.std())
    if sd < 1e-12 or len(y) < 3:
        return 0.0
    return float(np.mean(((y - mu) / sd) ** 3))


def decide_target_transform(y_train, task_type: str) -> dict[str, Any] | None:
    """偵測 y_train 是否該做 log1p / clip 非負。

    Returns:
      None — 不適用(分類任務 / y 全 NaN / 樣本太少)
      dict — {
        "log1p":       bool,      # 訓練 y 是否套 log1p
        "clip_nonneg": bool,      # 推論預測是否 clip 到 ≥ 0
        "y_min":       float,     # 訓練 y 觀察到的最小值
        "y_skew":      float,     # 偏度
        "reason":      str,       # 給 log 用的人可讀說明
      }

    判準:
      · clip_nonneg:y_min ≥ 0 (例如銷量、計數、價格,負值無意義)
      · log1p:同時滿足
          (a) y_min ≥ 0
          (b) skew ≥ 1.0  (右偏夠顯著,非 1.0 直接拒絕)
          (c) max/p50 ≥ 5 (動態範圍夠大,小範圍 log 反而失真)
          (d) p50 > 0     (避免一堆 0 把 p50 拉成 0)
    """
    if task_type != "regression":
        return None
    y = np.asarray(y_train, dtype=np.float64).ravel()
    y = y[np.isfinite(y)]
    if len(y) < 10:
        return None

    y_min = float(y.min())
    y_skew = _safe_skew(y)
    p50 = float(np.percentile(y, 50))
    y_max = float(y.max())

    clip_nonneg = y_min >= 0.0
    log1p = (
        clip_nonneg
        and y_skew >= 1.0
        and p50 > 0.0
        and (y_max / max(p50, 1.0)) >= 5.0
    )

    reason_parts = []
    if log1p:
        reason_parts.append(f"log1p (skew={y_skew:.2f}, max/p50={y_max/max(p50,1.0):.1f})")
    if clip_nonneg and not log1p:
        reason_parts.append(f"clip≥0 (y_min={y_min:.2g})")
    if not reason_parts:
        reason_parts.append("none (y 含負值或分佈近常態)")

    return {
        "log1p": bool(log1p),
        "clip_nonneg": bool(clip_nonneg),
        "y_min": y_min,
        "y_skew": y_skew,
        "reason": " + ".join(reason_parts),
    }


def apply_forward(y, info: dict | None):
    """訓練時對 y 做正向轉換。info=None 時直接 pass-through。"""
    if not info or not info.get("log1p"):
        return y
    return np.log1p(np.asarray(y, dtype=np.float64))


def apply_inverse(preds, info: dict | None):
    """推論時對 preds 還原 + clip。info=None 時直接 pass-through。"""
    if not info:
        return preds
    p = np.asarray(preds, dtype=np.float64)
    if info.get("log1p"):
        p = np.expm1(p)
    if info.get("clip_nonneg"):
        p = np.clip(p, 0.0, None)
    return p
