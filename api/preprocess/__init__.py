# api/preprocess/__init__.py
"""api.preprocess package.

兩條對外介面:
  - `run(file_bytes, filename)` — 既有 CSV 分析 (analysis + correlation + healthScore + log)
    給 /api/preprocess 端點使用，定義在 preprocess.py。**這是本地保留的舊端點**，
    daniel branch 沒有，merge 時別覆蓋掉。
  - `run_data_audit`, `preprocess_for_training`, `preprocess_for_inference` —
    直接從外層 preprocessing/ 模組匯入，避免重複維護兩份程式碼。
"""

from .preprocess import run

# 直接使用外層 preprocessing/ 模組，不再維護重複副本
from preprocessing import (
    run_data_audit,
    preprocess_for_training,
    preprocess_for_inference,
)

try:
    from preprocessing.interface import preprocess_for_timeseries
except (ImportError, AttributeError):
    preprocess_for_timeseries = None  # type: ignore

__all__ = [
    "run",
    "run_data_audit",
    "preprocess_for_training",
    "preprocess_for_inference",
    "preprocess_for_timeseries",
]
