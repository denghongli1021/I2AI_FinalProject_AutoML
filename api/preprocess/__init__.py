# api/preprocess/__init__.py
"""api.preprocess package.

兩條對外介面:
  - `run(file_bytes, filename)` — 既有 CSV 分析 (analysis + correlation + healthScore + log)
    給 /api/preprocess 端點使用,定義在 preprocess.py。**這是我們本地保留的舊端點**,
    daniel branch 沒有,merge 時別覆蓋掉。
  - `run_data_audit`, `preprocess_for_training`, `preprocess_for_inference`,
    `preprocess_for_timeseries` — 隊友(daniel)的 sklearn pipeline 預處理模組,
    定義在 interface.py。
"""

from .preprocess import run
from .interface import (
    run_data_audit,
    preprocess_for_training,
    preprocess_for_inference,
)

# preprocess_for_timeseries 是 daniel 新加的時序專用 path;
# 沒裝某些可選 deps 時 import 會失敗,所以包 try/except 不擋 module 載入。
try:
    from .interface import preprocess_for_timeseries
except ImportError:
    preprocess_for_timeseries = None  # type: ignore

__all__ = [
    "run",
    "run_data_audit",
    "preprocess_for_training",
    "preprocess_for_inference",
    "preprocess_for_timeseries",
]
