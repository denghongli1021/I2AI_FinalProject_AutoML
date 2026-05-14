"""api.preprocess package.

兩條對外介面:
  - `run(file_bytes, filename)` — 既有 CSV 分析 (analysis + correlation + healthScore + log)
    給 /api/preprocess 端點使用,定義在 preprocess.py。
  - `run_data_audit`, `preprocess_for_training`, `preprocess_for_inference`
    — 隊友的 sklearn pipeline 預處理模組,定義在 interface.py。
    api/main.py 之後改成呼叫這幾個就能切換成新版預處理。
"""

from .preprocess import run
from .interface import (
    run_data_audit,
    preprocess_for_training,
    preprocess_for_inference,
)

__all__ = [
    "run",
    "run_data_audit",
    "preprocess_for_training",
    "preprocess_for_inference",
]
