"""api.train package — 對外只暴露 run()。

實作在 train.py。隊友 B 改 train.py(或在資料夾內加新檔案,例如 algorithms.py、metrics.py)即可,
main.py 永遠透過 `from api import train; train.run(...)` 呼叫,介面不變。
"""

from .train import run

__all__ = ["run"]
