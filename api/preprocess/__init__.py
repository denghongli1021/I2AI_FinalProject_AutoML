"""api.preprocess package — 對外只暴露 run()。

實作在 preprocess.py。隊友 A 改 preprocess.py(或在資料夾內加新檔案)即可,
main.py 永遠透過 `from api import preprocess; preprocess.run(...)` 呼叫,介面不變。
"""

from .preprocess import run

__all__ = ["run"]
