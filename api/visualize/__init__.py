"""api.visualize package — 對外只暴露 run()。

實作在 visualize.py。隊友 C 改 visualize.py(或在資料夾內加新檔案,例如 charts.py、shap_utils.py)即可,
main.py 永遠透過 `from api import visualize; visualize.run(...)` 呼叫,介面不變。
"""

from .visualize import run

__all__ = ["run"]
