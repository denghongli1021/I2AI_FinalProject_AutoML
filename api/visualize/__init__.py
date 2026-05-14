"""api.visualize package.

兩條對外介面:
  - `run(model_bundle, chart_type, options)` — 既有的 ECharts JSON 產生器
    (給 /api/visualize 端點用,前端 chart.setOption 直接吃),定義在 visualize.py。
  - `AutoMLVisualizer` — 隊友的 SHAP + Plotly 模組,輸出 PNG 圖表到磁碟。
    定義在 visualizer.py。需要 shap / plotly / kaleido 套件。
"""

from .visualize import run

# AutoMLVisualizer 用到的 shap / plotly / kaleido 是重套件,
# 沒裝時不讓整個 api.visualize import 都炸掉 — callers 用 `if AutoMLVisualizer:` 判斷。
try:
    from .visualizer import AutoMLVisualizer
except ImportError:
    AutoMLVisualizer = None

__all__ = ["run", "AutoMLVisualizer"]
