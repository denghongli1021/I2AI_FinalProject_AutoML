"""
visualize 模組 — 隊友 C 的領地
==============================
對外只需要實作一個 `run()` 函式,main.py 會呼叫它。

可以在這個資料夾內任意新增其他 .py 檔 (例如 charts.py, shap_utils.py),
然後在這裡 import 進來即可。

------------------------------------------------------------
Input:
  model_bundle : dict     train.run 回傳的單一模型 (含 predictions / featureImportance ...)
  chart_type   : str      "feature_importance" | "predictions" | "residuals"
                          | "shap" | "confusion_matrix" | ...
  options      : dict     額外參數 (例如 SHAP 樣本 index)

Output: dict
  {"option": <ECharts option object>}
  前端會直接 chart.setOption(response.option),
  所以 schema 完全自由,只要符合 ECharts:
  https://echarts.apache.org/zh/option.html
------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException


# TODO(隊友 C): 補上 residuals / shap / confusion_matrix 等 chart_type
def run(
    model_bundle: dict[str, Any],
    chart_type: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    if chart_type == "feature_importance":
        return _feature_importance(model_bundle)
    if chart_type == "predictions":
        return _predictions(model_bundle)

    raise HTTPException(status_code=400, detail=f"unsupported chartType: {chart_type}")


def _feature_importance(m: dict[str, Any]) -> dict[str, Any]:
    names = m.get("featureNames", [])
    values = m.get("featureImportance", [])
    pairs = sorted(zip(names, values), key=lambda p: p[1], reverse=True)[:15]
    return {
        "option": {
            "title": {"text": f"Feature Importance - {m.get('name', m.get('type', ''))}", "textStyle": {"color": "#e2e8f0"}},
            "tooltip": {"trigger": "axis"},
            "xAxis": {"type": "value"},
            "yAxis": {"type": "category", "data": [p[0] for p in pairs]},
            "series": [{
                "type": "bar",
                "data": [p[1] for p in pairs],
                "itemStyle": {"color": "#3b82f6"},
            }],
        }
    }


def _predictions(m: dict[str, Any]) -> dict[str, Any]:
    actual = m.get("testTrue", [])
    pred = m.get("testPred", [])
    return {
        "option": {
            "title": {"text": "Actual vs Predicted", "textStyle": {"color": "#e2e8f0"}},
            "xAxis": {"name": "Actual"},
            "yAxis": {"name": "Predicted"},
            "series": [{
                "type": "scatter",
                "data": list(zip(actual, pred)),
                "itemStyle": {"color": "#22d3ee"},
            }],
        }
    }
