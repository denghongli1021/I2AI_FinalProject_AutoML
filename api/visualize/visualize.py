"""
visualize 模組
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


_TEXT_COLOR = "#e2e8f0"


def run(
    model_bundle: dict[str, Any],
    chart_type: str,
    options: dict[str, Any],
) -> dict[str, Any]:
    if chart_type == "feature_importance":
        return _feature_importance(model_bundle)
    if chart_type == "predictions":
        return _predictions(model_bundle)
    if chart_type == "confusion_matrix":
        return _confusion_matrix(model_bundle)
    if chart_type == "residuals":
        return _residuals(model_bundle)
    if chart_type == "shap":
        return _shap(model_bundle, options)

    raise HTTPException(status_code=400, detail=f"unsupported chartType: {chart_type}")


def _feature_importance(m: dict[str, Any]) -> dict[str, Any]:
    names = m.get("featureNames", [])
    values = m.get("featureImportance", [])
    pairs = sorted(zip(names, values), key=lambda p: p[1], reverse=True)[:15]
    return {
        "option": {
            "title": {"text": f"Feature Importance - {m.get('name', m.get('type', ''))}", "textStyle": {"color": _TEXT_COLOR}},
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
            "title": {"text": "Actual vs Predicted", "textStyle": {"color": _TEXT_COLOR}},
            "xAxis": {"name": "Actual"},
            "yAxis": {"name": "Predicted"},
            "series": [{
                "type": "scatter",
                "data": list(zip(actual, pred)),
                "itemStyle": {"color": "#22d3ee"},
            }],
        }
    }


def _confusion_matrix(m: dict[str, Any]) -> dict[str, Any]:
    """混淆矩陣 — 分類任務必備。ECharts heatmap option。"""
    actual = m.get("testTrue", []) or []
    pred = m.get("testPred", []) or []
    if not actual or not pred:
        raise HTTPException(status_code=400, detail="模型沒有 testTrue/testPred,無法繪製 confusion matrix")
    if m.get("taskType") and m.get("taskType") != "classification":
        raise HTTPException(status_code=400, detail="confusion_matrix 只適用於分類任務")

    # 共同 labels (用 str 比對,避開 0/0.0 / "0" 混在一起的型別問題)
    labels = sorted({str(v) for v in actual} | {str(v) for v in pred})
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]
    for a, p in zip(actual, pred):
        matrix[idx[str(a)]][idx[str(p)]] += 1

    # ECharts heatmap 資料格式:[xIdx, yIdx, value]
    data = []
    max_v = 0
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            data.append([j, i, v])
            if v > max_v:
                max_v = v

    return {
        "option": {
            "title": {"text": f"Confusion Matrix - {m.get('name', m.get('type', ''))}",
                      "textStyle": {"color": _TEXT_COLOR}},
            "tooltip": {"position": "top"},
            "grid": {"left": "12%", "right": "10%", "bottom": "15%"},
            "xAxis": {"type": "category", "data": labels, "name": "Predicted",
                      "axisLabel": {"color": _TEXT_COLOR}, "nameTextStyle": {"color": _TEXT_COLOR}},
            "yAxis": {"type": "category", "data": labels, "name": "Actual",
                      "axisLabel": {"color": _TEXT_COLOR}, "nameTextStyle": {"color": _TEXT_COLOR}},
            "visualMap": {
                "min": 0, "max": max_v or 1,
                "calculable": True,
                "orient": "horizontal", "left": "center", "bottom": "0%",
                "textStyle": {"color": _TEXT_COLOR},
                "inRange": {"color": ["#0f172a", "#1e3a8a", "#3b82f6", "#22d3ee"]},
            },
            "series": [{
                "name": "count",
                "type": "heatmap",
                "data": data,
                "label": {"show": True, "color": _TEXT_COLOR},
                "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}},
            }],
        }
    }


def _residuals(m: dict[str, Any]) -> dict[str, Any]:
    """殘差圖 — 回歸任務看模型偏差。X = 預測值,Y = 殘差 (actual - pred)。"""
    actual = m.get("testTrue", []) or []
    pred = m.get("testPred", []) or []
    if not actual or not pred:
        raise HTTPException(status_code=400, detail="模型沒有 testTrue/testPred,無法繪製 residuals")
    if m.get("taskType") and m.get("taskType") != "regression":
        raise HTTPException(status_code=400, detail="residuals 只適用於回歸任務")

    try:
        residuals = [float(a) - float(p) for a, p in zip(actual, pred)]
        pred_f = [float(p) for p in pred]
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="testTrue/testPred 含非數值,無法計算殘差")

    points = list(zip(pred_f, residuals))
    return {
        "option": {
            "title": {"text": f"Residuals - {m.get('name', m.get('type', ''))}",
                      "subtext": "Y = actual - predicted, 越靠近 0 越好",
                      "textStyle": {"color": _TEXT_COLOR},
                      "subtextStyle": {"color": "#94a3b8"}},
            "tooltip": {"trigger": "item",
                        "formatter": "predicted: {c0}<br/>residual: {c1}"},
            "grid": {"left": "12%", "right": "5%", "bottom": "15%"},
            "xAxis": {"type": "value", "name": "Predicted",
                      "axisLabel": {"color": _TEXT_COLOR},
                      "nameTextStyle": {"color": _TEXT_COLOR}},
            "yAxis": {"type": "value", "name": "Residual",
                      "axisLabel": {"color": _TEXT_COLOR},
                      "nameTextStyle": {"color": _TEXT_COLOR}},
            # y=0 參考線 — 完美預測會落在這條線上
            "series": [{
                "type": "scatter",
                "data": points,
                "symbolSize": 6,
                "itemStyle": {"color": "#22d3ee", "opacity": 0.6},
                "markLine": {
                    "silent": True,
                    "symbol": "none",
                    "lineStyle": {"color": "#f87171", "type": "dashed"},
                    "data": [{"yAxis": 0, "label": {"formatter": "y=0", "color": "#f87171"}}],
                },
            }],
        }
    }


def _shap(m: dict[str, Any], options: dict[str, Any]) -> dict[str, Any]:
    """
    SHAP 圖 — 若 model_bundle 有 shap_values 就畫真的;沒有就 fallback 到
    feature_importance + 回傳 fallback flag,讓前端知道顯示的不是真 SHAP。
    """
    shap_values = m.get("shapValues") or m.get("shap_values")
    if shap_values:
        # TODO: 等 train 模組真的存 shap_values 進 bundle 再實作 beeswarm/waterfall
        # 先回 placeholder 避免噴錯
        names = m.get("featureNames", [])
        # 取每個特徵的平均 |SHAP| 當 global importance
        try:
            mean_abs = [
                sum(abs(row[i]) for row in shap_values) / max(len(shap_values), 1)
                for i in range(len(names))
            ]
            pairs = sorted(zip(names, mean_abs), key=lambda p: p[1], reverse=True)[:15]
            return {
                "option": {
                    "title": {"text": f"SHAP — Mean |Impact| ({m.get('name', '')})",
                              "subtext": "shap 平均絕對值排序",
                              "textStyle": {"color": _TEXT_COLOR},
                              "subtextStyle": {"color": "#94a3b8"}},
                    "tooltip": {"trigger": "axis"},
                    "xAxis": {"type": "value"},
                    "yAxis": {"type": "category", "data": [p[0] for p in pairs]},
                    "series": [{
                        "type": "bar",
                        "data": [p[1] for p in pairs],
                        "itemStyle": {"color": "#a855f7"},
                    }],
                }
            }
        except (TypeError, IndexError):
            pass

    # Fallback:用 feature_importance 充當,並打 flag 告訴前端
    fi = _feature_importance(m)
    fi["fallback"] = True
    fi["fallbackReason"] = "模型未保存 SHAP values,使用 feature importance 替代"
    fi["option"]["title"]["text"] = f"Feature Importance (SHAP fallback) - {m.get('name', m.get('type', ''))}"
    return fi
