import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def calculate_score(y_true, y_pred, metric="f1"):
    """
    統一計算指標。
    metric: "f1" (Macro F1) 或 "accuracy"
    """
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    else:
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

def get_metric_name(metric="f1"):
    return "Macro F1" if metric == "f1" else "Accuracy"
