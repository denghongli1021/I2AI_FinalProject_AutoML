import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_score(y_true, y_pred, metric="f1", y_score=None):
    """
    y_pred : hard labels（f1 / accuracy 用）
    y_score: 機率矩陣 [N, C]（roc_auc 必填）
    """
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    elif metric == "roc_auc":
        if y_score is None:
            raise ValueError("roc_auc 需要 y_score（機率矩陣）")
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            s = y_score[:, 1] if y_score.ndim == 2 else y_score
            return roc_auc_score(y_true, s)
        return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    else:
        return f1_score(y_true, y_pred, average="macro", zero_division=0)

def get_metric_name(metric="f1"):
    if metric == "accuracy":
        return "Accuracy"
    if metric == "roc_auc":
        return "AUC-ROC"
    return "Macro F1"
