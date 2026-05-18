"""輸出最終 Submission CSV 並做基本驗證。"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .config import SUBMISSIONS_DIR


def generate_submission(
    preds: np.ndarray,
    test_ids: np.ndarray,
    label_encoder: LabelEncoder,
    out_name: str = "submission.csv",
    target_col: str = "target_feature",
    id_col: str = "id",
) -> str:
    """
    Parameters
    ----------
    preds         : [N_test] 整數標籤預測值
    test_ids      : [N_test] 測試集 ID
    label_encoder : 用於將整數還原成原始字串標籤
    out_name      : 輸出檔名（存於 SUBMISSIONS_DIR）
    """
    assert len(preds) == len(test_ids), "preds 與 test_ids 長度不符"
    assert not np.any(np.isnan(preds.astype(float))), "preds 含 NaN"

    labels = label_encoder.inverse_transform(preds.astype(int))
    df = pd.DataFrame({id_col: test_ids, target_col: labels})

    out_path = os.path.join(SUBMISSIONS_DIR, out_name)
    df.to_csv(out_path, index=False)

    print(f"  [Submit] Saved → {out_path}  shape={df.shape}")
    print(f"  [Submit] Class distribution:\n{df[target_col].value_counts().sort_index().to_string()}")
    return out_path
