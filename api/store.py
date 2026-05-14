"""共用的 in-memory 狀態 (跨 request 保留)。
正式環境會換成 Redis/DB,目前 demo 用 dict 即可。
"""

from __future__ import annotations
from typing import Any

# dataset_id -> {"df": pandas.DataFrame, "fileName": str, "loadedAt": float}
DATASETS: dict[str, dict[str, Any]] = {}

# model_id -> {
#   "bundle":    dict,           # JSON 安全的 model bundle (回給前端)
#   "estimator": sklearn obj,    # for /api/predict
#   "scaler":    StandardScaler, # 標準化用
#   "featureNames": [str, ...],
# }
MODELS: dict[str, dict[str, Any]] = {}

# preprocessor_id -> {
#   "preprocessor": sklearn ColumnTransformer (fitted),
#   "target":       str,
#   "datasetId":    str,
#   "featureNames": [str, ...],   # 處理後的欄位名 (含 OneHot 展開)
# }
PREPROCESSORS: dict[str, dict[str, Any]] = {}
