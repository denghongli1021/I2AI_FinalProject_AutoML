"""共用的 in-memory 狀態 (跨 request 保留) + per-user scope helpers。

正式環境會換成 Redis/DB,目前 demo 用 dict 即可。

User scoping 設計:
  每個 store entry 都有一個 "_owner" 欄位 (str),內容是:
    - "u{user_id}"  (登入使用者)
    - "guest"        (未登入的訪客 — 所有 guest 共用一個 namespace)
  讀取 / 列表時用 owner_id() 比對,不是自己的就當作不存在 (回 404 而非 403,
  避免 ID enumeration attack)。
"""

from __future__ import annotations
from typing import Any

from fastapi import HTTPException

# dataset_id -> {"df": pandas.DataFrame, "fileName": str, "loadedAt": float, "_owner": str}
DATASETS: dict[str, dict[str, Any]] = {}

# model_id -> {
#   "bundle":    dict, "estimator": ..., "scaler": ..., "featureNames": [...],
#   "X_test_df": ..., "preprocessorId": ..., "_owner": str,
# }
MODELS: dict[str, dict[str, Any]] = {}

# preprocessor_id -> {
#   "preprocessor": ColumnTransformer, "target": str, "datasetId": str,
#   "featureNames": [...], "X_train"/"X_test"/"y_train"/"y_test": ..., "_owner": str,
# }
PREPROCESSORS: dict[str, dict[str, Any]] = {}


# ============================================================
# Scope helpers
# ============================================================
def owner_id(user) -> str:
    """User 物件 → 字串 key。未登入回 'guest'。"""
    return f"u{user.id}" if user else "guest"


def stamp(entry: dict, user) -> dict:
    """在 entry 上蓋上擁有者印記,return 同一個 entry 方便鏈式呼叫。"""
    entry["_owner"] = owner_id(user)
    return entry


def get_owned(store: dict, entry_id: str, user, kind: str = "資源") -> dict:
    """從 store 拿 entry,並驗證擁有權。
    不是自己的或不存在都回 404 (避免別人枚舉 ID)。
    """
    entry = store.get(entry_id)
    if entry is None or entry.get("_owner") != owner_id(user):
        raise HTTPException(status_code=404, detail=f"{kind} 不存在")
    return entry


def list_owned(store: dict, user) -> list[tuple[str, dict]]:
    """列出當前 user 擁有的所有 (id, entry)。"""
    me = owner_id(user)
    return [(k, v) for k, v in store.items() if v.get("_owner") == me]
