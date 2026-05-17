"""密碼雜湊 + JWT 簽發 / 驗證。

JWT_SECRET 環境變數一定要設,否則 fallback 到 dev 預設 (不安全,只給本機用)。

bcrypt 直接呼叫 (不透過 passlib) — 因為 passlib 1.7.x 跟 bcrypt 5.x 不相容,
且 passlib 自 2020 後幾乎沒維護。SHA-256 預雜湊解掉 bcrypt 的 72-byte 限制
(Django / Devise 等現代框架都這樣做)。
"""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import datetime, timedelta
from typing import Any

import bcrypt
from jose import JWTError, jwt

_JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-do-not-use-in-prod-CHANGE-ME")
_JWT_ALG = "HS256"
_JWT_EXPIRE_DAYS = 30  # token 有效期 30 天


def _normalize_password(password: str) -> bytes:
    """SHA-256 預雜湊 + base64,解掉 bcrypt 72-byte 上限,並避免 null byte 問題。"""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    hashed = bcrypt.hashpw(_normalize_password(password), bcrypt.gensalt())
    return hashed.decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_normalize_password(password), password_hash.encode("utf-8"))
    except Exception:
        return False


def create_access_token(user_id: int, email: str) -> str:
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=_JWT_EXPIRE_DAYS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALG)


def decode_access_token(token: str) -> dict[str, Any] | None:
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALG])
    except JWTError:
        return None
