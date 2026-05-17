"""FastAPI Depends — 從 Authorization header 抽出 user。

兩種模式:
  - get_current_user      : Optional[User] (guest 允許,沒登入回 None)
  - get_required_user     : User (沒登入直接 401)
"""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from .db import User, get_db
from .security import decode_access_token


def _extract_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if not authorization.lower().startswith("bearer "):
        return None
    return authorization[7:].strip() or None


def get_current_user(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> User | None:
    """允許 guest — 沒帶或 token 失效都回 None,不丟錯。"""
    token = _extract_token(authorization)
    if not token:
        return None
    payload = decode_access_token(token)
    if not payload:
        return None
    try:
        user_id = int(payload.get("sub", 0))
    except (TypeError, ValueError):
        return None
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()


def get_required_user(
    user: User | None = Depends(get_current_user),
) -> User:
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要登入",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
