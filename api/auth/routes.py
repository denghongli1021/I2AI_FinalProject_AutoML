"""Auth 路由 — email/password + me + logout + OAuth (掛在子 router)。"""

from __future__ import annotations

import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from .db import User, get_db
from .dependencies import get_current_user
from .oauth import router as oauth_router
from .security import create_access_token, hash_password, verify_password


router = APIRouter(prefix="/auth", tags=["auth"])
router.include_router(oauth_router)  # /auth/oauth/google/start 等等


# ============================================================
# Schemas
# ============================================================
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    displayName: str | None = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    token: str
    user: dict


# ============================================================
# /auth/register
# ============================================================
@router.post("/register", response_model=AuthResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    # 基本密碼長度檢查
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密碼至少 6 字元")

    existing = db.query(User).filter(User.email == req.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="此 Email 已被註冊")

    user = User(
        email=req.email,
        display_name=req.displayName or req.email.split("@")[0],
        password_hash=hash_password(req.password),
        last_login_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return AuthResponse(token=token, user=user.to_dict())


# ============================================================
# /auth/login
# ============================================================
@router.post("/login", response_model=AuthResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not user.password_hash:
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")

    user.last_login_at = datetime.utcnow()
    db.commit()
    db.refresh(user)

    token = create_access_token(user.id, user.email)
    return AuthResponse(token=token, user=user.to_dict())


# ============================================================
# /auth/me  — 回傳當前使用者 (沒登入回 null)
# ============================================================
@router.get("/me")
def me(user: User | None = Depends(get_current_user)):
    if user is None:
        return {"user": None}
    return {"user": user.to_dict()}


# ============================================================
# /auth/logout  — server-side stateless,只回 200;前端負責清 token
# ============================================================
@router.post("/logout")
def logout():
    return {"ok": True}
