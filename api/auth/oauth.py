"""Google + GitHub OAuth 流程。

流程:
  前端按「用 Google 登入」 → 開新分頁到 /api/auth/oauth/google/start
                            → 後端 302 到 Google 認證頁
                            → 使用者授權 → Google 302 回 /api/auth/oauth/google/callback
                            → 後端拿 code 換 access_token,fetch user info
                            → 找/建立 DB user,簽 JWT
                            → 302 回前端 (FRONTEND_URL?token=xxx)
                            → 前端 JS 讀 ?token=xxx,存到 localStorage,清掉 query string

環境變數:
  GOOGLE_OAUTH_CLIENT_ID
  GOOGLE_OAUTH_CLIENT_SECRET
  GITHUB_OAUTH_CLIENT_ID
  GITHUB_OAUTH_CLIENT_SECRET
  FRONTEND_URL  (callback 完成後 302 的目的地,例:http://localhost:5500 或 https://yourname.github.io/repo)
  OAUTH_REDIRECT_BASE  (這個後端對外的 URL,例:http://localhost:8000 或 https://i2ai-automl-api.onrender.com)
"""

from __future__ import annotations

import os
from datetime import datetime
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from .db import SessionLocal, User
from .security import create_access_token


# prefix 只放 /oauth — 上層的 auth_router 已經有 /auth,避免雙重 prefix
router = APIRouter(prefix="/oauth", tags=["auth-oauth"])


# ---------- 環境變數讀取 ----------
def _env(key: str) -> str | None:
    v = os.environ.get(key, "").strip()
    return v or None


def _frontend_url() -> str:
    return _env("FRONTEND_URL") or "http://localhost:5500"


def _redirect_base() -> str:
    return _env("OAUTH_REDIRECT_BASE") or "http://localhost:8000"


def _provider_config(provider: str) -> dict:
    """provider → 環境變數 + endpoint URLs."""
    if provider == "google":
        return {
            "client_id":     _env("GOOGLE_OAUTH_CLIENT_ID"),
            "client_secret": _env("GOOGLE_OAUTH_CLIENT_SECRET"),
            "authorize_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url":     "https://oauth2.googleapis.com/token",
            "userinfo_url":  "https://www.googleapis.com/oauth2/v3/userinfo",
            "scope":         "openid email profile",
        }
    if provider == "github":
        return {
            "client_id":     _env("GITHUB_OAUTH_CLIENT_ID"),
            "client_secret": _env("GITHUB_OAUTH_CLIENT_SECRET"),
            "authorize_url": "https://github.com/login/oauth/authorize",
            "token_url":     "https://github.com/login/oauth/access_token",
            "userinfo_url":  "https://api.github.com/user",
            "email_url":     "https://api.github.com/user/emails",  # GitHub email 要另外 call
            "scope":         "read:user user:email",
        }
    raise HTTPException(status_code=400, detail=f"不支援的 OAuth provider: {provider}")


def _callback_url(provider: str) -> str:
    return f"{_redirect_base()}/api/auth/oauth/{provider}/callback"


# ---------- /start: 把使用者送去 provider 授權頁 ----------
@router.get("/{provider}/start")
def oauth_start(provider: str):
    cfg = _provider_config(provider)
    if not cfg["client_id"] or not cfg["client_secret"]:
        raise HTTPException(
            status_code=500,
            detail=f"{provider} OAuth 未設定 (缺 client_id 或 client_secret 環境變數)",
        )
    params = {
        "client_id": cfg["client_id"],
        "redirect_uri": _callback_url(provider),
        "scope": cfg["scope"],
        "response_type": "code",
        # access_type=offline 是 Google 專用 (拿 refresh_token,本案不需要但無害)
    }
    if provider == "google":
        params["prompt"] = "select_account"
    return RedirectResponse(f"{cfg['authorize_url']}?{urlencode(params)}")


# ---------- /callback: provider 回頭後處理 ----------
@router.get("/{provider}/callback")
async def oauth_callback(provider: str, request: Request):
    code = request.query_params.get("code")
    error = request.query_params.get("error")
    if error or not code:
        return RedirectResponse(
            f"{_frontend_url()}?auth_error={error or 'no_code'}",
            status_code=302,
        )
    cfg = _provider_config(provider)

    # 1. code → access_token
    async with httpx.AsyncClient(timeout=15.0) as client:
        token_params = {
            "client_id":     cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "code":          code,
            "redirect_uri":  _callback_url(provider),
            "grant_type":    "authorization_code",
        }
        headers = {"Accept": "application/json"}
        token_resp = await client.post(cfg["token_url"], data=token_params, headers=headers)
        if token_resp.status_code != 200:
            return RedirectResponse(
                f"{_frontend_url()}?auth_error=token_exchange_failed",
                status_code=302,
            )
        token_data = token_resp.json()
        access_token = token_data.get("access_token")
        if not access_token:
            return RedirectResponse(
                f"{_frontend_url()}?auth_error=no_access_token",
                status_code=302,
            )

        # 2. fetch user info
        user_headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
        userinfo_resp = await client.get(cfg["userinfo_url"], headers=user_headers)
        if userinfo_resp.status_code != 200:
            return RedirectResponse(
                f"{_frontend_url()}?auth_error=userinfo_failed",
                status_code=302,
            )
        info = userinfo_resp.json()

        # GitHub 的 email 預設不在 /user 回傳裡 (隱私),要另外 call /user/emails
        if provider == "github":
            emails_resp = await client.get(cfg["email_url"], headers=user_headers)
            if emails_resp.status_code == 200:
                emails = emails_resp.json()
                primary = next((e for e in emails if e.get("primary") and e.get("verified")), None)
                if primary:
                    info["email"] = primary["email"]

    # 3. 標準化 user info
    if provider == "google":
        oauth_id = info.get("sub")
        email    = info.get("email")
        name     = info.get("name") or info.get("given_name")
        avatar   = info.get("picture")
    else:  # github
        oauth_id = str(info.get("id", ""))
        email    = info.get("email") or f"github_{oauth_id}@noemail.local"
        name     = info.get("name") or info.get("login")
        avatar   = info.get("avatar_url")

    if not oauth_id:
        return RedirectResponse(
            f"{_frontend_url()}?auth_error=no_user_id",
            status_code=302,
        )

    # 4. find / create user in DB
    db: Session = SessionLocal()
    try:
        user = (
            db.query(User)
            .filter(User.oauth_provider == provider, User.oauth_id == oauth_id)
            .first()
        )
        if not user:
            # 看看是不是已經有用同 email 註冊的 (允許 link;簡化版用 email 配對)
            user = db.query(User).filter(User.email == email).first()
            if user:
                # link OAuth 到既有帳號
                user.oauth_provider = provider
                user.oauth_id       = oauth_id
                if not user.avatar_url:
                    user.avatar_url = avatar
                if not user.display_name:
                    user.display_name = name
            else:
                user = User(
                    email=email,
                    display_name=name,
                    oauth_provider=provider,
                    oauth_id=oauth_id,
                    avatar_url=avatar,
                )
                db.add(user)
        user.last_login_at = datetime.utcnow()
        db.commit()
        db.refresh(user)
        jwt_token = create_access_token(user.id, user.email)
    finally:
        db.close()

    # 5. 302 回前端,帶 token
    return RedirectResponse(
        f"{_frontend_url()}?token={jwt_token}",
        status_code=302,
    )
