"""SQLAlchemy DB setup + User model.

DATABASE_URL 環境變數決定連線目標:
  - 未設或設 sqlite → 本機 SQLite (api/auth.db)
  - postgresql://... → Supabase / 任何 Postgres
重啟後資料持久取決於 DB 是檔案還是雲端服務。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


def _get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        # 本機 fallback — 檔案放在 api/auth.db
        path = os.path.join(os.path.dirname(__file__), "..", "auth.db")
        path = os.path.abspath(path)
        return f"sqlite:///{path}"
    # Supabase / Render 給的 URL 有時開頭是 postgres:// (legacy),SQLAlchemy 要 postgresql://
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


_DATABASE_URL = _get_database_url()
_IS_SQLITE = _DATABASE_URL.startswith("sqlite")

# SQLite 多執行緒需要 check_same_thread=False;Postgres 不用
engine = create_engine(
    _DATABASE_URL,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    email           = Column(String(255), unique=True, index=True, nullable=False)
    display_name    = Column(String(120), nullable=True)
    password_hash   = Column(String(255), nullable=True)   # OAuth-only 使用者可為 None
    oauth_provider  = Column(String(20), nullable=True)    # 'google' | 'github' | None
    oauth_id        = Column(String(120), nullable=True)   # provider 給的 sub / id
    avatar_url      = Column(String(500), nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at   = Column(DateTime, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "displayName": self.display_name,
            "oauthProvider": self.oauth_provider,
            "avatarUrl": self.avatar_url,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }


def init_db() -> None:
    """app 啟動時呼叫 — 建表 (若已存在則 no-op)。"""
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Session:
    """with session_scope() as db: ... — 自動 commit / rollback / close。"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db():
    """FastAPI Depends 用。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
