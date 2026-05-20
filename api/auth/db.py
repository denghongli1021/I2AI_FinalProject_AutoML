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

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer,
    LargeBinary, String, Text, create_engine, Index,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker, relationship


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

# Supabase 免費版 session pooler (port 5432) 整個專案上限 15 連線。
# 多個部署 (HF Space + Render + local dev) 同時開,SQLAlchemy 預設 pool_size=5 + max_overflow=10
# 三個 server 加起來輕鬆超過 15 → 連 init_db 都進不了 DB。
# 縮小 pool 確保單一 server 最多 5 條,讓多副本可以共存。
# (理想是改用 transaction pooler port 6543,但要改 DATABASE_URL secret;這裡先把 pool 縮小當作雙保險)
_PG_POOL_KWARGS = {} if _IS_SQLITE else {"pool_size": 2, "max_overflow": 3, "pool_recycle": 1800}

# SQLite 多執行緒需要 check_same_thread=False;Postgres 不用
engine = create_engine(
    _DATABASE_URL,
    connect_args={"check_same_thread": False} if _IS_SQLITE else {},
    pool_pre_ping=True,
    **_PG_POOL_KWARGS,
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


# ============================================================
# ML state tables — authed users 的 dataset / preprocessor / model / 訓練紀錄
# Guests 走 api/store.py 的 in-memory dict,不會碰這些表
# ============================================================

class Dataset(Base):
    __tablename__ = "datasets"
    id              = Column(String(36), primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    file_name       = Column(String(255))
    row_count       = Column(Integer)
    col_count       = Column(Integer)
    headers_json    = Column(Text)                     # JSON list of column names
    csv_blob        = Column(LargeBinary)              # 原始 CSV bytes (使用時 pd.read_csv 還原)
    analysis_json   = Column(Text, nullable=True)      # 欄位 type / missing / unique 等快取
    extras_json     = Column(Text, nullable=True)      # health_score, correlation, processing_log
    created_at      = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    __table_args__ = (Index("ix_datasets_user_created", "user_id", "created_at"),)


class Preprocessor(Base):
    __tablename__ = "preprocessors"
    id              = Column(String(36), primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    dataset_id      = Column(String(36), ForeignKey("datasets.id"), index=True)
    target          = Column(String(120))
    test_size       = Column(Float, default=0.2)
    feature_names_json = Column(Text)
    preprocessor_pkl   = Column(LargeBinary)           # pickled sklearn ColumnTransformer
    train_test_pkl     = Column(LargeBinary)           # pickled tuple (X_tr_df, X_te_df, y_tr, y_te)
    created_at      = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)


class Model(Base):  # sklearn 引擎用;pipeline 不存到這裡 (沒保留 trained estimator)
    __tablename__ = "models"
    id              = Column(String(36), primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    dataset_id      = Column(String(36), index=True, nullable=True)
    preprocessor_id = Column(String(36), index=True, nullable=True)
    training_run_id = Column(String(36), ForeignKey("training_runs.id"), index=True, nullable=True)
    algorithm       = Column(String(50))               # 'xgboost', 'random_forest', ...
    name            = Column(String(120))              # 顯示用
    data_source     = Column(String(20))               # 'raw' | 'preprocessed'
    task_type       = Column(String(20))               # 'classification' | 'regression'
    target          = Column(String(120))
    bundle_json     = Column(Text)                     # metrics + featureNames + importance + testTrue/Pred
    hyperparameters_json = Column(Text, nullable=True) # estimator.get_params() 結果
    estimator_pkl   = Column(LargeBinary)              # pickled estimator
    scaler_pkl      = Column(LargeBinary)              # pickled StandardScaler
    x_test_pkl      = Column(LargeBinary, nullable=True)  # for SHAP visualizer
    feature_names_json = Column(Text)
    test_score      = Column(Float, index=True)        # 排序用
    train_time_ms   = Column(Float)
    created_at      = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)


class TrainingRun(Base):  # sklearn + pipeline 兩個引擎共用
    __tablename__ = "training_runs"
    id              = Column(String(36), primary_key=True)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    dataset_id      = Column(String(36), index=True)
    dataset_name    = Column(String(255))              # 冗餘存,方便 list 顯示
    engine          = Column(String(20), index=True)   # 'sklearn' | 'pipeline'
    target          = Column(String(120), index=True)
    task_type       = Column(String(20))
    sources_json    = Column(Text)                     # ["raw", "preprocessed"]
    options_json    = Column(Text)                     # 完整訓練設定,給 reproduce 用
    results_summary_json = Column(Text)                # sklearn: top model names + scores / pipeline: per-source
    model_ids_json  = Column(Text, nullable=True)      # sklearn 才有 (對應 Model.id 列表)
    status          = Column(String(20))               # 'running' | 'completed' | 'failed'
    error_msg       = Column(Text, nullable=True)
    has_predictions = Column(Boolean, default=False)   # pipeline option B 上傳了 test.csv
    started_at      = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    finished_at     = Column(DateTime, nullable=True)
    elapsed_sec     = Column(Float, nullable=True)
    # 即時進度 (sklearn / pipeline 訓練中,SSE event 旁路寫入 DB,給其他分頁/裝置 poll)
    progress_pct    = Column(Integer, default=0)        # 0~100
    current_step    = Column(String(120), nullable=True)   # 例如 "訓練 XGBoost" / "[原始] Scout XGB"
    latest_log_json = Column(Text, nullable=True)       # 最近 ~30 行 log,JSON list
    last_seen_at    = Column(DateTime, nullable=True)   # 上次 SSE 寫進度的時間,給 staleness 判斷


class PredictionArtifact(Base):  # pipeline Option B 的 test.csv 輸入 + 預測輸出
    __tablename__ = "prediction_artifacts"
    id              = Column(String(36), primary_key=True)
    training_run_id = Column(String(36), ForeignKey("training_runs.id"), index=True, nullable=False)
    user_id         = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    kind            = Column(String(40))               # 'input' | 'submission_raw' | 'submission_preprocessed' | 'submission_upload' | 'submission_proba'
    file_name       = Column(String(255))
    content_blob    = Column(LargeBinary)              # CSV bytes,直接存 DB
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)


def init_db() -> None:
    """app 啟動時呼叫 — 建表 (若已存在則 no-op) + 必要的線上欄位擴寬 / 補欄位。"""
    Base.metadata.create_all(bind=engine)
    # Postgres 不會自動跟著 model 改變欄位寬度 / 加新欄位;這裡做 idempotent migration。
    if not _IS_SQLITE:
        from sqlalchemy import text
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE prediction_artifacts "
                "ALTER COLUMN kind TYPE VARCHAR(40)"
            ))
            # 進度即時欄位 (新加,IF NOT EXISTS 才安全)
            conn.execute(text("""
                ALTER TABLE training_runs
                    ADD COLUMN IF NOT EXISTS progress_pct INTEGER DEFAULT 0,
                    ADD COLUMN IF NOT EXISTS current_step VARCHAR(120),
                    ADD COLUMN IF NOT EXISTS latest_log_json TEXT,
                    ADD COLUMN IF NOT EXISTS last_seen_at TIMESTAMP
            """))


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
