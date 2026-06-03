"""
api/bootstrap.py — 集中所有「環境變數」與「runtime 設定」
=========================================================
import 這個模組就會自動執行 (import 副作用):
  1. Windows 終端強制 UTF-8 (避免印中文 / emoji 亂碼)
  2. 載入 .env (本地開發;Render / HF Space 是平台直接注入,跳過)

並對外提供:
  subprocess_env(extra=None) -> dict
      回傳一份適合給 daniel pipeline subprocess 用的 env dict,
      已內建 PYTHONIOENCODING / PYTHONUNBUFFERED 等必要設定。

使用方式:
  api/main.py  最頂端就要 import:  from api import bootstrap  # noqa: F401
  其他模組需要 subprocess env 時:  from api.bootstrap import subprocess_env

⚠ 注意 import 順序:dotenv 必須在 db.py / 任何讀 os.environ 的模組之前
   執行,否則 DATABASE_URL 抓不到。
"""
from __future__ import annotations

import os
import sys
from typing import Optional


# ────────────────────────────────────────────────────────────────
# 1. Windows 終端 UTF-8
#    cp950 / cp1252 預設碰到中文 / emoji 會炸 UnicodeEncodeError
# ────────────────────────────────────────────────────────────────
def _force_utf8_console() -> None:
    if not hasattr(sys.stdout, "reconfigure"):
        return
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ────────────────────────────────────────────────────────────────
# 2. 載 .env
#    python-dotenv 預設不覆蓋現有 env var (shell 設的會贏),
#    所以 $env:DB_LOCAL='1' 之類的 shell override 仍然有效。
# ────────────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


# ────────────────────────────────────────────────────────────────
# 3. subprocess env helper
#    給 daniel_runner.py 啟動 pipeline subprocess 用。
#    集中在這裡,以後要加 TQDM_DISABLE / CUDA_VISIBLE_DEVICES 等
#    都只動這一個函式。
# ────────────────────────────────────────────────────────────────
def subprocess_env(extra: Optional[dict] = None) -> dict:
    """
    回傳給 subprocess.Popen(env=...) 用的 dict。

    內建:
      PYTHONIOENCODING=utf-8  : subprocess 印中文不亂碼
      PYTHONUNBUFFERED=1      : print 立刻 flush,SSE log 才能即時送
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


# ────────────────────────────────────────────────────────────────
# Import 時自動執行 (重要:dotenv 必須在 db.py 之前跑)
# ────────────────────────────────────────────────────────────────
_force_utf8_console()
_load_dotenv()


# ────────────────────────────────────────────────────────────────
# 4. Runtime 設定常數
#    放在 _load_dotenv() 之後,確保 .env 已生效再讀 env var。
# ────────────────────────────────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    """讀環境變數整數值;格式錯誤就 fallback 預設值。"""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[bootstrap] env {name}={raw!r} 不是整數,使用預設 {default}", file=sys.stderr)
        return default


# DB 模型 pickle blob 大小上限 (MB)。
# -1 = 不限制 (DB 塞不下時自動 fallback 到 file blob);可被環境變數 MODEL_BLOB_MAX_MB 覆寫。
# 用在 storage.save_model() — 包括 sklearn estimator / scaler / X_test 跟 Daniel ensemble bundle。
MODEL_BLOB_MAX_MB: int = _env_int("MODEL_BLOB_MAX_MB", -1)


def model_blob_max_bytes() -> Optional[int]:
    """回傳 blob 大小上限 (bytes);None = 不限制。"""
    if MODEL_BLOB_MAX_MB < 0:
        return None
    return MODEL_BLOB_MAX_MB * 1024 * 1024


# 大 blob 改寫到檔案系統的門檻 (MB)。
# SQLite 單 row 上限 ~1GB、Postgres bytea 加 TOAST 上限也是 1GB,
# 超過這個門檻就改寫到 MODEL_BLOB_DIR 下的檔案,DB 只存 "FILEBLOB:" + 絕對路徑。
# 預設 50MB:典型 sklearn estimator 跟小型 ensemble 都會塞 DB;Daniel ensemble 跟大資料集自動進檔案。
MODEL_BLOB_FILE_THRESHOLD_MB: int = _env_int("MODEL_BLOB_FILE_THRESHOLD_MB", 50)

# File blob 落地的目錄。預設專案根目錄下的 model_blobs/,可被環境變數 MODEL_BLOB_DIR 覆寫。
# 注意:這個目錄要列入 .gitignore;備份策略要把它一起備走。
MODEL_BLOB_DIR: str = os.environ.get("MODEL_BLOB_DIR") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "model_blobs",
)


def model_blob_file_threshold_bytes() -> int:
    """回傳大 blob 改寫到檔案的門檻 (bytes)。"""
    return max(0, MODEL_BLOB_FILE_THRESHOLD_MB) * 1024 * 1024
