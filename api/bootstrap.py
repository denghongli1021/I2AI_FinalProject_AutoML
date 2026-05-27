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
