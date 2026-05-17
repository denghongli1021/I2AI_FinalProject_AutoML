"""Auth module — email/password + Google/GitHub OAuth + JWT.

對外暴露:
  - router            : FastAPI router (mount 在 main.py)
  - get_current_user  : Depends 用 (回傳 User | None,guest 允許)
  - get_required_user : Depends 用 (沒登入丟 401)
  - init_db           : 啟動時建表
"""

from .db import init_db
from .dependencies import get_current_user, get_required_user
from .routes import router

__all__ = ["router", "get_current_user", "get_required_user", "init_db"]
