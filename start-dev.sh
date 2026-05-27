#!/usr/bin/env bash
# =============================================================================
# 本地測試環境一鍵啟動:後端 (FastAPI + 本地 SQLite) + 前端 (靜態檔伺服器)
#
# 用法 (在 Git Bash):
#     bash start-dev.sh        (Ctrl+C 一起關閉前後端)
# =============================================================================
set -u

# --- 在 Windows 上,`bash` 預設是 WSL;但專案套件 (uvicorn 等) 裝在 Windows Python ---
# WSL 跑 Windows Python 會有檔案監看 / env / Ctrl+C 的 interop 問題,Git Bash 則很順。
# 所以偵測到 WSL 就自動改用 Git Bash 重跑 (你照樣打 `bash start-dev.sh` 即可)。
if grep -qi microsoft /proc/version 2>/dev/null; then
  WIN_GIT_BASH="/mnt/c/Program Files/Git/bin/bash.exe"
  if [ -x "$WIN_GIT_BASH" ]; then
    echo "⚠️  偵測到 WSL → 自動改用 Git Bash 執行 (對 Windows Python 較穩)..."
    exec "$WIN_GIT_BASH" "${0/#\/mnt/}"   # /mnt/c/... → /c/... 給 Git Bash
  else
    echo "❌ 你在 WSL 跑,但 WSL 沒有專案套件,也找不到 Git Bash。"
    echo "   請改用 PowerShell 跑後端,或安裝 Git for Windows。"
    exit 1
  fi
fi

# 切到本 script 所在目錄 (= 專案根目錄)
cd "$(dirname "$0")"

BACKEND_PORT=8000
FRONTEND_PORT=5500

# --- 找「有裝專案套件 (uvicorn) 的」python ---
# 注意:Git Bash 內建的 /usr/bin/python3 是 MSYS2 的,沒有專案套件;
# 真正有 uvicorn 的是 Windows Python。所以用「能不能 import uvicorn」當判斷,
# 並優先掃 Windows Python 的安裝路徑 (bare 名稱在 Git Bash 常找不到)。
PY=""
for c in \
    /c/Users/*/AppData/Local/Programs/Python/Python*/python.exe \
    "$HOME"/AppData/Local/Programs/Python/Python*/python.exe \
    python py python3 ; do
  "$c" -c "import uvicorn" >/dev/null 2>&1 && { PY="$c"; break; }
done
if [ -z "$PY" ]; then
  echo "❌ 找不到有裝 uvicorn 的 python。"
  echo "   (Git Bash 內建的 python3 沒有專案套件,需要你的 Windows Python。)"
  echo "   解法:改用 PowerShell 跑後端,或把有裝套件的 Windows Python 加進 PATH。"
  exit 1
fi

[ -f "api/main.py" ] || { echo "❌ 請在專案根目錄執行 (找不到 api/main.py)"; exit 1; }

BACK_PID=""
FRONT_PID=""

# 收尾:★ 先解除 trap 再 kill,否則 kill 會再次觸發 trap → 無限「關閉中...」
cleanup() {
  trap - INT TERM
  echo
  echo "關閉中..."
  [ -n "$BACK_PID" ]  && kill "$BACK_PID"  2>/dev/null
  [ -n "$FRONT_PID" ] && kill "$FRONT_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

echo "=================================================="
echo " 啟動本地測試環境   (Ctrl+C 關閉)"
echo "   後端: http://127.0.0.1:${BACKEND_PORT}"
echo "   前端: http://127.0.0.1:${FRONTEND_PORT}"
echo "   python = ${PY}"
echo "=================================================="

# --- 後端:DB_LOCAL=1 → 本地 SQLite,不連雲端 ---
# ⚠️ 故意不開 --reload:在 Windows Git Bash 下,uvicorn reload 時送出的 SIGINT 會擴散到
# 整個 process group → 連 bash script 和前端伺服器一起被拉下水(畫面會看到「關閉中...」)。
# 要重啟後端就 Ctrl+C 整個 script,再 `bash start-dev.sh` 一次。
DB_LOCAL=1 "$PY" -m uvicorn api.main:app --port "$BACKEND_PORT" &
BACK_PID=$!

# --- 前端:綁 127.0.0.1 (避開 Windows localhost→IPv6 連不到後端的雷) ---
"$PY" -m http.server "$FRONTEND_PORT" --bind 127.0.0.1 &
FRONT_PID=$!

# 給 2 秒啟動,確認兩個都還活著;有人掛掉就報錯收尾 (不要卡在無限迴圈)
sleep 2
if ! kill -0 "$BACK_PID" 2>/dev/null; then
  echo "❌ 後端啟動失敗 — 最常見是 port ${BACKEND_PORT} 已被占用 (你可能已經有一個後端在跑),"
  echo "   或 uvicorn 未安裝。請看上方錯誤訊息。"
  cleanup
fi
if ! kill -0 "$FRONT_PID" 2>/dev/null; then
  echo "❌ 前端啟動失敗 — port ${FRONTEND_PORT} 可能被占用。請看上方錯誤訊息。"
  cleanup
fi

echo
echo "✅ 前後端都起來了 → 打開 http://127.0.0.1:${FRONTEND_PORT}"
echo "   → 設定開「本地模式」(API 指向 http://127.0.0.1:${BACKEND_PORT})"
echo "   → 本地 DB 全新,第一次先註冊 email/密碼帳號"
echo

# 等背景行程 (Ctrl+C 由 trap 收尾)
wait
cleanup
