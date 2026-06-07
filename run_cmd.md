視窗 1 — 後端（FastAPI）

conda activate ml_platform
cd /d d:\UserData\claude_project\人工智慧project
python -m uvicorn api.main:app --port 8000
視窗 2 — 前端（靜態伺服器）

conda activate ml_platform
cd /d d:\UserData\claude_project\人工智慧project
python -m http.server 5500

http://localhost:5500/