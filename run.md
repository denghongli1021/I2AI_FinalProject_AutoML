# I2AI AutoML 項目 — 本地測試指南

## 環境要求

- Python 3.8 以上(建議 3.9+)
- pip 或 conda
- Windows / macOS / Linux

---

## 第一步:環境設置

```powershell
# 1.1 創建虛擬環境(推薦)
python -m venv venv

# 1.2 激活虛擬環境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 1.3 升級 pip
python -m pip install --upgrade pip
```

---

## 第二步:數據收集(可選)

```powershell
# 2.1 收集 OpenML 數據集
python -m pip install openml pandas tqdm
python data_collect.py

# 2.2 收集時序數據集
python -m pip install aeon pandas tqdm
python data_collect_time.py
```

> 下載的數據將存放在 `Dataset/` 目錄下。

---

## 第三步:後端服務啟動

```powershell
# 3.1 安裝後端依賴
python -m pip install -r api/requirements.txt

# 3.2 (可選) 設定 .env — 啟用登入功能 / OAuth
cp .env.example .env
# 編輯 .env 填入 JWT_SECRET 等;沒設也能跑,只是登入會失效或用 SQLite fallback

# 3.3 啟動 FastAPI 服務器 (終端 1)
python -m uvicorn api.main:app --reload --port 8000
```

**服務器位址**:

| 用途 | URL |
|---|---|
| API 主頁 | http://127.0.0.1:8000/ |
| Swagger 文檔 | http://127.0.0.1:8000/docs |
| ReDoc 文檔 | http://127.0.0.1:8000/redoc |
| 線上版本 | https://i2ai-automl-api.onrender.com |

---

## 登入功能設定(可選)

支援三種登入方式:
- **Email + 密碼**(本地註冊)
- **Google OAuth**
- **GitHub OAuth**

> 不設定也能用 — 匯名訪客可以使用所有功能,只是資料不會跨裝置同步。

### A. Email / 密碼

不需任何外部設定,只要 `JWT_SECRET` 有設即可。

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
# 把輸出貼到 .env 的 JWT_SECRET=
```

### B. Google OAuth

1. 到 [Google Cloud Credentials](https://console.cloud.google.com/apis/credentials)
2. 建立 OAuth 用戶端 ID,類型「Web 應用程式」
3. 「已授權的重新導向 URI」加:
   - 本機:`http://localhost:8000/api/auth/oauth/google/callback`
   - Render:`https://YOUR-APP.onrender.com/api/auth/oauth/google/callback`
4. 拿到 Client ID / Secret,填到 `.env` 的 `GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_SECRET`

### C. GitHub OAuth

1. 到 [GitHub Developer Settings](https://github.com/settings/developers) → OAuth Apps → New
2. Authorization callback URL:
   - 本機:`http://localhost:8000/api/auth/oauth/github/callback`
   - Render:`https://YOUR-APP.onrender.com/api/auth/oauth/github/callback`
3. 拿到 Client ID,點 **Generate a new client secret** 拿 Secret
4. 填到 `.env` 的 `GITHUB_OAUTH_CLIENT_ID` / `GITHUB_OAUTH_CLIENT_SECRET`

### D. Supabase(持久化 DB,Render 重啟也不會清空)

1. 到 [Supabase](https://supabase.com) 建立免費 project
2. Project Settings → Database → Connection string → URI
3. 把 `[YOUR-PASSWORD]` 替換為實際密碼
4. 整串貼到 `.env` 的 `DATABASE_URL=`

> 不設這個 → fallback 用本機 SQLite (`api/auth.db`),Render 重啟就清空。

### E. FRONTEND_URL / OAUTH_REDIRECT_BASE

- `OAUTH_REDIRECT_BASE`:後端對外網址(本機 `http://localhost:8000`,Render 是 onrender.com 那個)
- `FRONTEND_URL`:前端對外網址(本機 `http://localhost:5500`,線上 GitHub Pages URL)

> 兩個都要跟你註冊 OAuth callback 時填的對得起來。

---

## 第四步:前端啟動

```powershell
# 4.1 啟動本地 HTTP 服務器 (終端 2)
python -m http.server 5500

# 4.2 訪問前端
# 打開瀏覽器 → http://localhost:5500
```

---

## 測試流程

### 測試用例 1 — 完整工作流程

1. 訪問 http://localhost:5500(前端)
2. 上傳 CSV 文件(或使用預置數據)
3. 進行數據預處理和審計
4. 訓練機器學習模型
5. 查看可視化結果(特徵重要性、SHAP 解釋等)

### 測試用例 2 — 後端 API 直接測試

1. 訪問 http://127.0.0.1:8000/docs
2. 展開各端點查看參數說明
3. 使用 Swagger UI 直接測試 API

### 測試用例 3 — 本地數據測試

1. 執行 `data_collect.py` 下載數據
2. 前端上傳下載的 CSV 文件
3. 完整執行工作流程

---

## 主要模組

```
api/
├── main.py                  # FastAPI 應用程序入口
├── requirements.txt         # Python 依賴
├── storage.py               # 統一 storage 層 (guest in-memory / authed DB)
├── store.py                 # in-memory dict (guest 用)
├── auth/                    # 登入 / OAuth / DB schema
│   └── db.py                # SQLAlchemy models
├── preprocess/              # 數據預處理模組
├── train/                   # 模型訓練模組
│   ├── train.py             # sklearn 多模型訓練
│   ├── daniel_runner.py     # Daniel pipeline subprocess wrapper
│   └── pipeline/            # Daniel pipeline 完整原始碼
└── visualize/               # 可視化模組 (SHAP)

前端:
├── index.html               # 前端主頁
├── js/                      # JavaScript 邏輯
│   ├── app.js
│   ├── api.js
│   ├── auth.js
│   ├── charts.js
│   ├── ml-engine.js
│   └── data-engine.js
└── css/style.css

數據收集:
├── data_collect.py          # OpenML 數據收集
└── data_collect_time.py     # 時序數據收集

開發者工具:
└── tools/
    └── wipe_data.py         # DB 資料清理 (dev only)
```

---

## 開發者工具:DB 資料清理

> 給 dev 用,**不對外暴露**(沒有 web endpoint,使用者看不到)。

`tools/wipe_data.py` 清空 DB 的 ML state:
- `datasets`
- `preprocessors`
- `models`
- `training_runs`
- `prediction_artifacts`

給開發測試「排除舊資料干擾」用。

```powershell
# 預覽不執行 (推薦先跑一次,看會刪幾筆)
python tools/wipe_data.py --all --dry-run

# 清掉所有 user 的 ML 資料 (帳號保留)
python tools/wipe_data.py --all

# 只清指定 user (用 user_id)
python tools/wipe_data.py --user 2

# 用 email 找 user
python tools/wipe_data.py --user denghongli1021@gmail.com

# 連 users 表也清 (注意:會把所有帳號刪掉,登入要重註冊)
python tools/wipe_data.py --all --include-users
```

**執行流程**:
1. 印出連線目標(從 `.env` 讀 `DATABASE_URL`)
2. 列出每張表預計刪除筆數
3. 等 user 輸入 `YES`(大寫)才真的執行
4. 順序:`prediction_artifacts` → `models` → `training_runs` → `preprocessors` → `datasets`

**不會動的**:
- `users` 表(除非加 `--include-users`)
- `.env` / 程式碼 / 設定

---

## 常見命令速查表

```powershell
# 安裝所有依賴(後端 + 數據收集)
python -m pip install -r api/requirements.txt openml aeon pandas tqdm shap plotly

# 同時啟動後端和前端(需要 2 個終端)
# 終端 1:
python -m uvicorn api.main:app --reload --port 8000
# 終端 2:
python -m http.server 5500

# 運行測試
python -m pytest api/visualize/tests/

# 清空 DB 資料(dev tool)
python tools/wipe_data.py --all --dry-run
python tools/wipe_data.py --all
```

**查看 API 文檔**:
- Swagger:http://127.0.0.1:8000/docs
- ReDoc:http://127.0.0.1:8000/redoc

---

## 故障排查

| 問題 | 解決 |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | `python -m pip install -r api/requirements.txt` |
| Port 8000 已被佔用 | `python -m uvicorn api.main:app --reload --port 8001` |
| 前端無法連接後端 | 確認後端服務已啟動、檢查防火牆設定、確保 CORS 已配置 |
| 數據集下載失敗 | 檢查網路連接、手動下載數據到 `Dataset/`、或使用本地 CSV |
| DB 連線失敗 | 確認 `.env` 的 `DATABASE_URL` 正確;沒設會 fallback 到 SQLite |
| `LogisticRegression is not JSON serializable` | 訓練 bundle 內藏了 sklearn estimator,已用 `default=str` 防禦 |
| Voting / Stacking 模型存不進 DB | pickle 太大時自動跳過(>20MB),洞察頁仍正常顯示 |
| 切換瀏覽器登入後看不到舊資料 | hydration 會自動從 DB 抓 — 若還是空,確認 `/api/training-runs` 跟 `/api/models` 200 回覆 |
