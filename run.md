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

> 不設定也能用 — 匿名訪客可以使用所有功能,只是資料不會跨裝置同步。

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

#### 臨時切回本地 SQLite(不動 .env)

開發時想避開雲端 DB 的延遲 / quota / statement_timeout,設環境變數 `DB_LOCAL=1` 就會**強制**走本地 SQLite,完全忽略 `.env` 裡的 `DATABASE_URL`:

```powershell
# PowerShell (一個 session 有效)
$env:DB_LOCAL='1'
python -m uvicorn api.main:app --reload --port 8000

# 或一行
$env:DB_LOCAL='1'; python -m uvicorn api.main:app --reload --port 8000
```

啟動 log 會看到 `[db] DB_LOCAL=1 → 強制使用本機 SQLite: ...api/auth.db`。關掉 terminal 或新開一個就回 Supabase。

### E. FRONTEND_URL / OAUTH_REDIRECT_BASE

- `OAUTH_REDIRECT_BASE`:後端對外網址(本機 `http://localhost:8000`,Render 是 onrender.com 那個)
- `FRONTEND_URL`:前端對外網址(本機 `http://localhost:5500`,線上 GitHub Pages URL)

> 兩個都要跟你註冊 OAuth callback 時填的對得起來。

---

## 第四步:前端啟動

```powershell
# 4.1 (改 index.html 的 Tailwind class 時才需要) 重新編譯 Tailwind CSS
# 沒裝過 node 依賴 → 先 `npm install`
npm run build:css       # 一次性編譯
# 或開發時持續監聽
npm run watch:css

# 4.2 啟動本地 HTTP 服務器 (終端 2)
python -m http.server 5500

# 4.3 訪問前端
# 打開瀏覽器 → http://localhost:5500
```

> Tailwind CSS 改用 CI 預編譯(舊版 CDN 已移除)。沒跑 `build:css` 之前新加的 class 不會生效。`tailwind.config.js` 的 `content` 設定會掃 `index.html` 跟 `js/**/*.js`。

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

### 測試用例 4 — 排行榜批次預測

1. 完成「實驗室 → 訓練」拿到模型
2. 切到「模型排行榜」
3. **header 中央**點「測試 CSV」上傳 test.csv,(選填)再點「submission 範本」
4. 任一列點「**預測**」按鈕 → 後端跑 batch predict → 自動下載 `submission.csv`(有範本)或 `{原檔名}_predicted.csv`(沒範本)
5. Pipeline ensemble / sklearn 引擎都支援。Pipeline 預處理 source 的 model 會自動套同個 ColumnTransformer 把 raw test → transformed → predict
6. 顯示「預測 (需重訓)」的灰字按鈕是舊版 placeholder Pipeline,DB 沒 estimator → 重訓一次就有

### 測試用例 5 — Daniel pipeline ensemble 持久化

1. 實驗室選「Pipeline 引擎」訓練(任務類型 + 資料來源任選)
2. 訓練完成自動 re-hydrate from DB → 排行榜出現「[原始/預處理] Pipeline Ensemble」
3. **重新登入 / 換瀏覽器** → ensemble 還在,可以繼續批次預測 / 跑 SHAP
4. Insights 頁的 SHAP 區塊對 ensemble 走 PermutationExplainer(慢但能用,約 30s~3min)

---

## 主要模組

```text
api/
├── main.py                  # FastAPI 應用程序入口
├── bootstrap.py             # 集中環境設定:UTF-8 console / load_dotenv / subprocess env helper
├── requirements.txt         # Python 依賴
├── storage.py               # 統一 storage 層 (guest in-memory / authed DB)
├── store.py                 # in-memory dict (guest 用)
├── auth/                    # 登入 / OAuth / DB schema
│   └── db.py                # SQLAlchemy models (含 idempotent 線上 migration)
├── preprocess/              # 數據預處理模組
│   ├── interface.py         # 對外:run_data_audit / preprocess_for_training / preprocess_for_inference
│   ├── preprocess.py        # 給 /api/preprocess (CSV → analysis) 用
│   └── processors/feature_generator.py   # MISelector + RobustDataCleaner 等
├── train/                   # 模型訓練模組
│   ├── train.py             # sklearn 多模型訓練
│   ├── daniel_runner.py     # Daniel pipeline subprocess wrapper
│   └── pipeline/            # Daniel pipeline 完整原始碼
│       ├── pipeline.py / pipeline_time.py  # 分類 / 時序回歸主引擎
│       ├── src/             # hpo, train, nas, preprocess, models, ensemble
│       └── _runner_entry.py # subprocess 入口 (含 ensemble bundle dump)
└── visualize/               # 可視化模組 (SHAP + Plotly)

前端:
├── index.html               # 前端主頁 (Tailwind 預編譯)
├── tailwind.config.js       # Tailwind 設定 (掃 index.html + js/**/*.js)
├── package.json             # npm run build:css / watch:css
├── css/tailwind.css         # 預編譯產物 (git tracked)
├── js/                      # JavaScript 邏輯
│   ├── app.js
│   ├── api.js
│   ├── auth.js
│   ├── charts.js
│   ├── ml-engine.js
│   └── data-engine.js
└── new-ui/                  # React 新介面 (Babel Standalone)
    ├── app.jsx
    ├── page-*.jsx
    └── api-client.js

數據收集:
├── data_collect.py          # OpenML 數據收集
└── data_collect_time.py     # 時序數據收集

開發者工具:
└── tools/
    └── wipe_data.py         # DB 資料清理 (dev only)

訓練產出 (Daniel pipeline 中間檔,可清):
└── api/train/pipeline/artifacts/api/<csv_name>/<timestamp>/
    ├── ensemble_bundle.pkl  # 整套 ensemble (folds + blender + stacker)
    ├── {tag}_oof.npy / _test.npy
    └── {tag}_best_model.pkl / .pt
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
| Supabase `statement_timeout` (ALTER TABLE / DROP) | `preprocessors` 大 blob 表會撞時限。臨時開發改用 `$env:DB_LOCAL='1'` 走本地 SQLite |
| `LogisticRegression is not JSON serializable` | 訓練 bundle 內藏了 sklearn estimator,已用 `default=str` 防禦 |
| Voting / Stacking 模型存不進 DB | pickle 太大時自動跳過(>20MB,Daniel ensemble 是 200MB),洞察頁仍正常顯示 |
| Daniel Pipeline 顯示「預測 (需重訓)」 | 舊版 placeholder(DB 沒 estimator)。重新訓練一次即可拿到完整 ensemble bundle |
| 批次預測 `model 不存在` | 點到 placeholder model 的 fake id (`daniel_db_*`)。同上,重訓即可 |
| 批次預測 `CSV 缺少 N 個欄位` | test.csv 欄位要跟訓練時的「原始欄位」一致。預處理 source 訓的模型會自動套 transform,raw source 訓的就直接用原欄 |
| Tailwind 新加的 class 沒效果 | `npm run build:css` 重編,或 `npm run watch:css` 開發時 auto rebuild |
| 切換瀏覽器登入後看不到舊資料 | hydration 會自動從 DB 抓 — 若還是空,確認 `/api/training-runs` 跟 `/api/models` 200 回覆 |
| 時間顯示偏 8 小時 | 已修(`storage._utc_naive_to_epoch`);還有偏就 hard refresh 撈最新 JS |
