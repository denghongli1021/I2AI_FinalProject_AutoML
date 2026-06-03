---
title: I2AI AutoML
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: End-to-end AutoML platform — tabular + time series, HPO/NAS/Ensemble + web UI
---

# I2AI AutoML

清大「人工智慧概論」期末專題 — 端對端 AutoML 平台，支援表格分類/回歸與時序分類/回歸，涵蓋完整 HPO → NAS → CV → Ensemble 流程並附 Web UI。

---

## 功能概覽

| 類別 | 內容 |
|------|------|
| **任務類型** | 表格分類、表格回歸、時序分類（UCR 格式）、時序回歸 |
| **傳統 ML** | LightGBM、XGBoost、CatBoost、RandomForest、ExtraTrees、LogisticRegression、KNN |
| **深度學習** | MLP / TSNet、CNN1D / TCN、Transformer（SignalTransformer）/ PatchTST |
| **HPO** | Optuna TPE；Scout 快速篩選 → Full HPO（Tabular）；2-fold 評分（DL） |
| **NAS** | MLP 共享權重超網路（表格）；TSNet 4 算子搜尋（時序） |
| **Ensemble** | Nelder-Mead 加權融合 + Meta-Learner Stacking（LGBM/Ridge） |
| **可解釋性** | SHAP TreeExplainer（樹模型）、PermutationExplainer（DL/Ensemble） |
| **對照組** | AutoGluon baseline（分類/回歸/時序） |
| **Auth** | Email 密碼、Google OAuth、GitHub OAuth；訪客模式（無帳號可用） |
| **DB** | Supabase Postgres（線上）/ SQLite fallback（本地） |

---

## 系統架構

```
瀏覽器 (localhost:5500)
    │  REST / SSE
    ▼
FastAPI 後端 (localhost:8000)
    ├── /api/preprocess   → 資料審計 + 特徵工程
    ├── /api/train        → sklearn 多模型訓練
    ├── /api/train/pipeline/stream  → Daniel Pipeline（HPO/NAS/Ensemble，SSE 串流）
    ├── /api/train/autogluon/stream → AutoGluon baseline（SSE 串流）
    ├── /api/visualize/shap         → SHAP 視覺化
    ├── /api/predict/batch/stream   → 批次推論
    └── /api/auth/**                → 登入/OAuth
```

### Pipeline 執行流程

```
原始 CSV
  │
  ├─[前處理] robust_clean + FeatureBuilder（10 種特徵集）
  │
  ├─[2a] Tabular Scout HPO — 快速 3-Fold 篩除弱模型
  ├─[2b] Tabular Full HPO  — Optuna TPE 5-Fold CV
  │
  ├─[3]  NAS — MLPNASSearcher（表格）/ TSNASSearcher（時序）
  ├─[4]  MLP / TSNet Train HPO
  ├─[5]  CNN1D / TCN HPO
  ├─[6]  Transformer / PatchTST HPO（AMP 混合精度）
  │
  ├─[7]  5-Fold CV → OOF + Test 預測（artifacts/ 快取）
  ├─[8]  Ensemble A：Nelder-Mead 加權融合
  └─[9]  Ensemble B：Meta-Learner Stacking → 最終預測
```

---

## 目錄結構

```
.
├── index.html               # 前端主頁（Tailwind 預編譯）
├── css/tailwind.css         # Tailwind 編譯產物
├── js/                      # 前端 JS（app / api / auth / charts / ml-engine）
├── api/
│   ├── main.py              # FastAPI 入口（所有路由）
│   ├── bootstrap.py         # 環境初始化（dotenv / UTF-8 console）
│   ├── requirements.txt     # Python 依賴
│   ├── storage.py           # 統一 storage（guest in-memory / authed DB）
│   ├── auth/                # 登入 / OAuth / DB schema
│   ├── preprocess/          # 資料審計 + 特徵工程
│   ├── train/               # sklearn 訓練 + Daniel Pipeline wrapper + AutoGluon
│   │   └── pipeline/        # Pipeline 引擎（hpo / nas / train / ensemble / models）
│   └── visualize/           # SHAP + Plotly 視覺化
├── src/                     # 離線 Pipeline 引擎（pipeline.py / pipeline_time.py）
├── run_pipeline.py          # 離線批次入口（表格分類 + 回歸）
├── run_pipeline_time.py     # 離線批次入口（時序分類 + 回歸）
├── run_baseline.py          # AutoGluon 對照組
├── scripts/                 # 資料集下載（OpenML / UCR）
├── openml_cc18_data/        # 表格分類資料集
├── openml_regression_data/  # 表格回歸資料集
└── ucr_ts_80_new(時序資料)/ # UCR 時序資料集
```

---

## 快速開始（本地）

### 1. 環境

```powershell
# 使用 Anaconda 環境 ml_platform
conda activate ml_platform
pip install -r api/requirements.txt
```

### 2. 啟動後端

```powershell
# 終端 1
python -m uvicorn api.main:app --reload --port 8000
```

### 3. 啟動前端

```powershell
# 終端 2
python -m http.server 5500
```

### 4. 開啟瀏覽器

```
http://localhost:5500
```

> **Tailwind CSS**：若修改 `index.html` 的 class，需重新編譯：
> ```powershell
> .\tailwindcss3.exe -i css/tailwind-input.css -o css/tailwind.css --minify --config tailwind.config.js
> ```

---

## 線上版本

| 服務 | URL |
|------|-----|
| 前端（GitHub Pages） | https://denghongli1021.github.io/I2AI_FinalProject_AutoML/ |
| 後端（HuggingFace Spaces） | https://i2ai-automl-api.onrender.com |
| API 文件（Swagger） | https://i2ai-automl-api.onrender.com/docs |

---

## 環境變數（`.env`，選填）

```env
JWT_SECRET=<隨機字串>                  # Email 登入必填
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
GITHUB_OAUTH_CLIENT_ID=...
GITHUB_OAUTH_CLIENT_SECRET=...
DATABASE_URL=postgresql://...          # Supabase；不填則 fallback SQLite
FRONTEND_URL=http://localhost:5500
OAUTH_REDIRECT_BASE=http://localhost:8000
```

不設定任何環境變數也能執行，訪客模式下所有功能可用，資料存於 in-memory（重啟清空）。

本地開發想跳過 Supabase 延遲：

```powershell
$env:DB_LOCAL='1'; python -m uvicorn api.main:app --reload --port 8000
```

---

## 離線批次評估

```powershell
# 表格分類（OpenML-CC18 前 10 個）
python run_pipeline.py --batch --top-n 10

# 表格回歸（前 10 個）
python run_pipeline.py --batch --top-n 10 --reg-top-n 10

# 時序（UCR，各取 10 個 CLS + REG）
python run_pipeline_time.py --new-ts-batch --cls-top-n 10 --reg-top-n 10

# AutoGluon 對照組
python run_baseline.py --batch --top-n 10

# 合併三份結果
python merge_final_results.py
```

---

## 開發工具

```powershell
# 清空 DB 測試資料（保留帳號）
python tools/wipe_data.py --all --dry-run   # 預覽
python tools/wipe_data.py --all             # 執行

# API 文件（本地）
# http://127.0.0.1:8000/docs
# http://127.0.0.1:8000/redoc
```

---

## License

MIT
