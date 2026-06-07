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
| **前處理** | 雙軌（Tree / DL）+ 對抗驗證自動剔除漂移特徵 |
| **可解釋性** | SHAP TreeExplainer（樹模型）、PermutationExplainer（DL/Ensemble） |
| **對照組** | AutoGluon baseline（分類/回歸/時序） |
| **Auth** | Email 密碼、Google OAuth、GitHub OAuth；訪客模式（無帳號可用） |
| **DB** | Supabase Postgres（線上）/ SQLite WAL fallback（本地） |

---

## 實驗結果

### 時序分類（vs AutoGluon Baseline）

| 資料集 | Pipeline F1 | Baseline F1 | Δ |
|--------|-------------|-------------|---|
| CLS_ACSF1 | 0.7947 | 0.7667 | +0.028 |
| CLS_Adiac | 0.7336 | 0.6785 | +0.055 |
| CLS_AllGestureWiimoteX | 0.5335 | 0.4439 | +0.090 |
| CLS_AllGestureWiimoteY | 0.6426 | 0.5351 | +0.108 |
| CLS_AllGestureWiimoteZ | 0.5659 | 0.4020 | +0.164 |
| **平均** | **0.610** | **0.565** | **+8.0%** |

### 時序回歸（R²，vs AutoGluon Baseline）

| 資料集 | Pipeline R² | Baseline R² | Δ |
|--------|-------------|-------------|---|
| AcousticContaminationMadrid | 0.677 | 0.302 | +0.375 |
| AluminiumConcentration | 0.758 | 0.789 | −0.031 |
| AppliancesEnergy | −0.017 | −0.046 | +0.029 |
| AustraliaRainfall | 0.128 | 0.120 | +0.008 |

Pipeline 在 4 個資料集中勝出 3 個；訓練時間為 baseline 的 10–40 倍，反映完整 HPO/NAS/Ensemble 搜尋的運算成本。

---

## 系統架構

```
瀏覽器 (localhost:5500)
    │  REST / SSE
    ▼
FastAPI 後端 (localhost:8000)
    ├── /api/preprocess          → 資料審計 + 雙軌特徵工程 + 對抗驗證
    ├── /api/train               → sklearn 多模型訓練
    ├── /api/train/pipeline/stream  → Daniel Pipeline（HPO/NAS/Ensemble，SSE 串流）
    ├── /api/train/autogluon/stream → AutoGluon baseline（SSE 串流）
    ├── /api/visualize/shap      → SHAP 可解釋性視覺化
    ├── /api/predict/batch/stream → 批次推論
    └── /api/auth/**             → 登入/OAuth
```

### Pipeline 執行流程

```
原始 CSV
  │
  ├─[前處理] RobustDataCleaner + AutoRouter 雙軌分流
  │           Tree 軌（OrdinalEncoder，上限 800 維）
  │           DL 軌（StandardScaler + OHE，上限 300 維）
  │           + 對抗驗證自動剔除 Train/Test 漂移特徵
  │
  ├─[2a] Tabular Scout HPO — 快速 3-Fold 篩除弱模型，鎖定最佳特徵集
  ├─[2b] Tabular Full HPO  — Optuna TPE 5-Fold CV，依分數比例分配 trial
  │
  ├─[3]  NAS — One-Shot Supernet + 遺傳演化搜尋
  │            表格：MLPNASSearcher；時序：TSNASSearcher（4 算子）
  ├─[4]  MLP / TSNet Train HPO
  ├─[5]  CNN1D / TCN HPO
  ├─[6]  Transformer / PatchTST HPO（AMP 混合精度 1.5× 加速）
  │
  ├─[7]  5-Fold CV → OOF + Test 預測（artifacts/ .npy 快取）
  ├─[8]  Ensemble A：Nelder-Mead 加權融合（log-space Softmax 無約束優化）
  └─[9]  Ensemble B：Meta-Learner Stacking → 最終預測
```

---

## 目錄結構

```
.
├── index.html               # 前端主頁（Vanilla JS SPA，Glassmorphism UI）
├── js/                      # 前端 JS（app / api / auth / charts / ml-engine / data-engine）
├── css/style.css            # 客製化樣式（暗黑模式、玻璃擬物化、動畫）
├── api/
│   ├── main.py              # FastAPI 入口（所有路由 + SSE 串流管理）
│   ├── bootstrap.py         # 環境初始化（dotenv / UTF-8 console）
│   ├── requirements.txt     # Python 依賴
│   ├── storage.py           # 統一 storage（guest in-memory / authed SQLite/PG）
│   ├── auth/                # JWT / OAuth / SQLAlchemy DB schema（WAL 模式）
│   ├── preprocess/          # 資料審計 + 對抗驗證 + 特徵工程
│   ├── train/               # sklearn 訓練 + Daniel Pipeline + AutoGluon
│   └── visualize/           # SHAP + Plotly 視覺化（PNG 輸出）
├── src/                     # 離線 Pipeline 引擎
│   ├── pipeline.py          # 表格 AutoML 引擎（HPO/NAS/CV/Ensemble）
│   ├── pipeline_time.py     # 時序 AutoML 引擎
│   ├── hpo.py               # TabularHPO + DLHPO
│   ├── nas.py               # MLPNASSearcher / TSNASSearcher
│   ├── train.py             # run_cv（AMP 混合精度 + Mixup）
│   ├── ensemble.py          # NelderMeadBlender + MetaLearnerStacker
│   └── models/              # MLP / CNN1D / TCN / SignalTransformer / PatchTST
├── preprocessing/           # 全域雙軌前處理模組
│   ├── interface.py         # preprocess_for_training / preprocess_for_inference
│   ├── core/                # AutoRouter + PipelineAssembler + TSDataProcessor
│   └── utils/               # 對抗驗證 / 記憶體優化 / 資料健康診斷
├── visualization/           # SHAP 可解釋性（TreeExplainer / PermutationExplainer）
├── run_pipeline.py          # 離線批次入口（表格分類 + 回歸）
├── run_pipeline_time.py     # 離線批次入口（時序分類 + 回歸）
├── run_baseline.py          # AutoGluon 對照組
├── scripts/                 # 資料集下載（OpenML / UCR）
└── combined_results.csv     # Pipeline vs Baseline 批次評估結果
```

---

## 快速開始（本地）

### 1. 環境

```powershell
conda activate ml_platform
pip install -r api/requirements.txt
```

### 2. 啟動後端

```powershell
# 終端 1（不加 --reload 可避免 reloader 子行程問題）
C:\Users\Danie\.conda\envs\ml_platform\python.exe -m uvicorn api.main:app --port 8000
```

### 3. 啟動前端

```powershell
# 終端 2
C:\Users\Danie\.conda\envs\ml_platform\python.exe -m http.server 5500
```

### 4. 開啟瀏覽器

```
http://localhost:5500
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

## 環境變數（`.env`，選填）

```env
JWT_SECRET=<隨機字串>
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
GITHUB_OAUTH_CLIENT_ID=...
GITHUB_OAUTH_CLIENT_SECRET=...
DATABASE_URL=postgresql://...   # Supabase；不填則 fallback SQLite（WAL 模式）
FRONTEND_URL=http://localhost:5500
OAUTH_REDIRECT_BASE=http://localhost:8000
```

不設定任何環境變數也能執行，訪客模式下所有功能可用，資料存於 in-memory（重啟清空）。

---

## 線上版本

| 服務 | URL |
|------|-----|
| 前端（GitHub Pages） | https://denghongli1021.github.io/I2AI_FinalProject_AutoML/ |
| 後端（Render） | https://i2ai-automl-api.onrender.com |
| API 文件（Swagger） | https://i2ai-automl-api.onrender.com/docs |

---

## License

MIT
