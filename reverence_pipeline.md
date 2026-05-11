# AI HW3 Pipeline Architecture

本專案實作了一個完整的自動化機器學習 Pipeline，用於解決 15 個類別的 Tabular / 訊號分類任務。架構設計包含了資料處理、特徵工程、多樣化模型（傳統機器學習與深度學習）、以及進階的模型融合技術（Ensemble）。

## 1. 核心設定與資料載入 (`src/config.py`, `src/data.py`)
- **環境設定 (`config.py`)**：統一管理亂數種子 (`SEED`)、資料夾路徑 (`artifacts`, `submissions`)、並自動偵測 CPU/GPU (`DEVICE`)。
- **資料切割 (`data.py`)**：採用 `StratifiedKFold(n_splits=5)`，確保切分出來的 Fold 能夠維持原本 15 個類別的資料比例。
- **PyTorch 整合**：客製化 `TabularDataset` 與 `make_loader`，方便將資料批次化（Batching）餵給深度學習模型。

## 2. 嚴謹的特徵工程 (`src/preprocess.py`)
為了避免資料洩漏（Data Leakage），實作了 `FeatureBuilder` 類別，**只在 Training Set 上做 Fit，再 Transform 到 Val/Test Set**。
流程包含：`SimpleImputer(median)` 補值 -> `IQR Clipping` 去除極端值 -> 提取特徵 -> `StandardScaler` 標準化。

支援多種特徵組合 (`feature_set`)：
- `raw`: 原始的 1280 維特徵。
- `signal`: 將 `raw` 特徵加上「逐列標準化 (Row-Normalize)」後的特徵合併。
- `pca64` / `svd64`: 使用 PCA 或 TruncatedSVD 進行降維，保留最關鍵的 64 維。
- `raw_stat`: 原始特徵 + 16 維的統計特徵 (均值、標準差、偏度、峰度等) + 將資料切為 8 段的局部統計特徵 (32 維)。
- `raw_stat_fft`: 在 `raw_stat` 的基礎上，加上 26 維的 **FFT 頻域特徵** (振幅、各頻段能量占比等)。

## 3. 多樣化的模型庫 (`src/models/`)
模型分為兩大類，分別針對不同的特徵表現來捕捉資料規律：
- **Tabular 模型 (`tabular.py`)**：
    - 線性模型：`logreg` (Logistic Regression), `svm`。
    - 樹狀模型 (Boosting & Bagging)：`rf` (Random Forest), `lgbm` (LightGBM), `xgb` (XGBoost), `catboost`。
    - 這類模型支援 Scikit-learn API 並且內建 Early Stopping 機制。
- **Deep Learning 模型**：
    - `mlp.py`: 基礎的多層感知機 (Multi-Layer Perceptron)。
    - `cnn1d.py`: 1D ResNet 殘差神經網路，適合捕捉連續訊號的局部特徵。
    - `transformer.py`: 1D SignalTransformer，基於 Patch Embedding 與 Multi-Head Attention 來捕捉全域的序列關係。

## 4. 交叉驗證訓練迴圈 (`src/train.py`, `run_pipeline.py`)
- **訓練流程 (`run_cv`)**：自動執行 5-Fold 交叉驗證，根據模型與特徵的組合進行訓練 (例如 `lgbm + raw_stat_fft`, `transformer + raw`)。
- **深度學習訓練細節**：
    - 使用 `AdamW` Optimizer 與 `CosineAnnealingLR` 排程器。
    - 採用 `CrossEntropyLoss(label_smoothing=0.05)` 來增加模型泛化能力。
    - 實作了 **Mixup Data Augmentation** (機率 50%)，將兩筆樣本按比例混合，有效提升模型的強健性。
- **Artifact 產出**：每個 Fold 訓練完會儲存最佳模型，最終會產出 Out-Of-Fold (OOF) 的預測機率矩陣與 Test 的預測機率矩陣存為 `.npy` 供 Ensemble 使用。

## 5. 模型融合 Ensemble (`src/ensemble.py`)
結合前面所有基底模型（Base Models）的預測結果，透過兩種策略最大化預測效能 (Macro F1 Score)：
- **[Ensemble A] Weighted Blending**：
    - 使用**幾何平均**來融合預測機率。
    - 利用 `scipy.optimize.minimize (Nelder-Mead)` 演算法，以 OOF F1 分數為目標函數，**自動搜尋出各模型最佳的權重比例**。
- **[Ensemble B] Stacking**：
    - 將所有模型 OOF 的預測機率當成「第二層特徵 (Level-2 Features)」。
    - 再次使用 5-Fold CV，訓練一個 Meta-Model (`lgbm` 或 `logreg`) 來做出最終的決策。這種方法能學習到不同模型在不同類別上的強弱項。

## 6. 產出最終結果 (`src/make_submission.py`)
- 讀取 Ensemble 輸出的 `.npy` 檔，並驗證長度、檢查是否有 NaN 或 Inf。
- 利用 Pandas 合併 `sample_submission.csv` 的格式。
- 儲存成最終的 submission `.csv`，並自動印出各類別預測數量的分布狀況以供除錯參考。

---
### 總結流程圖 (Pipeline Flow)

```text
1. 原始資料 (train/test.csv)
      │
2. FeatureBuilder 萃取特徵 (raw, pca, fft...) 
      │
      ├──> Tabular Models (LGBM, XGB, CatBoost, RF, LogReg)
      └──> Deep Learning Models (CNN1D, MLP, Transformer) ──> Mixup + Label Smoothing
      │
3. 5-Fold Cross Validation (產出 oof_preds.npy, test_preds.npy)
      │
4. Ensemble (融合各模型預測結果)
      ├──> Nelder-Mead Weighted Blending
      └──> Meta-Learner Stacking (LGBM)
      │
5. Make Submission (輸出 sub_A_blend.csv / sub_B_stack.csv)
```