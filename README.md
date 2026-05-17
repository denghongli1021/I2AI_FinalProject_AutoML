此檔案提供 Claude Code 在此專案中的操作指引。

## 安裝套件
執行 requirements.txt

## 常用指令

### Pipeline（自製系統）

```cmd
# 批次評估：前 5 個 OpenML + 前 5 個 UCR 資料集（80/20 split）
python run_pipeline.py --batch --top-n 5

# 快速模式（縮減 HPO/NAS 次數）
python run_pipeline.py --batch --fast --top-n 5

# 設定時間上限（秒，0=無限制）
python run_pipeline.py --batch --time-limit 3600

# 跳過深度學習模型（僅跑傳統 ML）
python run_pipeline.py --batch --skip-dl --top-n 3

# 跳過 NAS（表格模式用預設 MLP，時序模式用預設 TSNet）
python run_pipeline.py --batch --no-nas

# 指定評估指標（預設 f1）
python run_pipeline.py --batch --metric accuracy

# 單一 CSV 評估（表格資料）
python run_pipeline.py --csv openml_cc18_data/37_diabetes.csv

# 單一 CSV 評估（時序資料，啟用 TS 模式）
python run_pipeline.py --csv "ucr_ts_80(時序資料)/dataset.csv" --ts

# 單一 CSV 評估（指定目標欄）
python run_pipeline.py --csv data.csv --target label --ts

```

### 對照組 AutoGluon Baseline

```cmd
# 批次模式
python run_baseline.py --batch --top-n 5

# 單一資料集
python run_baseline.py --csv openml_cc18_data/22_mfeat-zernike.csv

# 指定時間上限與 presets
python run_baseline.py --batch --top-n 5 --time-budget 120 --presets medium_quality
```

### 資料集下載

```cmd
# 下載 OpenML-CC18 表格資料集 → openml_cc18_data/
python data_collect.py

# 下載 UCR 時序資料集 → ucr_ts_80(時序資料)/
python data_collect_time.py
```

---

## 架構概覽

整個 Pipeline 分為兩個執行層：

- **`run_pipeline.py`**：資料載入、80/20 split、特徵前處理、呼叫 pipeline 引擎、結果輸出
- **`pipeline.py`**：Pipeline 引擎，負責 HPO → NAS → CV → Ensemble 全流程

### Pipeline 執行流程（`pipeline.py`）

```
原始 CSV
  │
  ▼
[1] 資料前處理（run_pipeline.py / preprocess.py）
      robust_clean_dataframe：datetime 解析、型別修正、常數欄移除
      FeatureBuilder：8 種特徵集（見下方）
      TSFeatureBuilder（時序模式）：lag/rolling/momentum/diff 特徵
  │
  ▼
[2a] Tabular Scout HPO（TabularHPO.scout）
      快速 holdout 評分，篩除弱模型（保留前 2/3 且不低於最佳 7%）
      模型池：lgbm / xgb / catboost / rf / logreg / extra_trees / knn
  │
  ▼
[2b] Tabular Full HPO（TabularHPO.run）
      Optuna TPE，5-Fold CV，依 scout 分數分配 trial 數
      鎖定每模型最佳 feature_set，保留 top_k 組超參數
      黃金預設值保底（src/best_presets.json）
  │
  ▼
[3] MLP NAS（表格模式）/ TSNet NAS（時序模式）
      表格模式：MLPNASSearcher + OneShotSupernet（MLP 共享權重超網路）
      時序模式：TSNASSearcher + TSNetSupernet（4 種算子：conv_k3/conv_k5/tcn_d2/tcn_d4）
      no_nas=True 時：表格用 _DEFAULT_MLP_ARCH，時序用 _DEFAULT_TSNET_ARCH
      小型表格（< 2000 筆）且非時序時自動跳過
  │
  ▼
[4] MLP / TSNet 訓練 HPO
      表格模式：MLPTrainHPO（lr / dropout / weight_decay，StratifiedKFold）
      時序模式：TSNetTrainHPO（Walk-forward CV，feature_set 從 TS_DL_FEATURE_SETS）
  │
  ▼
[5] CNN1D / TCN HPO（DLHPO）
      表格模式：CNN1D 或 ResNet1D_18
      時序模式：TCN（Temporal Convolutional Network，因果擴張卷積）
  │
  ▼
[6] Transformer / PatchTST HPO（DLHPO）
      表格模式：SignalTransformer（CLS token）
      時序模式：PatchTST（patch 嵌入 + mean pooling）
  │
  ▼
[7] 5-Fold CV → OOF + Test 預測（run_cv）
      支援 artifacts/ 快取（.npy），可重複執行加速
      StratifiedKFold（分類）/ RepeatedStratifiedKFold / TimeSeriesSplit（預測）
  │
  ▼
[8] Ensemble A：Nelder-Mead Weighted Blending（NelderMeadBlender）
      幾何平均 + log-space softmax 權重最佳化
  │
  ▼
[9] Ensemble B：Meta-Learner Stacking（MetaLearnerStacker）
      OOF 拼接 + 可選原始特徵 → LGBM/XGB/LogReg meta-learner
      輸出最終預測（test_blend / test_stack 取較佳者）
```

---

## 目錄結構

```
人工智慧project/
├── pipeline.py             # Pipeline 引擎（HPO/NAS/CV/Ensemble）
├── run_pipeline.py         # 批次 + 競賽執行入口
├── run_baseline.py         # AutoGluon 對照組
├── data_collect.py         # 下載 OpenML-CC18 資料集
├── data_collect_time.py    # 下載 UCR 時序資料集
├── src/
│   ├── config.py           # 全域設定（SEED=42, DEVICE, ARTIFACTS_DIR）
│   ├── preprocess.py       # FeatureBuilder（8 種特徵集）+ TSFeatureBuilder
│   ├── data.py             # get_folds(), get_ts_folds(), TabularDataset
│   ├── metrics.py          # calculate_score(), get_metric_name()
│   ├── hpo.py              # TabularHPO, DLHPO, MLPTrainHPO, TSNetTrainHPO
│   ├── nas.py              # MLPNASSearcher, TSNASSearcher（TSNet + CausalConv1d）
│   ├── train.py            # run_cv, run_tabular_cv, run_dl_cv
│   ├── ensemble.py         # NelderMeadBlender, MetaLearnerStacker
│   ├── make_submission.py  # generate_submission()
│   ├── best_presets.json   # 黃金預設超參數（各模型最佳設定）
│   └── models/
│       ├── mlp.py          # 可配置 MLP（depth/hidden_dim/activations/skip）
│       ├── cnn1d.py        # CNN1D, ResNet1D_18, TCN
│       ├── transformer.py  # SignalTransformer, PatchTST
│       └── tabular.py      # build_tabular_model() 工廠函式
├── openml_cc18_data/       # 72 個 OpenML-CC18 表格資料集（CSV）from data_collect.py 
├── ucr_ts_80(時序資料)/    # UCR 時序資料集（每列為一條序列） from data_collect_time.py
├── artifacts/              # OOF/test 預測快取（.npy，run_cv 自動建立）
├── submissions/            # 最終提交 CSV
└── autogluon_models/       # AutoGluon 模型快取
```

---

## 關鍵設定與慣例

### 目標欄自動偵測順序
`target` → `label` → `class` → `y` → `c` → 最後一欄

### 任務自動判斷邏輯（`run_pipeline.py:_auto_detect_task`）
- `dtype == object / bool` → 分類
- 整數且 `nunique ≤ 50` 且比例 < 30% → 分類
- 否則 → 回歸（批次模式暫時跳過）

### 10 種特徵集（`src/preprocess.py`）

| 特徵集 | 說明 | 適用模型 |
|--------|------|---------|
| `raw` | StandardScaler 後的原始特徵 | 所有模型 |
| `signal` | raw + 列正規化（L2 norm） | MLP, CNN |
| `pca64` | PCA 降至 64 維 | LogReg, SVM |
| `svd64` | TruncatedSVD 降至 64 維 | LogReg, SVM |
| `kpca32` | Kernel PCA（rbf）降至 32 維 | LogReg |
| `raw_stat` | raw + 全域統計（20 項）+ 局部統計（n_seg×4）+ KMeans 距離 | LGBM, XGB |
| `raw_stat_fft` | raw_stat + FFT 頻域特徵（頻帶能量/重心/熵等） | MLP, Transformer |
| `poly2` | raw + Top-N 特徵的 degree-2 交互項（自適應記憶體上限） | LGBM, XGB, RF |
| `ts_tabular` | X_scaled + 一階差分 + Lag（1,2）+ Rolling Mean/Std（w=3,5）+ 全域統計 | TS Tabular 模型 |
| `ts_tabular_fft` | ts_tabular 的基礎上再加 FFT 頻域特徵 | TS Tabular 模型 |

各模型可用特徵集：
- `TABULAR_FEATURE_SETS = ["raw", "raw_stat", "poly2"]`
- `CATBOOST_FEATURE_SETS = ["raw", "raw_stat"]`（CatBoost 不用 poly2，防 bad_alloc）
- `LINEAR_FEATURE_SETS = ["pca64", "svd64", "kpca32"]`
- `MLP_FEATURE_SETS = ["raw", "raw_stat", "raw_stat_fft", "signal"]`
- `CNN_FEATURE_SETS = ["raw", "signal"]`
- `TRANSFORMER_FEATURE_SETS = ["raw", "raw_stat_fft", "signal"]`
- `TS_TABULAR_FEATURE_SETS = ["ts_tabular", "ts_tabular_fft"]`（時序 Tabular 模型）
- `TS_DL_FEATURE_SETS = ["raw", "signal"]`（時序 DL 模型）

### TimeBudget 自適應縮放（`pipeline.py`）
- `time_limit=0` = 無限制
- `budget.scale_trials(n)` 依剩餘時間比例縮減 trial 數
- `budget.should_skip(0.20)` 當剩餘時間 < 20% 時跳過該階段

### get_cfg() 模式（依資料量）
| 資料量 | 模式 |
|--------|------|
| `--fast` | 所有階段最小化（tabular_trials=5, nas_epochs=5） |
| < 500 筆 | 小資料：n_repeats=2, n_seeds=3, kpca/kmeans 開啟 |
| < 50,000 筆 | 標準：tabular_trials=20, meta_trials=15 |
| ≥ 50,000 筆 | 大資料：縮減 trial 數，關閉 kpca/kmeans |

### 時序模式特殊行為
- **切分策略**：`get_folds(is_timeseries=True)` → TimeSeriesSplit Walk-forward；非 TS → StratifiedKFold
- **NAS**：TSNASSearcher 搜尋 4 種算子（conv_k3 / conv_k5 / tcn_d2 / tcn_d4）
  - `--no-nas` 時使用 `_DEFAULT_TSNET_ARCH`（conv_k3 + tcn_d2 + tcn_d4，channels=64）
- **DL 模型**：TSNet（取代 MLP）+ TCN（取代 CNN1D）+ PatchTST（取代 Transformer）
- **Tabular 特徵集**：`ts_tabular` / `ts_tabular_fft`（per-row 時序視窗特徵）
- **DL 特徵集**：`raw` / `signal`（lag/rolling 已內嵌於特徵集）
- **Mixup 停用**：時序模式下訓練自動停用（防混合引入未來資訊洩漏）
- **make_loader**：`drop_last=False`，確保每一筆資料都被訓練/預測到
- `TSFeatureBuilder`（跨時間步特徵）：lag=(1,2,3,7)、rolling window=(3,5,10,20)、momentum、diff

### artifacts 快取機制
`artifacts/{dataset_name}/{tag}_oof.npy` + `_test.npy`：CV 完成後自動儲存，重複執行直接載入。

