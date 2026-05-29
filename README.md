# AutoML Pipeline — 人工智慧 Final Project

自製 AutoML 系統，支援表格資料分類與時序資料分類 / 回歸。以 AutoGluon 為對照組，在 OpenML-CC18 與 UCR 時序資料集上進行批次評估。

---

## 安裝

```cmd
pip install -r requirements.txt
```

---

## 常用指令

### Pipeline（自製系統）

```cmd
# 批次評估：前 5 個 OpenML-CC18 分類資料集
python run_pipeline.py --batch --top-n 5

# 批次評估：分類 + 回歸同時跑（--reg-top-n 控制 openml_regression_data 筆數）
python run_pipeline.py --batch --top-n 10 --reg-top-n 10

# 批次評估：後 10 個（--last 旗標）
python run_pipeline.py --batch --top-n 10 --last

# 快速模式（縮減 HPO/NAS 次數）
python run_pipeline.py --batch --fast --top-n 5

# 設定時間上限（秒，0=無限制）
python run_pipeline.py --batch --time-limit 3600

# 跳過深度學習模型（僅跑傳統 ML）
python run_pipeline.py --batch --skip-dl --top-n 3

# 跳過 NAS（表格模式用預設 MLP）
python run_pipeline.py --batch --no-nas

# 指定分類評估指標（預設 f1）
python run_pipeline.py --batch --metric accuracy

# 指定回歸評估指標（預設 rmse）
python run_pipeline.py --batch --reg-metric r2

# 競賽模式（讀取 test/train.csv + test/test.csv，輸出提交 CSV）
python test/run_submission.py

# 單一 CSV 評估（表格資料）
python run_pipeline.py --csv openml_cc18_data/37_diabetes.csv

# 單一 CSV 評估（指定目標欄）
python run_pipeline.py --csv data.csv --target label

# 預切分模式（手動提供 TRAIN / TEST）
python run_pipeline.py --train train.csv --test test.csv

# 訓練完成後自動產生 SHAP 視覺化（需要 shap + plotly + kaleido）
python run_pipeline.py --csv data.csv --target label --viz
python run_pipeline.py --batch --top-n 5 --viz
```

### 時序專用 Pipeline（分類 + 回歸）

```cmd
# 新TS批次：CLS 與 REG 分開控制各取 N 個（推薦用法）
python run_pipeline_time.py --new-ts-batch --cls-top-n 10 --reg-top-n 10

# 新TS批次：快速模式
python run_pipeline_time.py --new-ts-batch --cls-top-n 5 --reg-top-n 5 --fast

# 新TS批次：結果附加至指定 CSV
python run_pipeline_time.py --new-ts-batch --cls-top-n 10 --reg-top-n 10 --result-file result.csv

# 舊批次模式（前 N 個，不分 CLS/REG）
python run_pipeline_time.py --batch --top-n 10

# 批次評估：後 10 個（斷點續跑）
python run_pipeline_time.py --batch --top-n 10 --last

# 指定分類 / 回歸指標
python run_pipeline_time.py --batch --cls-metric accuracy --reg-metric r2

# 單一 CSV（_TRAIN.csv 格式，自動尋找對應 _TEST.csv）
python run_pipeline_time.py --csv "ucr_ts_80_new(時序資料)/REG_VentilatorPressure_TRAIN.csv"
```

### SHAP 視覺化（訓練後單獨執行）

```cmd
# 從現有 artifacts 直接產生 SHAP 圖（tabular 模型，不重新訓練）
python generate_shap.py --dataset test/dataset.csv --target RiskPerformance

# DL 模型（SignalTransformer）
python generate_shap.py --dataset test/dataset.csv --target RiskPerformance --model dl

# 同時產生 tabular + DL 兩種 SHAP 圖
python generate_shap.py --dataset test/dataset.csv --target RiskPerformance --model both
```

每個輸出目錄包含三張 PNG：`*_global.png`（全局特徵重要性）、`*_waterfall.png`（單筆解釋）、`*_dependence.png`（特徵交互分析）。

### 合併批次結果

```cmd
# 合併 pipeline + baseline + pipeline_time 三份結果到 pipeline_batch_results.csv
python merge_final_results.py
```

### 對照組 AutoGluon Baseline

```cmd
# 批次模式（OpenML-CC18分類 + UCR-TS + OpenML回歸）
python run_baseline.py --batch --top-n 5

# 批次：同時指定分類與回歸各取幾個
python run_baseline.py --batch --top-n 10 --reg-top-n 10

# 批次評估：後 10 個
python run_baseline.py --batch --top-n 10 --last

# 新TS批次（ucr_ts_80_new，各取 3 個 CLS + 3 個 REG）
python run_baseline.py --new-ts-batch

# 單一資料集
python run_baseline.py --csv openml_cc18_data/22_mfeat-zernike.csv

# 指定時間上限與 presets
python run_baseline.py --batch --top-n 5 --time-budget 120 --presets medium_quality
```

### 資料集下載

```cmd
# 下載 OpenML-CC18 表格資料集 → openml_cc18_data/
python scripts/data_collect.py

# 下載 OpenML 回歸資料集 → openml_regression_data/
python scripts/data_collect_reg.py

# 下載 UCR 時序資料集 → ucr_ts_80_new(時序資料)/
python scripts/data_collect_time.py
```

---

## 架構概覽

整個系統分為四個執行入口與兩個 Pipeline 引擎：

| 入口腳本 | 引擎 | 任務 |
|---------|------|------|
| `run_pipeline.py` | `pipeline.py` | 表格分類（openml_cc18_data）+ 表格回歸（openml_regression_data） |
| `run_pipeline_time.py` | `pipeline_time.py` | TS 分類（委派 pipeline.py）+ TS 回歸 |
| `run_baseline.py` | AutoGluon | 表格分類與回歸（對照組） |
| `merge_final_results.py` | — | 合併三份批次結果 CSV |

### Pipeline 執行流程（`pipeline.py`）

```
原始 CSV
  │
  ▼
[1] 資料前處理（run_pipeline.py / preprocess.py）
      robust_clean_dataframe：datetime 解析、型別修正、常數欄移除
      FeatureBuilder：10 種特徵集（見下方）
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
      表格模式：SignalTransformer（CLS token + 固定 Sinusoidal PE + norm_first HPO 搜尋）
      時序模式：PatchTST（Patch Embedding + Mean Pooling）
      獨立搜尋空間：大 n_epochs（80–200）、低 lr；HPO 評分 2-fold 平均；AMP 加速
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
├── run_pipeline.py         # 批次 + 單 CSV 執行入口（表格分類 + 表格回歸）
├── run_pipeline_time.py    # 時序專用執行入口（CLS + REG，支援 --new-ts-batch）
├── run_baseline.py         # AutoGluon 對照組（分類 + 回歸）
├── generate_shap.py        # 從現有 artifacts 產生 SHAP 視覺化（不重新訓練）
├── merge_final_results.py  # 合併三份批次結果 CSV → pipeline_batch_results.csv
├── scripts/
│   ├── data_collect.py     # 下載 OpenML-CC18 資料集
│   ├── data_collect_reg.py # 下載 OpenML 回歸資料集
│   └── data_collect_time.py# 下載 UCR 時序資料集
├── src/
│   ├── pipeline.py         # 通用 Pipeline 引擎（分類；HPO/NAS/CV/Ensemble）
│   ├── pipeline_time.py    # 時序專用 Pipeline 引擎（分類 + 回歸）
│   ├── config.py           # 全域設定（SEED=42, DEVICE, ARTIFACTS_DIR）
│   ├── preprocess.py       # FeatureBuilder（10 種特徵集）+ TSFeatureBuilder + robust_clean_dataframe
│   ├── data.py             # get_folds(), get_ts_folds(), TabularDataset
│   ├── metrics.py          # calculate_score(), get_metric_name()
│   ├── hpo.py              # TabularHPO, DLHPO（2-fold）, _transformer_arch/train_space, MLPTrainHPO, TSNetTrainHPO
│   ├── nas.py              # MLPNASSearcher, TSNASSearcher（TSNet + CausalConv1d）
│   ├── train.py            # run_cv, run_tabular_cv, run_dl_cv（AMP Mixed Precision）
│   ├── ensemble.py         # NelderMeadBlender, MetaLearnerStacker
│   ├── make_submission.py  # generate_submission()
│   ├── best_presets.json   # 黃金預設超參數（各模型最佳設定）
│   └── models/
│       ├── mlp.py          # 可配置 MLP（depth/hidden_dim/activations/skip）
│       ├── cnn1d.py        # CNN1D, ResNet1D_18, TCN
│       ├── transformer.py  # SignalTransformer（Sinusoidal PE, norm_first HPO）, PatchTST
│       └── tabular.py      # build_tabular_model() 工廠函式
├── preprocessing/          # 雙軌前處理模組（樹模型軌 + 深度學習軌）
│   ├── interface.py        # preprocess_for_training()：統一呼叫入口，回傳 {"tree":…, "dl":…}
│   ├── data_loader.py      # 資料載入與型別解析
│   ├── core/
│   │   ├── router.py       # 欄位型別路由（數值/類別/文字/時序/影像）
│   │   └── assembler.py    # 兩軌特徵組裝（樹軌：OrdinalEncoder；DL 軌：OHE）
│   ├── processors/         # 各型別處理器（numeric / category / text / time / image / feature_generator）
│   └── utils/              # 記憶體優化（memory_optimizer）、資料健康度（data_health）、自訂轉換器
├── visualization/
│   └── visualizer.py       # AutoMLVisualizer：SHAP 視覺化（TreeExplainer 精確解 / PermutationExplainer 最多 500 筆取樣，輸出高畫質 PNG）
├── test/
│   ├── dataset.csv               # 測試用資料集
│   ├── ground_truth.csv          # 測試標準答案
│   ├── run_submission.py         # 競賽提交（v3 完整 pipeline）
│   ├── run_baseline_submission.py # AutoGluon 競賽提交
│   └── compare_submissions.py    # 比較三份提交 CSV
├── logs/                   # 執行 log（pipeline_dataset_run.log、baseline_dataset_run.log）
├── pipeline_batch_results.csv    # 合併後的完整批次結果（pipeline + baseline + pipeline_time）
├── result.csv              # 時序批次結果（pipeline_time）
├── openml_cc18_data/       # OpenML-CC18 表格分類資料集（CSV）
├── openml_regression_data/ # OpenML 表格回歸資料集（CSV）
├── ucr_ts_80_new(時序資料)/ # UCR 時序資料集（預切分格式：*_TRAIN.csv + *_TEST.csv）
├── artifacts/              # OOF/test 預測快取（.npy，run_cv 自動建立）
│   ├── single/{dataset}/   # 單一 CSV 執行的模型與 SHAP 輸出
│   └── batch/{dataset}/    # 批次執行的模型與 SHAP 輸出
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
- 否則 → 回歸（`run_pipeline.py` 批次模式由 `--reg-top-n` 控制；`run_pipeline_time.py` 支援 REG_* 前綴）

### 時序任務判斷（`run_pipeline_time.py:_auto_detect_task`）
- 檔名以 `REG_` 開頭 → 回歸（走 `pipeline_time.run_regression()`）
- 否則依 dtype + nunique 判斷（與 run_pipeline.py 相同）

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
| `--fast` | tabular_trials=5, nas_epochs=5, dl_trials=3；**transformer_trials 維持 15**（Transformer 需要更多 trial） |
| < 500 筆 | 小資料：n_repeats=2, n_seeds=3, kpca/kmeans 開啟；transformer_trials=12 |
| < 50,000 筆 | 標準：tabular_trials=20, meta_trials=15；transformer_trials=15 |
| ≥ 50,000 筆 | 大資料：縮減 trial 數，關閉 kpca/kmeans；transformer_trials=10 |

### 時序模式特殊行為

**分類（UCR 格式，樣本獨立）**
- **切分策略**：UCR 分類視為樣本獨立 → StratifiedKFold（非 TimeSeriesSplit）
- **DL 模型**：TSNet（取代 MLP）+ TCN（取代 CNN1D）+ PatchTST（取代 Transformer）
- **NAS**：TSNASSearcher 搜尋 4 種算子（conv_k3 / conv_k5 / tcn_d2 / tcn_d4）
  - `--no-nas` 時使用 `_DEFAULT_TSNET_ARCH`（conv_k3 + tcn_d2 + tcn_d4，channels=64）
- **Mixup 停用**：時序模式下訓練自動停用（防混合引入未來資訊洩漏）
- `TSFeatureBuilder`（跨時間步特徵）：lag=(1,2,3,7)、rolling window=(3,5,10,20)、momentum、diff

**回歸（`REG_*` 前綴 CSV，時間序列預測）**
- **入口**：`run_pipeline_time.py` + `pipeline_time.run_regression()`
- **切分策略**：Chronological 順序切分（最後 20% 為測試集）
- **Tabular 模型**：LGBMRegressor / XGBRegressor / CatBoostRegressor / Ridge / RF / ExtraTrees / KNN
- **DL 模型**：TSNet + TCN + PatchTST（MSE loss）；**不做 NAS**（固定 `_DEFAULT_TSNET_ARCH`）
- **CV**：KFold Walk-forward（`get_ts_folds()`）
- **Ensemble**：Nelder-Mead 加權算術平均 + Meta-Learner Stacking（Ridge / LGBMRegressor）
- **評估指標**：RMSE + R²（`--reg-metric rmse/r2/mae`，預設 rmse）

### artifacts 快取機制
`artifacts/{dataset_name}/{tag}_oof.npy` + `_test.npy`：CV 完成後自動儲存，重複執行直接載入。

### 斷點續跑（`run_pipeline_time.py`）
批次模式啟動時自動載入既有 `pipeline_time_batch_results.csv`，跳過已有有效結果的 dataset；錯誤列（`task == "?"`）會重新嘗試。

---

## 已知問題 / 改善備忘

- **大數據 poly2 OOM**：> 100k 筆時 poly 交互項 tensor 可能 GPU OOM → 已加自適應 n_top 上限（50M 元素預算）
- **小數據過擬合**：< 500 筆時 poly2 + NAS + Stacking 三重風險 → n_repeats/n_seeds 已加大
- **CatBoost 高維 timeout**：n_features > 300 時自動設 per-model timeout 防卡住
- **NAS 在小表格資料跳過**：< 2000 筆自動略過 NAS 與 CNN/Transformer HPO
