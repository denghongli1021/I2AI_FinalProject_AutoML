# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```cmd
# 對單一 CSV 執行完整 AutoML（全自動偵測任務與目標欄）
python run_automl.py --csv openml_cc18_data\37_diabetes.csv

# 指定目標欄、任務類型、HPO 次數
python run_automl.py --csv data.csv --target label --task classification --trials 50

# 時序資料，停用 NAS（加快速度），輸出預測結果
python run_automl.py --csv data.csv --ts --no-nas --output result.csv

# 批次驗證（5 表格 + 5 時序資料集，結果存至 automl_platform/results/）
python -m automl_platform.validate

# 競賽提交格式（train.csv / test.csv 放在 test/ 下）
python test/run_submission.py

# 下載 OpenML-CC18 資料集
python code/data_collect.py
```

## Architecture Overview

三階段串接架構，定義在 `automl_platform/pipeline.py` 的 `AutoMLPipeline`：

```
原始 CSV
  │
  ▼
[1] FeatureExtractionPipeline  (feature_extraction.py)
      表格資料分支：MI 特徵選擇 → 多項式交互 → 群組聚合 → Z-score → SHAP 剪枝
      時序資料分支：TS 統計特徵（均值/標準差/分位數/自相關等 15+ 項）→ Z-score
      全程以 PyTorch GPU 加速（DEVICE = cuda / cpu 自動切換）
  │
  ▼
[2] TPEOptimizer  (hpo.py)
      對 XGB / LGB / RF 各自用 Optuna TPE + MedianPruner 跑 n_trials 次 5-fold CV
      每個模型保留前 top_k 組超參數設定
  │
  ▼
[3] StackingEnsemble  (ensemble.py)
      L1：XGB top-3、LGB top-3（OOF 預測）+ NAS MLP（可選）
      L2：LogisticRegression（分類）/ Ridge（回歸）學習 meta-features
```

**NAS**（`nas.py`）：One-Shot Supernet（4 層 128-unit MLP，共享權重），每 batch 隨機抽子架構訓練，再用演化搜尋（隨機初始族群 → 突變 → 截斷選擇）找最佳深度與活化組合。

## Key Conventions

**目標欄自動偵測順序**：`target` → `label` → `class` → `y` → `c` → 最後一欄

**UCR 時序資料集命名規則**：`CLS_` 前綴 = 分類，`REG_` 前綴 = 回歸

**任務自動判斷邏輯**（`run_automl.py:auto_detect_task`）：dtype=object/bool → 分類；整數且 nunique ≤ 50 且比例 < 30% → 分類；否則回歸

**時序模式**（`--ts`）：跳過 MI/Poly/GroupAgg，改跑 `TimeSeriesFeatureExtractor`，同時自動關閉 SHAP 剪枝（`use_shap_pruning=False`）

**資料目錄**：
- `openml_cc18_data/` — 72 個 OpenML-CC18 表格資料集（CSV，目標欄通常為 `target` 或 `c`）
- `ucr_ts_80(時序資料)/` — UCR 時序資料集（每列為一條完整序列）
- `automl_platform/results/` — `validate.py` 的輸出報表
- `test/` — 競賽提交腳本（需自備 `train.csv` / `test.csv`）

## Known Issues / Improvement Notes

詳見 `improve.md`，主要問題：
- 大數據（> 100k 筆）Poly 特徵 `[N, C, C]` tensor 可能 GPU OOM → 需分批計算
- 小數據（< 500 筆）Poly + NAS + Stacking 三重過擬合風險
- NAS OOF 在大數據下極慢 → 建議加採樣上限（MAX 20k 筆）
