# Pipeline v2 執行教學

## 前置準備

將資料放到 `test/` 資料夾：

```
test/
  train.csv   ← 含 target_feature 欄位
  test.csv    ← 含 id 欄位（無標籤）
```

## 基本用法

```cmd
python run_pipeline.py
```

## 參數說明

| 參數 | 說明 |
|------|------|
| `--fast` | 大幅縮減 HPO/NAS 次數，適合快速驗證流程是否正常 |
| `--skip-tabular` | 跳過傳統模型 HPO，直接讀取 `artifacts/` 快取 |
| `--skip-dl` | 跳過所有深度學習模型（MLP / CNN1D / Transformer） |
| `--no-nas` | 跳過 NAS，MLP 改用預設架構（depth=3, hidden=256） |

## 範例

```cmd
# 完整流程（自動依資料量決定 HPO 次數）
python run_pipeline.py

# 快速驗證（5 trials/model，適合測試環境是否正常）
python run_pipeline.py --fast

# 跳過深度學習，只跑 Tabular 模型
python run_pipeline.py --skip-dl

# 跳過 NAS（加快速度），其餘完整跑
python run_pipeline.py --no-nas

# 快速 + 跳過 DL（最快，只有 Tabular HPO）
python run_pipeline.py --fast --skip-dl
```

## Pipeline 流程（共 9 步）

| 步驟 | 內容 |
|------|------|
| 1 | 載入 `test/train.csv` + `test/test.csv`，標籤編碼 |
| 2 | Tabular HPO：LGBM / XGB / CatBoost / RF / LogReg / SVM |
| 3 | MLP NAS：One-Shot Supernet + 演化搜尋最佳架構 |
| 4 | MLP 訓練超參數 HPO |
| 5 | CNN1D HPO（架構 + 訓練參數 + feature_set） |
| 6 | Transformer HPO |
| 7 | 所有 top-k config 執行 5-Fold CV → OOF + Test 預測（存 `artifacts/`） |
| 8 | Ensemble A：Nelder-Mead Weighted Blending → `submissions/sub_A_blend.csv` |
| 9 | Ensemble B：Meta-Learner Stacking → `submissions/sub_B_stack.csv` |

## HPO 次數自動調整（依資料量）

| 資料量 | 規模 | tabular_trials | nas_epochs |
|--------|------|---------------|------------|
| < 500 筆 | 小 | 30 | 10 |
| 500 ~ 50k 筆 | 中 | 50 | 30 |
| ≥ 50k 筆 | 大 | 20 | 10 |
| `--fast` | — | 5 | 5 |

## 輸出說明

- `submissions/sub_A_blend.csv`：Nelder-Mead 加權融合結果
- `submissions/sub_B_stack.csv`：Meta-Learner Stacking 結果
- `artifacts/*.npy`：各模型 OOF / Test 預測快取（重跑時自動略過）
