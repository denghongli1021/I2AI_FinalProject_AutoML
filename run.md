# AutoML 執行教學

## 基本用法

```cmd
python run_automl.py --csv 你的資料集.csv
```

## 參數說明

| 參數 | 必填 | 預設值 | 說明 |
|------|------|--------|------|
| `--csv` | 是 | — | CSV 檔案路徑 |
| `--target` | 否 | 自動偵測 | 目標欄位名稱（自動尋找 target/label/class/y/c 或最後一欄） |
| `--task` | 否 | 自動偵測 | 任務類型：`classification` 或 `regression` |
| `--ts` | 否 | 自動偵測 | 加上此旗標強制視為時序資料 |
| `--trials` | 否 | `20` | HPO 超參數搜索次數（越高越準但越慢） |
| `--no-nas` | 否 | 啟用 NAS | 加上此旗標停用神經架構搜索 |
| `--test-size` | 否 | `0.2` | 測試集比例（0~1 之間） |
| `--seed` | 否 | `42` | 隨機種子，確保結果可重現 |
| `--output` | 否 | 不輸出 | 將預測結果存成 CSV 檔案路徑 |

## 範例

```cmd
# 最簡單用法（全自動偵測）
python run_automl.py --csv openml_cc18_data\37_diabetes.csv

# 指定目標欄位與任務類型
python run_automl.py --csv data.csv --target price --task regression

# 時序資料分類，增加搜索次數
python run_automl.py --csv data.csv --task classification --ts --trials 50

# 停用 NAS，加快執行速度，並輸出預測結果
python run_automl.py --csv data.csv --no-nas --output result.csv

# 完整參數範例
python run_automl.py --csv data.csv --target label --task classification --trials 30 --test-size 0.3 --seed 0 --output predictions.csv
```

## 輸出說明

- **分類任務**：顯示 Accuracy 與 F1-macro
- **回歸任務**：顯示 RMSE 與 R²
- `--output` 指定路徑時，會產生含 `y_true` / `y_pred` 欄位的 CSV
