# AutoML Pipeline 測試報告

**測試日期**: 2026-06-04  
**測試引擎**: python-sklearn (Daniel pipeline deps 未安裝)  
**前端**: http://localhost:5500 | **後端**: http://127.0.0.1:8000

---

## 個別 Task 結果

| Task | 資料集 | 任務 | 引擎 | 主要指標 | SHAP | 特徵名 | 結論 |
|------|--------|------|------|---------|------|--------|------|
| 1 | House Prices (1460r×81c) | 非時序回歸 | sklearn | R²=0.805, RMSE=39,223 | ✅ G/W/D | 原始 CSV (LotArea, GrLivArea…) | ✅ PASS |
| 2 | Adult Income (48842r×15c) | 非時序分類 | sklearn | Acc=0.803, F1=0.435, AUC=0.812 | ✅ G/W/D | 原始 CSV (capital-gain, age…) | ✅ PASS |
| 3 | Daily Delhi Climate (1462r×5c) | 時序回歸 | sklearn+timeSeries | R²=0.763, RMSE=2.754 | ✅ G/W/D | 衍生日期名 (date_month, humidity…) | ✅ PASS (workaround) |
| 4 | UCR Coffee (56r×287c) | 時序分類 | sklearn+timeSeries | Acc=1.0, F1=1.0 | ✅ G/W/D | 原始欄位索引 ("215", "74"…) | ✅ PASS |

**總計: 4 / 4 通過**

---

## Task 詳細

### Task 1 — 非時序回歸 (House Prices)
```json
{
  "dataset": "train.csv (1460 rows, 81 cols)",
  "target": "SalePrice",
  "engine": "sklearn",
  "split": "random 80/20",
  "metrics": { "testR2": 0.8049, "testRMSE": 39223.4, "testMAE": 22667.0 },
  "shap": { "global": true, "waterfall": true, "dependence": true },
  "feature_names_sample": ["LotArea", "GrLivArea", "OverallQual", "GarageArea"],
  "submission_csv": "submission_task1_house_prices.csv (1459 rows, no NaN)",
  "status": "PASS"
}
```

### Task 2 — 非時序分類 (Adult Income)
```json
{
  "dataset": "adult.csv (48842 rows, 15 cols)",
  "target": "income (binary: <=50K / >50K)",
  "engine": "sklearn",
  "split": "random 80/20",
  "metrics": { "testAccuracy": 0.8029, "f1": 0.4350, "auc": 0.8118, "precision": 0.6634, "recall": 0.3236 },
  "shap": { "global": true, "waterfall": true, "dependence": true },
  "feature_names_sample": ["capital-gain", "capital-loss", "hours-per-week", "age"],
  "status": "PASS"
}
```

### Task 3 — 時序回歸 (Daily Delhi Climate)
```json
{
  "dataset": "DailyDelhiClimateTrain.csv (1462 rows, 5 cols incl. date)",
  "target": "meantemp",
  "engine": "sklearn",
  "split": "chronological (timeSeries=true in options)",
  "metrics": { "testR2": 0.7631, "testRMSE": 2.7541 },
  "shap": { "global": true, "waterfall": true, "dependence": true },
  "feature_names_sample": ["date_month", "date_dayofyear", "humidity", "wind_speed"],
  "workaround_required": "手動預先生成 date_year/month/dayofweek/dayofyear 欄位",
  "status": "PASS (with workaround)"
}
```

### Task 4 — 時序分類 (UCR Coffee)
```json
{
  "dataset": "CLS_Coffee.csv (56 rows, 287 cols, binary 0/1)",
  "target": "target",
  "engine": "sklearn",
  "split": "chronological (timeSeries=true in options)",
  "metrics": { "testAccuracy": 1.0, "f1": 1.0 },
  "note": "56 筆小資料集，286 特徵，模型完美分類（過擬合風險高）",
  "shap": { "global": true, "waterfall": true, "dependence": true },
  "feature_names_sample": ["215", "74", "156", "42"],
  "status": "PASS"
}
```

---

## 各 Task 共通異常

| # | 問題 | 影響範圍 | 嚴重性 |
|---|------|----------|--------|
| 1 | **Pipeline 引擎不可用** — torch/optuna/xgb/lgb/catboost 未安裝，點選 pipeline 引擎立即失敗 | Task 1~4 全部 | 🔴 阻塞 (pipeline 路徑) |
| 2 | **API 演算法 key 不一致** — 前端傳 `"rf"`，後端 `_ALGO_LABELS` 用 `"random_forest"`，導致 valid_algos 空 → training 報「無有效資料來源」 | 直接 API 呼叫 | 🟡 中 (UI 有自動映射，不影響前端操作) |
| 3 | **時序預處理 bug** — `preprocess_for_training` 中 AutoRouter 偵測 date 衍生欄後，這些欄未被寫入 DataFrame 便被 `numeric_cols_for_decision` 引用，拋 `KeyError: ['date_year',…] not in index` | Task 3 (含 date 欄的資料) | 🔴 阻塞 (時序預處理路徑) |
| 4 | **Batch predict NaN 失敗** — raw-source 訓練的模型不存 preprocessorId，對含 NaN 的 test.csv 直接預測時 Ridge 報錯 | Task 1 批次預測 | 🟡 中 (需使用者自行填補 NaN) |

---

## 整體判定

| 功能 | 狀態 |
|------|------|
| 資料集上傳 & 品質審查 | ✅ 正常 |
| 預處理（非時序） | ✅ 正常 |
| 預處理（時序 / 含 date 欄） | ❌ Bug — 需 workaround |
| sklearn 訓練（回歸 / 分類） | ✅ 正常 |
| Pipeline 訓練 (Daniel) | ❌ 未安裝依賴 |
| 排行榜顯示 | ✅ 正常 |
| SHAP 三圖（Global/Waterfall/Dependence） | ✅ 正常 |
| 特徵名（原始 CSV，非 f0/f1） | ✅ 正常 |
| 批次預測 + submission.csv | ⚠️ 含 NaN 需 workaround |

### 最終判定：**部分功能異常**

- Core sklearn 訓練 → 正常可上線
- SHAP 分析 → 正常可上線  
- **阻塞性問題（上線前必修）**:
  1. Pipeline 引擎依賴安裝（`pip install torch optuna xgboost lightgbm catboost`）
  2. 時序預處理 date 欄 bug（`preprocess_for_training` 中 date feature materialization 順序錯誤）

---

*報告由 Claude 自動化測試生成 | 2026-06-04*
