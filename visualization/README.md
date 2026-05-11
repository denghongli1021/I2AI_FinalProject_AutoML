# 可視化模組(visualizer.py) — SHAP 解釋 module

實作 AutoML Pipeline 中的可視化與決策支援層。  
接收訓練完成的模型與測試集，計算 SHAP 值，並輸出三種可解釋性圖表。

---

## 在 Pipeline 中的位置

```
原始資料
   ↓
資料預處理模組
   ↓
特徵工程模組
   ↓
模型訓練（XGBoost / LightGBM / 神經網路）
   ↓
【AutoMLVisualizer】← this 
   ↓
Global / Local / Interaction 圖表（PNG）
```

---

## 介面說明

```python
from visualizer import AutoMLVisualizer

viz = AutoMLVisualizer(
    model=trained_model,       # 任何 sklearn 相容的模型物件
    X_test=X_test,             # pd.DataFrame，前處理後的測試集
    output_dir="results"       # 圖表輸出資料夾
)

viz.generate_all_plots(
    sample_index=0,                  # 局部解釋要分析的樣本編號
    target_feature="feature_name",   # Interaction 圖要分析的目標欄位
    prefix="my_dataset"              # 輸出檔名前綴
)
```

**輸入參數**

| 參數 | 型別 | 說明 |
|------|------|------|
| `model` | sklearn 相容物件 | 訓練完成的模型。樹模型使用 TreeSHAP（精確解）；神經網路自動 fallback 至 `shap.Explainer`（近似解）。 |
| `X_test` | `pd.DataFrame` | 前處理後的測試集。欄位名稱必須具有業務語意（不可為 `x1`、`x2` 等代號）。 |
| `output_dir` | `str` | PNG 輸出資料夾，不存在時自動建立。 |
| `sample_index` | `int` | 局部瀑布圖要解釋的樣本索引。 |
| `target_feature` | `str` | Interaction 依賴散佈圖的目標特徵欄位名稱。 |
| `prefix` | `str` | 輸出檔名前綴，例如 `credit` → `credit_global.png`。 |

**輸出檔案**

| 檔案 | 圖表類型 | 說明 |
|------|---------|------|
| `{prefix}_global.png` | 全局特徵重要性 | 所有測試樣本的 SHAP 絕對值平均，排序長條圖 |
| `{prefix}_waterfall.png` | 局部瀑布圖 | 單一樣本的逐特徵 SHAP 貢獻量拆解 |
| `{prefix}_dependence.png` | 交互依賴散佈圖 | 目標特徵的 SHAP 值對原始特徵值分布，第三變數以顏色編碼 |

---

## 支援的模型類型

| 模型 | SHAP 方法 | 速度 | 精確度 |
|------|----------|------|--------|
| XGBoost | TreeExplainer | 快 | 精確解 |
| LightGBM | TreeExplainer | 快 | 精確解 |
| CatBoost | TreeExplainer | 快 | 精確解 |
| Random Forest | TreeExplainer | 快 | 精確解 |
| 神經網路（TCN、LSTM 等） | shap.Explainer（fallback） | 慢 | 近似解 |

模組在初始化時自動偵測模型類型，切換模型不需要修改任何程式碼。

> **神經網路注意事項**：模型的 `predict` 函數必須能接受 `pd.DataFrame` 或 `np.ndarray` 作為輸入。若模型只接受 PyTorch tensor，需由模型訓練組提供一個 wrapper 函數。

---

## 驗證資料集

以下五組資料集用於驗證模組在不同任務類型與資料來源下的通用性。

| 資料集 | 任務類型 | 資料來源 | 執行腳本 |
|--------|---------|---------|---------|
| 台灣信用卡違約預測 | 分類 | OpenML API（ID: 42477） | `main_test.py` |
| IBM HR 員工離職預測 | 分類 | Kaggle CSV | `test_generality.py` |
| 房價預測（Ames） | 回歸 | Kaggle CSV | `test_generality.py` |
| ETT 電力變壓器油溫 | 時序回歸 | GitHub CSV | `test_timeseries.py` |
| Diabetes 130-US Hospitals | 分類（類別不平衡、醫療場景） | OpenML API（ID: 4541） | `test_diabetes.py` |

---

## 環境安裝

```bash
pip install shap plotly kaleido xgboost pandas numpy
```

---

## 資料集下載

**ETTh1（時序）**
```bash
wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv -O dataset/ETTh1.csv
```

**HR Attrition 與 House Prices（Kaggle，需手動下載）**
- HR：https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- House Prices：https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques
- 下載後放入 `dataset/` 資料夾

**信用卡違約 與 Diabetes 130（OpenML）**  
執行腳本時透過 `fetch_openml()` 自動下載，無需手動處理。

---

## 執行方式

```bash
# 主秀：台灣信用卡違約（OpenML 串接）
python main_test.py

# 通用性測試：HR 離職 + 房價（Kaggle CSV）
python test_generality.py

# 時序測試：ETTh1 電力變壓器油溫
python test_timeseries.py

# 醫療場景測試：Diabetes 130
python test_diabetes.py
```

---

## 資料夾結構

```
visualization/
├── visualizer.py          # 核心模組，AutoMLVisualizer 類別
├── main_test.py           # 信用卡違約 demo
├── test_generality.py     # HR 離職 + 房價通用性測試
├── test_timeseries.py     # ETTh1 時序測試
├── test_diabetes.py       # Diabetes 130 醫療場景測試
└── README.md              # 本文件
```

執行各腳本後會自動產生對應的輸出資料夾：
```
visualization/
├── Final_Credit_Results/
├── test_hr_results/
├── test_house_results/
├── timeseries_results/
└── diabetes_results/
```
