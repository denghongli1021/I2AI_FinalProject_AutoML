# 🚀 AutoML 資料預處理

- 🧠 **自動特徵分類大腦 (Auto-Router)**：自動偵測並分類 `Numeric` (數值)、`Categorical` (類別)、`Text` (長文本) 與 `Datetime` (時間) 特徵。
- 🏭 **四大專屬加工廠 (Processors)**： (待實作)
  - **Numeric**: 中位數填補 + Z-score 標準化 (`StandardScaler`)
  - **Category**: 眾數填補 + 獨熱編碼 (`OneHotEncoder` 具備未知類別容錯機制)
  - **Text (NLP)**: 空值處理 + `TfidfVectorizer` + `TruncatedSVD` 降維
  - **Time**: 自動提取年、月、日、星期及週末特徵
- 🏥 **資料健檢中心 (Data Health Audit)**：內建資料掃描 API，提供缺失值警告與資料維度報告，方便介接前端 UI 儀表板。

##  檔案架構 

```text
├── preprocessing/
│   ├── __init__.py
│   ├── interface.py           # 對外 API 窗口
│   ├── core/
│   │   ├── __init__.py
│   │   ├── router.py          # 特徵分類大腦
│   │   └── assembler.py       # 管線組裝廠
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── numeric_processor.py
│   │   ├── category_processor.py
│   │   ├── text_processor.py  
│   │   └── time_processor.py  
│   └── utils/
│       ├── __init__.py
│       └── data_health.py     # 健檢中心報告生成器
├── test_run.py                # 測試腳本 (需要放在和proprocessing檔案同一層級)
└── README.md