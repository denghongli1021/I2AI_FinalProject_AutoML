# 預處理模組（Preprocessing Module）

> 這份 README 說明我們實作的四個核心檔案——它們在做什麼、為什麼這樣設計、怎麼使用、以及哪些地方需要注意。
> 讀完這份文件，你應該能夠獨立修改任何一個檔案，並且理解為什麼要這樣寫。

---


## 這個模組在做什麼

機器學習模型無法直接消化「原始資料」，需要先把資料清理、轉換成數字，這個過程叫**預處理（Preprocessing）**。

傳統做法需要手動判斷每個欄位的類型、選擇填補方法、決定編碼方式，每換一個資料集就要重寫一次。這個模組的目標是：

> **讓使用者只需要指定「目標欄位叫什麼名字」，其餘全部自動完成。**

```python
# 使用者只需要這樣：
X_train, X_test, y_train, y_test, preprocessor = preprocess_for_training(
    raw_df,
    target_col="purchased"
)
```

---

## 檔案結構

```
preprocessing/
├── __init__.py
├── interface.py            ← 對外唯一入口（其他人只跟這個互動）
│
├── core/
│   ├── __init__.py         ← 匯出 AutoRouter, PipelineAssembler
│   ├── router.py           ← 欄位分類大腦（本文件說明）
│   └── assembler.py        ← 管線組裝廠（本文件說明）
│
├── processors/             ← 其他組員負責，本模組呼叫
│   ├── numeric_processor.py
│   ├── category_processor.py
│   ├── text_processor.py
│   ├── time_processor.py
│   └── image_processor.py  （目前空白，未實作）
│
└── utils/
    ├── __init__.py
    ├── data_health.py      ← 體檢報告生成器（本文件說明）
    └── custom_transformers.py  （目前空白，供未來擴充）
```

**原則：其他組員（UI 組、模型組）只需要 import `interface.py` 的函式，不需要直接碰 router、assembler、data_health。**

---

## 快速開始：三行程式碼

```python
from preprocessing.interface import run_data_audit, preprocess_for_training, preprocess_for_inference

# ① 訓練前先做健康檢查（給 UI 組顯示問題）
report = run_data_audit(raw_df, target_col="purchased")

# ② 正式預處理（給模型組使用）
X_train, X_test, y_train, y_test, preprocessor = preprocess_for_training(
    raw_df, target_col="purchased"
)

# ③ 新資料進來時套用同樣的轉換（推論時使用）
X_new_clean = preprocess_for_inference(new_data_df, preprocessor)
```

---

## 四個核心檔案說明

---

### `interface.py`（總指揮）

**角色：** 整個模組對外的唯一入口，負責協調 `data_health`、`router`、`assembler` 三個子模組。

**為什麼要有這個檔案？**
使用者不需要知道 Router 怎麼分類、Assembler 怎麼組裝，只需要一個簡單的 API。`interface.py` 把複雜性隱藏在裡面，對外只暴露三個函式。

#### 對外三個函式

---

##### `run_data_audit(raw_df, target_col)`

**給誰用：** UI 組（在使用者上傳資料後，顯示資料品質報告）

**重要：這個函式只「診斷」，不修改資料、不執行任何 sklearn 計算。**

```python
report = run_data_audit(raw_df, target_col="purchased")
# report 是一個 dict，可以直接序列化成 JSON 傳給前端
```

| 參數 | 類型 | 說明 |
|------|------|------|
| `raw_df` | `pd.DataFrame` | 原始完整資料（包含 target 欄） |
| `target_col` | `str` | 目標欄名稱，例如 `"purchased"` |

**回傳值：** `dict`，包含以下欄位：

```python
{
    "total_rows": 1000,           # 總列數
    "total_columns": 15,          # 總欄數
    "perfect_columns": 12,        # 無缺失值的欄數
    "missing_summary": {          # 各欄缺失數量與比例
        "age": {"count": 50, "ratio": 0.05}
    },
    "duplicate_rows": 3,          # 重複列數量
    "constant_columns": ["country"],       # 常數欄清單
    "suspected_id_columns": ["user_id"],   # 疑似 ID 欄清單
    "inf_columns": ["income"],             # 含 inf/-inf 的欄清單
    "outlier_summary": [...],              # IQR 法偵測的離群值
    "target_info": {                       # 目標欄分析
        "task_type": "classification",
        "class_distribution": {"0": 0.6, "1": 0.4}
    },
    "leakage_candidates": [...],           # 疑似目標洩漏的欄位
    "warnings": [...],                     # 警告訊息清單
    "info": [...]                          # 一般資訊清單
}
```

---

##### `preprocess_for_training(raw_df, target_col, test_size=0.2, schema_override=None)`

**給誰用：** 模型訓練組

**執行完整 7 步驟預處理流程，自動處理所有欄位類型。**

```python
X_train, X_test, y_train, y_test, preprocessor = preprocess_for_training(
    raw_df,
    target_col="purchased",
    test_size=0.2,
    # schema_override 是選填的，只有在程式判斷錯誤時才需要
    schema_override={"zipcode": "high_cardinality"}
)
```

| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `raw_df` | `pd.DataFrame` | 必填 | 原始資料（含 target 欄） |
| `target_col` | `str` | 必填 | 目標欄名稱 |
| `test_size` | `float` | `0.2` | 測試集比例（0.0 ~ 1.0） |
| `schema_override` | `dict` | `None` | 手動覆蓋欄位分類（見下方說明） |

**回傳值：** 5 個值的 tuple

```python
X_train_clean  # pd.DataFrame  訓練集特徵（已轉換，可直接餵模型）
X_test_clean   # pd.DataFrame  測試集特徵（與訓練集欄位完全一致）
y_train        # pd.Series     訓練標籤（原始，未做任何處理）
y_test         # pd.Series     測試標籤（原始，未做任何處理）
preprocessor   # ColumnTransformer  已學習的預處理管線（保存備用）
```

**⚠️ 務必保存 `preprocessor`！** 推論時需要它。

**`schema_override` 怎麼用：**

```python
# 當程式自動判斷錯誤時，用這個手動糾正
schema_override = {
    "zipcode":       "high_cardinality",  # 郵遞區號不要被誤判為 dropped
    "event_date":    "datetime",           # 確保被當日期處理
    "product_code":  "high_cardinality",  # 高基數類別，防 OHE 爆炸
    "user_uuid":     "dropped",           # 明確指定要刪掉
}
```

合法的類型值：`"numeric"` / `"categorical"` / `"high_cardinality"` / `"text"` / `"datetime"` / `"dropped"`

---

##### `preprocess_for_inference(new_data_df, fitted_preprocessor)`

**給誰用：** 推論/預測組

**⚠️ 核心規則：只能 `transform`，絕對不能 `fit`。**

```python
# fitted_preprocessor 就是 preprocess_for_training 回傳的第五個值
X_new_clean = preprocess_for_inference(new_data_df, preprocessor)

# 接著就可以餵進模型預測
predictions = model.predict(X_new_clean)
```

| 參數 | 類型 | 說明 |
|------|------|------|
| `new_data_df` | `pd.DataFrame` | 新資料（不含 target 欄，欄位名稱需與訓練時一致） |
| `fitted_preprocessor` | `ColumnTransformer` | 由 `preprocess_for_training()` 保存的預處理器 |

**回傳值：** `pd.DataFrame`，欄位名稱與 `X_train_clean` 完全相同，可直接送進模型。

#### `interface.py` 的私有輔助函式（不對外暴露）

| 函式 | 功能 |
|------|------|
| `_clean_raw_data(df)` | 替換 `inf/-inf` 為 NaN、刪除完全重複列 |
| `_array_to_dataframe(array, feature_names)` | 將 ColumnTransformer 輸出（可能是稀疏矩陣）安全轉成 DataFrame |

---

### `data_health.py`（體檢師）

**位置：** `preprocessing/utils/data_health.py`

**角色：** 在資料進入 sklearn 管線之前，掃描並回傳診斷報告。

**重要原則：只診斷，不修改資料。**

#### 主要函式

```python
from preprocessing.utils.data_health import generate_health_report, print_health_report

# 產生報告（回傳 dict）
report = generate_health_report(df, target_col="purchased")

# 在終端機印出易讀格式（開發/debug 用）
print_health_report(report)
```

#### 七個掃描項目

| 項目 | 觸發條件 | 警告等級 |
|------|----------|----------|
| 缺失值掃描 | 任何欄位有缺失 | 🟢 < 30%（資訊）/ 🟡 30–60%（警告）/ 🔴 > 60%（強警告） |
| inf/-inf 掃描 | 數值欄含無限大 | 🔴 警告（sklearn 遇到 inf 直接崩潰） |
| 重複列掃描 | 有完全相同的列 | 🟡 警告（造成評估指標虛高） |
| 常數欄掃描 | 唯一值 ≤ 1 | 🔴 警告（StandardScaler 除以零崩潰） |
| 疑似 ID 欄 | 整數型且唯一率 ≥ 95% | 🟡 警告（讓模型記 ID 而非學規律） |
| IQR 離群值 | 超過 2% 的值是極端異常值 | 🟡 警告（建議改用 RobustScaler） |
| 目標洩漏偵測 | Pearson > 0.8（數值欄）或 Cramér's V > 0.8（類別欄） | 🔴 高風險 / 🟡 中風險 |


---

### `router.py`（分類大腦）

**位置：** `preprocessing/core/router.py`

**角色：** 掃描訓練集的每個欄位，判斷它屬於哪種類型，分配到對應的處理管線。

#### 使用方式

`router.py` 不需要直接呼叫（`interface.py` 會代勞），但如果需要查看分類結果：

```python
from preprocessing.core.router import AutoRouter

router = AutoRouter(
    categorical_threshold=50,   # 唯一值 ≤ 50 → categorical（interface.py 預設值）
    text_length_threshold=20,   # 平均字串長度 > 20 字元 → text
    id_ratio_threshold=0.95,    # 整數欄唯一率 ≥ 95% → dropped
    missing_drop_threshold=0.60, # 缺失率 > 60% → dropped
    schema_override={"zipcode": "high_cardinality"}  # 選填
)

feature_groups = router.fit_predict(X_train, target_col="purchased")

# 查看詳細判斷記錄
router.print_routing_log()
```

#### 六大分類群組

| 群組 | 代表欄位 | 後續處理 | 說明 |
|------|---------|---------|------|
| `numeric` | age, income, score | 中位數填補 → StandardScaler | 數值型（int/float）|
| `categorical` | gender, status | 眾數填補 → OneHotEncoder | 低基數字串（唯一值 ≤ 50）|
| `high_cardinality` | zipcode, city | 眾數填補 → OrdinalEncoder | 高基數字串（唯一值 > 50，字串短）|
| `text` | review, description | TF-IDF → TruncatedSVD | 自由文字（字串長）|
| `datetime` | signup_date, created_at | 提取年/月/日/星期 | 日期時間型 |
| `dropped` | user_id, uuid, 常數欄 | 不進入模型 | 無資訊量或有害欄位 |

#### 判斷順序（不能亂調換）

```
欄位進來
  ↓
① 是 target_col？→ 跳過（不分類）
  ↓
② 有在 schema_override？→ 直接套用（最高優先）
  ↓
③ 缺失率 > 60%？→ dropped
  ↓
④ 唯一值 ≤ 1（常數欄）？→ dropped
  ↓
⑤ 是 datetime64 型別？→ datetime
  ↓
⑥ 是布林型別？→ categorical  ← 必須在 numeric 之前（bool 繼承 int）
  ↓
⑦ 是 CategoricalDtype？→ categorical
  ↓
⑧ 是數值型（int/float/Int64）？→ 整數且唯一率 ≥ 95% → dropped（ID）
                                 → 否則 → numeric
  ↓
⑨ 是字串/object？→ 能解析為日期？→ datetime
                  → 符合 UUID/Email/Hash 格式？→ dropped（字串型 ID）
                  → 唯一值 ≤ categorical_threshold？→ categorical
                  → 平均長度 > text_length_threshold？→ text
                  → 否則 → high_cardinality
```

### `assembler.py`（組裝廠）

**位置：** `preprocessing/core/assembler.py`

**角色：** 接收 Router 的 `feature_groups`，把對應的 sklearn Pipeline 綁定到對應的欄位，組裝成一個可以直接執行的 ColumnTransformer。

#### 使用方式

`assembler.py` 不需要直接呼叫（`interface.py` 會代勞），但如果需要查看組裝結果：

```python
from preprocessing.core.assembler import PipelineAssembler

assembler = PipelineAssembler(feature_groups)
preprocessor = assembler.build()

# 查看組裝摘要
assembler.describe()

# 查看原始 feature_groups
groups = assembler.get_feature_groups()
```

#### 組裝出來的管線結構

```
ColumnTransformer
├── num_pipeline        ← numeric 欄位
│   └── 中位數填補 → StandardScaler
│
├── cat_pipeline        ← categorical 欄位
│   └── 眾數填補 → OneHotEncoder(handle_unknown='ignore')
│
├── high_card_pipeline  ← high_cardinality 欄位
│   └── 眾數填補 → OrdinalEncoder(unknown_value=-1)
│
├── text_{col}          ← 每個 text 欄位各自一條（重要！）
│   └── 展平+NaN填補 → TfidfVectorizer(1000詞) → TruncatedSVD(50維)
│
├── time_pipeline       ← datetime 欄位
│   └── 提取年/月/日/星期/是否週末
│
└── remainder='drop'    ← dropped 欄位，直接忽略
```

#### 兩個重要的設計修正

**修正一：TF-IDF 維度衝突**

`text_processor.py` 原本的管線順序有 bug：

```
SimpleImputer（輸出 2D）→ TfidfVectorizer（需要 1D）→ 崩潰！
```

Assembler 在內部用 `_build_text_column_pipeline()` 繞過這個問題，自己處理 NaN 填補再傳 1D 給 TF-IDF。

**修正二：每個文字欄獨立一條管線**

如果把多個文字欄合併傳給同一個 TF-IDF，sklearn 會傳 2D DataFrame，TF-IDF 報錯。改為每欄獨立一條，最後水平拼接。

#### ColumnTransformer 的兩個關鍵參數

```python
ColumnTransformer(
    remainder="drop",        # dropped 群組的欄位直接忽略，不進模型
    sparse_threshold=0.3,    # TF-IDF 輸出很稀疏時保留 sparse 格式節省記憶體
)
```

---

## 資料流程與防洩漏規則

### `preprocess_for_training()` 的完整 7 步驟

```
① 進場清理
  raw_df → inf 替換為 NaN → 刪除重複列 → clean_df

② 切分 X 和 y
  clean_df → X（特徵）+ y（標籤）

③ Train/Test Split（必須在任何 fit 之前完成）
  X, y → X_train (80%), X_test (20%), y_train, y_test
  分類問題自動使用 Stratified split（保持類別比例）

④ Router 分析（只看 X_train）
  X_train → feature_groups（六大群組）

⑤ Assembler 組裝
  feature_groups → ColumnTransformer（preprocessor）

⑥ fit_transform 訓練集（「學習」+ 轉換）
  preprocessor.fit_transform(X_train) → X_train_clean

⑦ transform 測試集（只轉換，不學習）
  preprocessor.transform(X_test) → X_test_clean
```
**Router 也只分析訓練集：** 欄位的類型判斷（唯一值數量、字串長度等）只基於 X_train，確保測試集資訊不影響任何決策。
 
## 🧩 核心處理器模組 (Core Processors)

### 1. 數值處理器 (Numeric Processor)
* **功能定位**：處理連續型數值資料，如硬體效能計數器 (HPC) 的數值（例如：Cache misses, Branch mispredictions）、CPU 溫度或頻率。
* **採用演算法**：Median Imputation (中位數填補) + RobustScaler (強健標準化)。
* **原理解析**：
  硬體側信號 (Side-channel) 數據常夾帶極端的雜訊或離群值（Outliers）。傳統的 $Z$-score 標準化容易被極端值扭曲，因此我們採用基於四分位距的 RobustScaler：**將每一筆特徵數值減去該特徵的「中位數」，再除以「四分位距（即數據中間 50% 的分佈範圍）」**。相較於使用平均值與標準差，此做法能確保防禦模型在特徵縮放時，對異常的微架構攻擊訊號保持敏銳且不被雜訊干擾。

---

### 2. 類別處理器 (Categorical Processor)
* **功能定位**：處理低基數 (Low-Cardinality) 的離散特徵，例如：行程權限級別 (User/Kernel mode) 或觸發的 Opcode 類別。
* **採用演算法**：One-Hot Encoding (獨熱編碼) 搭配未知類別防護 (`handle_unknown='ignore'`)。
* **原理解析**：
  **為原本特徵中的每一個獨立類別建立一個專屬的「二元指標欄位」**。當資料屬於該類別時，對應的欄位標記為 1，其餘欄位皆標記為 0。這樣能將類別資料展開到多維度獨立空間中，徹底消除模型對類別代碼產生「大小」或「順序」的錯誤推論。

---

### 3. 文字處理器 (Text Processor)
* **功能定位**：處理非結構化字串，例如：反組譯組合語言片段、系統異常日誌或威脅情資。
* **採用演算法**：TF-IDF Vectorization + Truncated SVD (LSA 降維)。
* **原理解析**：
  * **A. TF-IDF 權重計算**：評估一個字詞的重要性。具體作法是將「字詞在單一文件中出現的頻率」乘上「該字詞在所有文件中稀有程度的懲罰權重」。越罕見且在特定文件中頻繁出現的字詞，代表性與權重就越高。
  * **B. Truncated SVD 降維**：透過矩陣分解技術，將 TF-IDF 產生的高維度且極度稀疏的字詞矩陣，壓縮成較低維度的稠密矩陣。這不僅能大幅節省運算資源，還能從中萃取出隱藏的潛在語意特徵，並動態確保降維目標小於樣本數，避免系統崩潰。

---

### 4. 時間處理器 (Temporal Processor)
* **功能定位**：處理時間戳記，捕捉攻擊行為在時間維度上的週期性規律。
* **採用演算法**：時間特徵萃取 (Datetime Extraction) 與週期性編碼。
* **原理解析**：
  捨棄傳統的線性數值表示法，**改以三角函數（正弦與餘弦）將時間點（例如 24 小時制的每個小時）映射到一個二維單位圓的座標點上**。這使得模型能夠完美理解時間的「頭尾相接」特性，例如深夜 23:59 與凌晨 00:01 在數值上雖然落差極大，但在單位圓的特徵空間中是非常靠近的。

---

### 5. 視覺/圖片特徵處理器 (Visual/Image Processor)
* **功能定位**：處理 2D 頻譜圖 (Spectrograms) 或熱力圖。從影像中萃取統計特徵，避免高維度像素導致的記憶體溢出。
* **採用演算法**：Global Statistical Pooling (全域統計池化)。
* **原理解析**：
  若將 RGB 頻譜圖攤平會產生數十萬個維度，因此我們改為在各個色彩通道上計算統計量：
  * **A. 通道平均值 (Mean)**：將影像在各色彩通道的所有像素值計算平均，藉此捕捉頻譜圖的整體能量分佈。
  * **B. 通道標準差 (Standard Deviation)**：計算像素值偏離平均值的程度，用於量化影像的對比度以及紋理的複雜變化（代表側信號波動的劇烈程度）。
  * **C. 感知亮度 (Perceived Luminance)**：依照人類視覺對不同顏色的敏感度（綠色最高、藍色最低），對 R、G、B 通道進行加權總和，濃縮出一個最具代表性的絕對亮度數值。
  
  透過此數學轉換，系統能將龐大的圖片路徑瞬間壓縮成 7 維的高密度數值特徵（3 個平均值 + 3 個標準差 + 1 個亮度），並無縫融合至下游的樹狀模型中。