# 可改進項目

## 1. HPO 與 NAS 未串聯

**問題：** NAS 的訓練超參數（`lr=1e-3`、`dropout=0.2`、`weight_decay=1e-4`、`batch_size=64`、`HIDDEN_SIZES`）全部寫死，HPO 只搜尋樹模型（XGB/LGB/RF），兩者完全獨立。

**改法：** 在 HPO 的 `SEARCH_SPACES` 新增 NN 搜尋空間，將最佳參數傳入 `NeuralArchitectureSearcher`：

```python
# hpo.py：新增 NN 搜尋空間
"nn": lambda trial: {
    "lr":           trial.suggest_float("lr", 1e-4, 1e-2, log=True),
    "dropout":      trial.suggest_float("dropout", 0.1, 0.5),
    "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
    "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128]),
},

# nas.py：改為接收 HPO 傳入的最佳參數
best_nn_params = hpo_results["nn"][0][1]   # Top-1 NN 配置
nas = NeuralArchitectureSearcher(
    dropout=best_nn_params["dropout"],
    # 並在 _train_supernet 內套用 lr / weight_decay / batch_size
)
```

---

## 2. 大/小數據集適應性

### 結論總覽

| 規模 | 整體評估 | 主要風險 |
|------|----------|----------|
| 小數據（< 500 筆） | ⚠️ 有過擬合風險 | 多項式特徵爆炸 + NAS + Stacking 三重過擬合 |
| 中型數據（500~50k） | ✅ 設計吻合 | — |
| 大數據（> 100k） | ⚠️ 效能瓶頸 | GPU OOM + NAS OOF 極慢 |

---

### 小數據問題

#### 2-1. 多項式特徵爆炸（`feature_extraction.py`）

`PolynomialInteractionGenerator(max_input_cols=15)` 會產生 C(15,2)=**105 個交叉項**。  
若訓練集只有 200 筆，5-fold 每折僅 160 筆訓練樣本，特徵數遠大於樣本數 → 嚴重過擬合。

**修正建議：**
```python
# 在 FeatureExtractionPipeline.fit_transform 中加守衛
n_train = len(X)
poly_cols = min(self.poly_max_cols, max(2, int(np.sqrt(n_train / 5))))
self.poly_gen_ = PolynomialInteractionGenerator(max_input_cols=poly_cols)
```

---

#### 2-2. NAS 對小數據過擬合（`ensemble.py`）

`NASWrapper` 在 OOF 階段對每折資料（可能只有幾十筆）訓練神經網路，極易過擬合。

**修正建議：**
```python
# 在 L1BaseLayer.fit_predict 中，小數據自動關閉 NAS
if use_nas and len(X) < 300:
    print("  [L1] NAS disabled: too few samples")
    use_nas = False
```

---

#### 2-3. 5-fold CV 每折樣本過少（`hpo.py`、`ensemble.py`）

100 筆資料做 5-fold → 每折訓練集僅 80 筆，CV 分數方差極大，HPO 容易找到假最優解。

**修正建議：**
```python
# TPEOptimizer.optimize 與 L1BaseLayer.fit_predict 中動態調整 n_folds
n_folds = self.n_folds if len(X) >= 200 else min(self.n_folds, max(3, len(X) // 30))
```

---

### 大數據問題

#### 2-4. 多項式 GPU OOM（`feature_extraction.py`）

```python
outer = arr.unsqueeze(2) * arr.unsqueeze(1)  # [N, C, C]
```
N=200k、C=15 → tensor 大小約 **1.7 GB**，N=1M 時直接 OOM。

**修正建議：分批計算交叉項**
```python
def transform(self, X: pd.DataFrame) -> pd.DataFrame:
    arr = _to_tensor(X[self.input_cols_].fillna(0))
    BATCH = 10_000
    results = []
    for start in range(0, len(arr), BATCH):
        chunk = arr[start:start+BATCH]
        outer = chunk.unsqueeze(2) * chunk.unsqueeze(1)
        results.append(_to_numpy(outer[:, self._i_idx, self._j_idx]))
    return pd.DataFrame(np.vstack(results), columns=self.feature_names_, index=X.index)
```

---

#### 2-5. NAS OOF 大數據極慢（`ensemble.py`）

大數據（50k+）做 5-fold NAS OOF = 5 次神經網路訓練 × 完整訓練集，時間可達數小時。

**修正建議：大數據自動採樣後做 NAS**
```python
# L1BaseLayer._nas_oof 中加採樣上限
MAX_NAS_SAMPLES = 20_000
if len(X) > MAX_NAS_SAMPLES:
    idx = np.random.choice(len(X), MAX_NAS_SAMPLES, replace=False)
    X_nas, y_nas = X[idx], y_enc[idx]
else:
    X_nas, y_nas = X, y_enc
```

---

#### 2-6. HPO CV 在大數據耗時過長（`hpo.py`）

3 個模型 × 20 trials × 5-fold = **300 次訓練**，每次訓練 100k 筆，總計耗時極長。

**修正建議：大數據改用 holdout 替代 CV**
```python
# TPEOptimizer.optimize 中，大數據改用單次 holdout 加速
if len(X) > 50_000:
    from sklearn.model_selection import train_test_split
    X_s, X_v, y_s, y_v = train_test_split(X, y, test_size=0.2, random_state=42)
    # objective 改成 fit(X_s) + score(X_v)
```

---

#### 2-7. GroupAggregation 類別基數過高（`feature_extraction.py`）

`max_unique_ratio=0.05`：100k 筆資料時允許 **5000 個唯一類別**的欄位參與聚合，幾乎等於 high-cardinality 欄位，產生大量雜訊特徵。

**修正建議：**
```python
max_unique = min(50, max(2, int(n * self.max_unique_ratio)))  # 加絕對上限 50
```

---

### 快速修正優先順序

| 優先 | 問題 | 影響 |
|------|------|------|
| 🔴 高 | 多項式 GPU OOM（大數據） | 程式直接崩潰 |
| 🔴 高 | NAS + 小數據過擬合 | 模型效能嚴重下降 |
| 🟡 中 | NAS OOF 大數據極慢 | 等待時間不可接受 |
| 🟡 中 | HPO CV 大數據耗時 | 等待時間過長 |
| 🟢 低 | 5-fold 小數據方差 | CV 結果不穩定 |
| 🟢 低 | GroupAgg 高基數上限 | 雜訊特徵增加 |

---

## 3. 類別不平衡未處理（實驗依據：1049_pc4）

**問題：** pc4 資料集正樣本（有 bug）約佔 10–15%，pipeline accuracy 贏 AutoGluon（0.904 vs 0.894），但 f1_macro 輸（0.740 vs 0.751）。  
典型症狀：accuracy 高但少數類 recall 低，模型偏向預測多數類。  
XGB/LGB 訓練時未傳入任何類別權重參數，AutoGluon 內建 `auto_class_weights` 自動補救。

**修正建議（`hpo.py`）：**
```python
# TPEOptimizer 的 XGB objective 中，分類任務加入 scale_pos_weight
if task == "classification":
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    extra_fit_params["scale_pos_weight"] = neg / max(pos, 1)   # XGBoost

# LightGBM 改用 is_unbalance（多分類時改 class_weight='balanced'）
if task == "classification":
    extra_fit_params["is_unbalance"] = True   # LGBMClassifier
```

**優先順序：** 🔴 高（直接影響 f1_macro，不平衡資料集普遍存在）

---

## 4. 小樣本下 Stacking 層數過深（實驗依據：CLS_ACSF1，n=160）

**問題：** n_train=160、n_features=1460 的時序資料集，AutoGluon 勝出（f1: 0.825 vs 0.769）。  
Pipeline 執行了完整 L1（XGB top-3 + LGB top-3 OOF）→ L2（LR meta-learner）流程：  
5-fold on 160 筆 = 每折只有 128 筆訓練，六個 L1 模型各自過擬合後，L2 再對過擬合的 OOF 預測做擬合，誤差層層累積。

**修正建議（`ensemble.py`）：**
```python
# StackingEnsemble.fit 中，小數據縮減 L1 模型數量
if len(X) < 300:
    top_k = min(self.top_k_hpo, 2)   # 每模型最多取 top-2，而非 top-3
    # 同時強制關閉 NAS（已在 2-2 提及）
```

**優先順序：** 🔴 高（小樣本 + 深 stacking 是 f1 劣勢的核心原因）

---

## 5. HPO trials 分配效率低（資料量無關的系統性問題）

**問題：** `N_HPO_TRIALS=20` 由三個模型（XGB / LGB / RF）共享，等於每模型僅 ~6–7 次 trial。  
XGBoost 的核心超參數（`max_depth × learning_rate × subsample × colsample_bytree`）構成 4 維連續空間，6 次 trial 遠不足以找到可靠的最優點。

**修正建議（`hpo.py`、`run_comparison.py`）：**
```python
# 方案 A：讓每個模型獨立跑 n_trials 次（總量改為 per-model）
for model_name, space_fn in SEARCH_SPACES.items():
    study = optuna.create_study(...)
    study.optimize(objective, n_trials=n_trials)   # n_trials 已是 per-model

# 方案 B：依資料量動態調整 trials
n_trials = max(10, min(50, 200 // max(1, n_features // 10)))
```

> 確認 `hpo.py` 的 `n_trials` 是 per-model 還是 total，若是 total 則直接改為 per-model 即可解決此問題。

**優先順序：** 🟡 中（影響所有資料集的 HPO 品質）

---

### 快速修正優先順序（更新版）

| 優先 | 問題 | 影響 |
|------|------|------|
| 🔴 高 | 多項式 GPU OOM（大數據） | 程式直接崩潰 |
| 🔴 高 | NAS + 小數據過擬合 | 模型效能嚴重下降 |
| 🔴 高 | 類別不平衡未處理 | f1_macro 系統性偏低 |
| 🔴 高 | 小樣本 Stacking 層數過深 | 小資料集效能劣於 baseline |
| 🟡 中 | HPO trials 分配效率低 | 所有資料集 HPO 品質不足 |
| 🟡 中 | NAS OOF 大數據極慢 | 等待時間不可接受 |
| 🟡 中 | HPO CV 大數據耗時 | 等待時間過長 |
| 🟢 低 | 5-fold 小數據方差 | CV 結果不穩定 |
| 🟢 低 | GroupAgg 高基數上限 | 雜訊特徵增加 |
