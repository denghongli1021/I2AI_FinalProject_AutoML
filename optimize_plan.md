# Pipeline 優化計畫書（針對 `test/run_submission.py` 提升 public micro F1）

> 目標：將 `submission.csv` 的 public score（micro F1）從目前水準（自製 v1 = 0.71514，AutoGluon baseline = 0.83005）顯著拉高，最少不再被 AutoGluon 壓制。

## 1. 資料 / 任務特徵

| 項目 | 數值 |
|------|------|
| Train shape | 10,590 × 1,280 (+1 target) |
| Test shape | 2,640 × 1,280 |
| 特徵型別 | 全為 float64（值域 ~155–180，疑為感測器序列） |
| 任務型別 | 15 類分類 |
| 類別分佈 | **完全平衡** (每類 706 筆，6.67%) |
| 缺失值 | 無 |
| 評分指標 | micro F1（平衡多類 → 等同 accuracy） |

**關鍵推論**：類別完全平衡 → `scale_pos_weight`、`class_weight` 等不平衡修正完全不適用；提升「整體正確率」即提升 micro F1。特徵尺度近似一致、彼此高度相關，疑似 sensor / time series flatten 形式。

## 2. 現行 v2 Pipeline 表現診斷（自 `custom_run_v2.log`）

| 階段 | OOF/CV 分數 | 觀察 |
|------|------------|------|
| MI selection | — | 1280 → 1280（mi_k=2000 已自動 cap）= 等同跳過 |
| Poly + GroupAgg | — | 多出 55 個交互特徵，效益不明，疑似雜訊 |
| XGB best CV | **0.8327** | |
| LGB best CV | **0.8331** | |
| RF best CV | **0.8143** | RF 明顯偏弱，抑低 stacking 上限 |
| **NAS (best)** | **0.8837** | 最佳深度 = 1，shallow MLP 是最強單模 |
| L2（LR meta）→ 推估 final | 約 0.83 | LR L2 把 NAS 的 0.88 拉回 tree-level |
| 總訓練時間 | 21,160 s ≈ 5.9h | NAS 跑在 **CPU**（log 顯示），是主要瓶頸 |

### 核心痛點

1. **L2 LR meta-learner 抑低集成上限**：NAS OOF（0.88）遠高於樹模型（0.83），但 LR 會線性混合所有 base 預測，表現不夠的樹模型反而把 NAS 拉低；最終分數約落在 0.83 級，浪費 NAS 的 0.05 領先優勢。
2. **NAS 單一架構、單一 seed**：只取 1 組最佳架構、1 個訓練 seed，集成多樣性極低；明明是最強模型卻只進場一次。
3. **NAS 跑在 CPU**：torch 有 cu121 但實際 device='cpu'（log 第 31 行），fold 訓練變慢 10x+，限制了 epoch / candidate 預算。
4. **RF 是 dead weight**：CV 0.8143 比 XGB/LGB 落後 2 個百分點，加進 stacking 並未帶來互補性，反而拉低 LR meta 的權重分配能力。
5. **Poly 對近一致尺度的特徵幾乎只是雜訊**：1280 維中對前 10 維作兩兩乘積 → 加 45 個高度相關特徵，CV 沒看到提升。
6. **沒有 fold-bagging**：每模型最終都 refit on full data，丟掉 5 個 fold-model 的平均效益。
7. **L1 多樣性不足**：只有 XGB / LGB 兩種樹 + 一個 NAS MLP，缺少 ExtraTrees、KNN 等 cheap diversifiers。

## 3. 改進方案（依 ROI 排序）

### 🔴 P0：把 NAS 的領先優勢實際反映到最終分數

#### 3.1 L2 改為「策略選擇器」（OOF-driven blending selector）

不是單一 LR meta-learner，而是同時計算多種混合策略，**用 OOF micro F1 自動選最強**：

| 候選策略 | 公式 |
|---------|------|
| `lr` | 現行 LogisticRegression（保留 baseline） |
| `argmax_best_single` | 直接挑 OOF 最佳的單一 base 模型（NAS） |
| `weighted_avg` | 各 base proba 依 OOF accuracy 加權平均，權重 = `softmax(acc / T)` |
| `power_weighted` | `proba_i^w_i` 幾何混合，w_i 正比於 `acc_i ** k`（k≈8 強化最佳模型） |
| `top_k_avg` | 只平均 top-k 個 OOF 最強的 base 模型（去掉拖油瓶 RF） |

最終實作：對每個策略計算 OOF micro F1（即直接用 stacking 的 OOF meta-features 做 argmax 並比照 y），挑分數最高者作為 inference 用策略。

> 這是最關鍵改動。如果 NAS OOF 達 0.88，「best single」或「power_weighted」就能讓最終分數逼近 0.88，相較 LR 的 ~0.83 直接 +5 個百分點。

#### 3.2 多 seed NAS 集成（增加 NAS 在 L1 的權重）

讓 NAS 以 **3 個不同 seed** 各跑一次完整 OOF + refit，視為 3 個獨立 base learner：
- `nas_mlp_seed42`、`nas_mlp_seed7`、`nas_mlp_seed123`
- 每個 seed 的 supernet 訓練、演化搜尋路徑都不同，最終架構往往略異 → 集成多樣性

這直接讓「最強模型陣營」在 base learner pool 中佔 3 票（樹陣營只有 4 票：xgb_0/1, lgb_0/1），加權平均時 NAS 簇權重更大。

#### 3.3 NAS GPU 化 + 預算重新分配

確保 `NeuralArchitectureSearcher` 真的跑在 GPU 上：
- `nas.py` 中已有 `self.device = ... ("cuda" if available)`，但需確認執行時 torch 真的看到 GPU
- 若可確認 GPU，把 NAS supernet epochs 從 15→**30**、arch_candidates 10→**20**、evolution_rounds 2→**4**
- 若無 GPU，維持現狀但降低 fold OOF 開銷：把 OOF NAS 的 epochs 進一步降到 10，refit 階段才跑 30 epochs（核心架構搜尋 vs 最終模型）

### 🟠 P1：清理 / 調整現有架構

#### 3.4 拿掉 RF 對最終 ensemble 的貢獻

兩種選擇：
- (A) 仍保留 RF 做 HPO（記錄分數）但 **不進入 L1 base layer**
- (B) HPO 時跳過 RF（省時）

選 (A)：把 `L1BaseLayer.fit_predict` 中 `for model_name in ["xgb", "lgb"]:` 維持現狀（已沒有 rf），確認 HPO 結果中 RF 仍被記錄即可。實際檢查 `ensemble.py:122` —— **本來就沒把 RF 放進 L1**！只 HPO。所以這條已經做了，但 HPO 預算還在浪費。改為：HPO 仍跑 RF，但只跑 15 trials（其他模型維持 40）。

#### 3.5 加入低成本多樣性 base learner

加入兩個固定設定（不需 HPO）的 base learner：
- **ExtraTreesClassifier**（n_estimators=400, max_depth=None, max_features='sqrt'）
- **KNN**（n_neighbors=15, weights='distance'，標準化後輸入）

這兩個各加 1 次 5-fold OOF 訓練，總共增加幾分鐘但能拉開模型多樣性，meta-blend 通常能多 0.3–0.7 個百分點。

#### 3.6 Fold-bagging：保留每折模型用於 test 預測

修改 `L1BaseLayer.predict`：
- 對 XGB / LGB / ExtraTrees / KNN：保留 `n_folds` 個 fold-models，**predict_proba 取平均**
- 對 NAS：仍只用 refit-on-full 的單一模型（fold 級重訓 NAS 太貴）
- 額外：若記憶體允許，每個 fold model 只在預測時逐個載入

預期 +0.1–0.3% 穩定性提升。

#### 3.7 關閉 PolyInteraction（針對此資料集）

`run_submission.py` 中設 `poly_max_cols=0`（需在 `feature_extraction.py` 加守衛：`max_input_cols=0` 時 Poly 階段直接跳過）。對 1280 維近一致尺度的特徵，多 45 個相關交互項只會稀釋 LR meta 的判別力。

### 🟢 P2：HPO / 訓練細節微調

#### 3.8 HPO 搜索空間擴展（XGB / LGB）

樹模型對於 1280 維高維資料常需更深 / 更多樹。擴展：
```python
"n_estimators": (50, 800),         # 原 (50, 300)
"max_depth": (3, 12),               # 原 (3, 8)
"min_child_weight": (1, 30),        # 新增 (XGB)
"min_data_in_leaf": (5, 100),       # 新增 (LGB)
```
與 early stopping（要求 fit 時帶 eval_set）結合可避免大 n_estimators 訓過頭，但 sklearn `cross_val_score` 介面不便傳 eval_set。折衷：仍走 fixed n_estimators，**top_k_hpo 增至 5**（更多組合進 stacking）。

#### 3.9 確保 numeric stability

`pipeline.py` 已有 `np.nan_to_num`，沒問題。

#### 3.10 移除（或讓 mi_k 自動 cap）的冗餘步驟確認

當特徵數已經 ≤ mi_k，MIFeatureSelector 等同於恆等變換，但仍讀取 sklearn 計算 MI 分數一次（10590 × 1280 → 約 20–60s）。可加守衛：`if X.shape[1] <= self.k: skip MI`。

## 4. 實作步驟

| # | 檔案 | 變更摘要 |
|---|------|---------|
| 1 | `automl_platform/ensemble.py` | 新增 `L2EnsembleLayer` 多策略 + OOF 自動選擇器；`L1BaseLayer` 增加多 seed NAS、ExtraTrees、KNN；fold-bagging |
| 2 | `automl_platform/feature_extraction.py` | `poly_max_cols=0` 時跳過 Poly；MI 短路（特徵數 ≤ k 時直接通過） |
| 3 | `automl_platform/hpo.py` | 擴展 XGB / LGB 搜索範圍；RF trials 減半 |
| 4 | `automl_platform/nas.py` | （可選）若 GPU 可用，預設 epochs / candidates 上調 |
| 5 | `test/run_submission.py` | 設 `poly_max_cols=0`、`top_k_hpo=5`、其他超參調整；列印每策略 OOF F1 |

## 5. 預期效益

| 改動 | 預期 micro F1 增量 |
|------|------------------|
| L2 多策略選擇器（讓 NAS 主導） | **+3 ~ +5 %** |
| 多 seed NAS | +0.3 ~ +0.8 % |
| ExtraTrees + KNN diversity | +0.3 ~ +0.6 % |
| Fold-bagging | +0.1 ~ +0.3 % |
| 拿掉 Poly 雜訊 | 0 ~ +0.3 % |
| **合計**（保守估計） | **+4 ~ +7 %**，目標 0.86 – 0.90 |

## 6. 風險與備援

- **多策略選擇器在 OOF 上挑最佳，但 OOF 與 public LB 分佈可能略有偏差**：保留 LR 作為 fallback，並印出所有策略的 OOF F1，方便人工審核。
- **多 seed NAS 仍在 CPU 上跑會大幅延長時間**：若偵測到 device=cpu，自動降為單 seed。
- **KNN 在 1280 維高維下表現不穩**：放在 base layer 而非單獨使用，blend 時權重會被自動調低，不會拖累。
- **新加的 ExtraTrees 預設 n_estimators=400 在 10k 樣本約 30–60s**，可接受。

## 7. 不在本次範圍

- 不加 CatBoost（環境未安裝，避免引入新依賴）
- 不變更 NAS 架構搜尋空間本體
- 不改變 train/test split 與 5-fold 結構
- 不對單一 fold 用 GPU 分批 / 大 batch 重寫 NAS（保留現有 batch_size=64）
