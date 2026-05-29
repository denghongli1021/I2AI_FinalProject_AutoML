"""
執行方式：python adversarial_help.py
對抗驗證模組使用手冊 — 隨時可查閱
"""

B   = "\033[1m"
DIM = "\033[2m"
R   = "\033[0m"
CY  = "\033[96m"
GR  = "\033[92m"
YL  = "\033[93m"
RD  = "\033[91m"
BL  = "\033[94m"
MG  = "\033[95m"
WH  = "\033[97m"

SEP  = f"{CY}{'═'*70}{R}"
SEP2 = f"{DIM}{'─'*70}{R}"
SEP3 = f"{BL}{'─'*70}{R}"

print()
print(SEP)
print(f"{B}{WH}  對抗驗證 (Adversarial Validation) — 使用手冊{R}")
print(f"{DIM}  給所有組員：這份檔案在哪、怎麼用、吃什麼吐什麼{R}")
print(SEP)

print(f"\n{B}{CY}【1】檔案在哪裡？{R}")
print(SEP2)
print(f"""
  {B}主程式{R}
  {GR}preprocessing/utils/adversarial.py{R}

  {B}模組樹狀結構{R}
  {DIM}preprocessing/{R}
  {DIM}├── utils/{R}
  {DIM}│   ├── {R}{GR}adversarial.py{R}      {YL}← 這裡{R}
  {DIM}│   ├── data_health.py{R}
  {DIM}│   └── __init__.py{R}       {DIM}(已 export 三個公開函式){R}
  {DIM}└── interface.py{R}
""")

print(f"{B}{CY}【2】前置需求（先確認有裝）{R}")
print(SEP2)
print(f"""
  {B}必要套件{R}
  {YL}pip install lightgbm{R}      {DIM}← 唯一需要額外裝的，scikit-learn 和 pandas 應該都有了{R}

  {B}確認已安裝{R}
  {BL}python -c "import lightgbm; print('OK', lightgbm.__version__)"{R}
""")

print(f"{B}{CY}【3】三個公開函式（對外 API）{R}")
print(SEP2)

print(f"""
  {B}{GR}① run_adversarial_validation(){R}   {DIM}← 主角，你最常呼叫這個{R}
  {SEP3}
  {B}用途{R}  把 Train 貼標籤 0、Test 貼標籤 1，訓練 LightGBM 分類器，
        用 AUC 量化兩邊的分布差異程度，並找出「肇事特徵」。

  {B}輸入{R}
  ┌─────────────────────┬──────────────────────────────────────────────────────────┐
  │ 參數                │ 說明                                                      │
  ├─────────────────────┼──────────────────────────────────────────────────────────┤
  │ {CY}train_df{R}  {YL}必填{R}      │ 訓練集 DataFrame（含或不含 target 欄皆可）               │
  │ {CY}test_df{R}   {YL}必填{R}      │ 測試集 DataFrame                                        │
  │ {CY}target_col{R} {DIM}選填{R}    │ 目標欄名稱，提供後自動從特徵中排除。預設 None             │
  │ {CY}n_splits{R}   {DIM}選填{R}    │ 交叉驗證折數，預設 {B}5{R}                                  │
  │ {CY}top_n_drop{R} {DIM}選填{R}    │ 建議刪除的特徵數量上限，預設 {B}10{R}                       │
  │ {CY}lgb_params{R} {DIM}選填{R}    │ 自訂 LightGBM 參數 dict，預設使用內建調好的參數          │
  └─────────────────────┴──────────────────────────────────────────────────────────┘

  {B}輸出{R}  {DIM}一個 Python dict，結構如下：{R}
  {BL}{{
    "auc_mean"           : 0.9644,          {DIM}← 各折 AUC 平均值（最重要的數字）{R}
    "auc_std"            : 0.0295,          {DIM}← 各折 AUC 標準差（越小越穩定）{R}
    "auc_per_fold"       : [0.961, 0.989, ...],  {DIM}← 每一折的 AUC{R}
    "verdict"            : "danger",        {DIM}← "ok" | "warning" | "danger"{R}
    "message"            : "AUC=0.96 危險...",   {DIM}← 中文結論，直接給人看{R}
    "n_train"            : 300,
    "n_test"             : 100,
    "n_features"         : 25,
    "feature_importances": [               {DIM}← 依重要性排序，第一名最需要注意{R}
      {{"feature": "feature_a", "importance": 1665.3}},
      {{"feature": "feature_b", "importance": 340.9}},
      ...
    ],
    "features_to_drop"   : ["feature_a", "feature_b", ...],  {DIM}← 直接給你要刪哪些{R}
  }}{R}
""")

print(f"""
  {B}{YL}② print_adversarial_report(){R}   {DIM}← 把 result 印成人看得懂的格式{R}
  {SEP3}
  {B}用途{R}  把 run_adversarial_validation() 的回傳值印成有色彩、有視覺化的報告。
        不修改任何資料，純輸出用。

  {B}輸入{R}
  ┌─────────────────────┬──────────────────────────────────────────────────────────┐
  │ {CY}result{R}    {YL}必填{R}      │ run_adversarial_validation() 的回傳值（那個 dict）       │
  └─────────────────────┴──────────────────────────────────────────────────────────┘

  {B}輸出{R}  直接印到終端機，{B}沒有回傳值{R}。印出內容：
  • AUC 數字 ± 標準差
  • 各折 AUC 明細
  • 視覺化量表 ████░░ (0.5 → 1.0)
  • 中文 verdict 訊息（✅ / ⚠️ / 🚨）
  • Top 特徵重要性長條圖 + 「建議刪除」標記
  • 建議刪除欄位的完整清單
""")

print(f"""
  {B}{MG}③ drop_adversarial_features(){R}   {DIM}← 執行刪除，一行完成{R}
  {SEP3}
  {B}用途{R}  根據 result 裡的 features_to_drop 清單，
        同時從 Train 和 Test 刪除偏差特徵，回傳乾淨的兩個 DataFrame。

  {B}輸入{R}
  ┌─────────────────────┬──────────────────────────────────────────────────────────┐
  │ {CY}train_df{R}  {YL}必填{R}      │ 原始訓練集（會回傳刪除後的副本，不修改原本）             │
  │ {CY}test_df{R}   {YL}必填{R}      │ 原始測試集                                              │
  │ {CY}result{R}    {YL}必填{R}      │ run_adversarial_validation() 的回傳值                   │
  └─────────────────────┴──────────────────────────────────────────────────────────┘

  {B}輸出{R}  {DIM}tuple，兩個 DataFrame：{R}
  {BL}(train_clean, test_clean){R}
  {DIM}已刪除偏差欄位的訓練集和測試集，原本的 train_df / test_df 不受影響。{R}
""")

print(f"{B}{CY}【4】完整使用流程（複製貼上就能跑）{R}")
print(SEP2)
print(f"""
  {DIM}# ── Step 0：import ─────────────────────────────────────────────{R}
  {GR}from preprocessing.utils.adversarial import ({R}
  {GR}    run_adversarial_validation,{R}
  {GR}    print_adversarial_report,{R}
  {GR}    drop_adversarial_features,{R}
  {GR}){R}

  {DIM}# ── Step 1：準備你的 train_df 和 test_df ────────────────────────{R}
  {DIM}# train_df 和 test_df 是 pandas DataFrame，欄位要有共同的 feature 欄{R}
  {DIM}# train_df 可以含有 target 欄（例如 "label"），test_df 通常沒有{R}

  {DIM}# ── Step 2：執行對抗驗證 ────────────────────────────────────────{R}
  {BL}result = run_adversarial_validation({R}
  {BL}    train_df,{R}
  {BL}    test_df,{R}
  {BL}    target_col="label",   {DIM}# 你的 target 欄名稱，沒有可以不填{R}
  {BL}){R}

  {DIM}# ── Step 3：印出報告，看 AUC 和問題特徵 ────────────────────────{R}
  {BL}print_adversarial_report(result){R}

  {DIM}# ── Step 4a：如果 verdict 是 "warning" 或 "danger" → 刪特徵 ───{R}
  {BL}if result["verdict"] != "ok":{R}
  {BL}    train_clean, test_clean = drop_adversarial_features({R}
  {BL}        train_df, test_df, result{R}
  {BL}    ){R}
  {BL}    {DIM}# 之後用 train_clean / test_clean 繼續訓練{R}

  {DIM}# ── Step 4b：如果 verdict 是 "ok" → 直接用原資料就好 ───────────{R}
  {BL}else:{R}
  {BL}    train_clean, test_clean = train_df, test_df{R}
  {BL}    {DIM}# 分布一致，無需任何處理{R}
""")

print(f"{B}{CY}【5】AUC 判讀速查表（最重要，背起來）{R}")
print(SEP2)
print(f"""
  ┌──────────────┬────────────┬─────────────────────────────────────────────────┐
  │ AUC 範圍     │ verdict    │ 意義 & 行動                                      │
  ├──────────────┼────────────┼─────────────────────────────────────────────────┤
  │ {GR}0.50 ~ 0.62{R} │ {GR}"ok"{R}      │ ✅ 完美，Train/Test 分布一致，什麼都不用做     │
  │ {YL}0.62 ~ 0.70{R} │ {YL}"ok"{R}      │ ✅ 輕微差異，可接受，建議記錄觀察             │
  │ {YL}0.70 ~ 0.85{R} │ {YL}"warning"{R} │ ⚠️  明顯差異，檢視並考慮刪除 Top 特徵        │
  │ {RD}0.85 ~ 1.00{R} │ {RD}"danger"{R}  │ 🚨 嚴重不一致，刪特徵，必要時重新審視資料    │
  └──────────────┴────────────┴─────────────────────────────────────────────────┘

  {B}result["verdict"]{R} 已幫你判好了，不用自己比較數字。
  {B}result["features_to_drop"]{R} 已幫你列出建議刪除的欄位清單。
""")

print(f"{B}{CY}【6】注意事項（避免踩雷）{R}")
print(SEP2)
print(f"""
  {YL}①{R} {B}在預處理管線「之前」跑{R}
     對抗驗證要吃「原始 DataFrame」，不是 preprocess_for_training() 出來的結果。
     正確順序：對抗驗證 → 刪偏差特徵 → 再跑前處理管線。

  {YL}②{R} {B}train_df 和 test_df 必須有共同的特徵欄{R}
     程式只看兩邊都有的欄位（取交集），多的欄位自動忽略，不會報錯。

  {YL}③{R} {B}target 欄要記得傳 target_col{R}
     如果 train_df 含有 "label" 欄，一定要傳 target_col="label"，
     不然 LightGBM 會把 target 當成特徵，AUC 數字會無意義地暴增。

  {YL}④{R} {B}不要對 preprocess 後的資料跑{R}
     前處理會把欄位名稱改掉（例如 num__feature_a、cat__city），
     對抗驗證看的是「原始欄位名稱」，跑在處理後的資料上結果難以解釋。

  {YL}⑤{R} {B}features_to_drop 是「建議」，不是命令{R}
     刪之前先看那個特徵有沒有業務意義，再決定要不要刪。
     可以只刪 Top 1~3 個，跑完再重新驗證 AUC 有沒有降。
""")

print(f"{B}{CY}【7】常用 result dict 存取方式{R}")
print(SEP2)
print(f"""
  {BL}result["auc_mean"]{R}              {DIM}← 最重要的數字，AUC 平均值{R}
  {BL}result["verdict"]{R}               {DIM}← 直接判斷："ok" / "warning" / "danger"{R}
  {BL}result["features_to_drop"]{R}      {DIM}← 建議刪除的欄位名稱 list{R}
  {BL}result["feature_importances"][0]{R} {DIM}← 第一名（最能區分 Train/Test 的特徵）{R}
  {BL}result["feature_importances"][0]["feature"]{R}    {DIM}← 欄位名稱{R}
  {BL}result["feature_importances"][0]["importance"]{R} {DIM}← 重要性分數{R}
""")

print(SEP)
print(f"{B}{WH}  檔案路徑  {R}{GR}preprocessing/utils/adversarial.py{R}")
print(f"{B}{WH}  手冊路徑  {R}{GR}adversarial_help.py{R}  {DIM}（就是這支，隨時 python adversarial_help.py）{R}")
print(f"{B}{WH}  簡報路徑  {R}{GR}adversarial_validation_report.pptx{R}")
print(SEP)
print()
