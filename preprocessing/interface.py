# preprocessing/interface.py
"""
對外 API 窗口（Facade）。

這是整個預處理模組唯一對外暴露的介面：
  - run_data_audit()           → 給前端 UI 使用，回傳資料健康報告
  - preprocess_for_training()  → 給模型訓練組使用，回傳乾淨的訓練/測試集
  - preprocess_for_inference() → 給推論/預測使用，套用已學好的轉換規則

設計模式（方案 A：中控樞紐模式）：
  interface.py 是總指揮，協調以下三個獨立模組：
    data_health.py → 診斷報告（供 UI 顯示）
    router.py      → 欄位類型分類（決定如何處理）
    assembler.py   → 管線組裝（實際執行轉換）
  三個模組互不直接 import，職責清晰，可以獨立測試。
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Tuple, Dict, Any, Optional, Union, List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 保留原版 import 方式，依賴 core/__init__.py 的 export 設定
from .core import AutoRouter, PipelineAssembler
from .utils.data_health import (
    generate_health_report,
    export_report_json,
    export_report_html,
    print_health_report
)
from .processors.feature_generator import MIFeatureSelector
from .data_loader import load_and_merge_data
from .processors.feature_generator import RobustDataCleaner

# 💡 新增功能: 對抗驗證
try:
    from .utils.adversarial import (
        run_adversarial_validation, 
        drop_adversarial_features, 
        print_adversarial_report
    )
except ImportError:
    pass # 如果沒有這個檔案，後續用 try-except 接住

import warnings

# 忽略預期中的 Imputer 幽靈警告，保持 Log 乾淨
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="sklearn.impute",
    message=".*Skipping features without any observed values.*"
)

# ──────────────────────────────────────────────────────────────────
# 私有輔助函式（原版沒有，新增用來處理 inf 值與稀疏矩陣）
# ──────────────────────────────────────────────────────────────────

def _clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    進場前清理（在進入 sklearn 管線之前完成）。
    ⚙️ 記憶體優化版：
    移除 cleaned = df.copy() 這種完全複製行為（記憶體翻倍的元兇）。
    改用覆蓋賦值（Reassignment），利用 Pandas 內部的區塊共享機制（Block Sharing），
    既能省下 90% 的複製記憶體，又能 100% 避免修改到外部原始資料的副作用！
    """
    # ❌ 移除這行：cleaned = df.copy()

    # ── 1. inf → NaN ──────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # 用原本的 df 來計算統計量，不佔額外空間
    n_inf = np.isinf(df[numeric_cols]).sum().sum()

    if n_inf > 0:
        # ✅ 安全無副作用：不開 inplace=True。
        # 當執行賦值給局部變數 df 時，Python 會自動打破與外部大表的直接引用綁定
        df = df.replace([np.inf, -np.inf], np.nan)
        print(f"   [前處理] 替換了 {n_inf:,} 個 inf / -inf 值為 NaN")

    # ── 1.5. 字串型缺失標記 → NaN ──────────────────────────────────
    # 許多真實資料集（如 adult/census）用 '?'、'NA'、'none' 等字串標記缺失值，
    # 若不替換則 OHE 會為這些「假類別」建立獨立欄位，對模型毫無幫助。
    _MISSING_TOKENS = frozenset({'?', 'na', 'n/a', 'none', 'null', 'nan',
                                  'missing', 'unknown', '-', '--'})
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if obj_cols:
        replace_map = {}
        n_replaced_total = 0
        for col in obj_cols:
            unique_vals = df[col].dropna().unique()
            bad_vals = [v for v in unique_vals
                        if str(v).strip().lower() in _MISSING_TOKENS]
            if bad_vals:
                replace_map[col] = {v: np.nan for v in bad_vals}
                n_replaced_total += int(df[col].isin(bad_vals).sum())
        if replace_map:
            df = df.replace(replace_map)
            print(f"   [前處理] 字串型缺失標記替換：{n_replaced_total:,} 個值 → NaN"
                  f"（欄位：{list(replace_map.keys())}）")

    # 🚀🚀🚀 新增這段：1.8 消毒 Pandas 專屬缺失值 (pd.NA -> np.nan) 🚀🚀🚀
    # 解決 reduce_mem_usage 造成的 pd.NA 與 sklearn 衝突問題
    sanitized_count = 0
    for col in df.columns:
        # 如果是 Pandas 擴充型態 (Int8, UInt16, boolean...) 且裡面有缺失值
        if str(df[col].dtype).startswith(('Int', 'UInt', 'boolean')):
            if df[col].isna().any():
                df[col] = df[col].astype(float) # 強制轉為 float，pd.NA 就會乖乖變回 np.nan
                sanitized_count += 1
    if sanitized_count > 0:
        print(f"   [前處理] 已將 {sanitized_count} 個 Pandas 特規壓縮欄位轉回 Float，以相容 Scikit-learn (消滅 pd.NA)")
    # 🚀🚀🚀 新增結束 🚀🚀🚀

    # ── 2. 刪除完全重複列 ──────────────────────────────────────────
    n_before = len(df)
    
    # ✅ 安全無副作用：drop_duplicates() 會回傳一個只包含唯一列的新 View/DataFrame
    # 記憶體開銷遠小於一開始就複製整張大表
    df = df.drop_duplicates()
    
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        df = df.reset_index(drop=True)
        print(f"   [前處理] 刪除了 {n_dropped:,} 筆完全重複列")

    # ── 3. 👻 刪除幽靈欄位 (100% 缺失值) ───────────────────────────
    ghost_cols = df.columns[df.isnull().all()].tolist()
    if ghost_cols:
        print(f"   [前處理] 警告：偵測到 {len(ghost_cols)} 個欄位缺失率高達 100%！")
        print(f"   [前處理] 已自動刪除無效欄位 (範例: {ghost_cols[:5]}...)")
        # ✅ 安全無副作用：直接回傳排除幽靈欄位後的矩陣
        df = df.drop(columns=ghost_cols)

    return df


def _array_to_dataframe(
    array: Any, feature_names: np.ndarray
) -> pd.DataFrame:
    """
    將 ColumnTransformer 的輸出安全轉成 DataFrame。

    為什麼不直接用 pd.DataFrame(array, columns=...)?
    當管線包含 TF-IDF 時，ColumnTransformer 可能輸出 scipy sparse matrix。
    直接傳 sparse matrix 給 pd.DataFrame 會拋 ValueError。
    必須先用 .toarray() 轉成 dense array。
    """
    if sp.issparse(array):
        array = array.toarray()
    df = pd.DataFrame(array, columns=feature_names)
    # 【inf / float32 溢位防禦】
    # feature engineering (poly2 / raw_stat / FFT 等) 對大值欄位做 X_i × X_j
    # 後可能產生 > float32 上限 (~3.4e38) 的值，astype(np.float32) 直接溢位成 ±inf。
    # 下游 StandardScaler / KNNImputer / 對抗驗證的 classifier 看到 inf 就拋
    # "Input X contains infinity or a value too large for dtype('float32')."
    #
    # 修法（順序很重要）：
    #   ① 先把 inf → NaN（保留「異常值」訊號讓 Imputer 接手，不要降級成 F32_MAX）
    #   ② clip 殘餘的超大有限值到 float32 安全範圍（防 astype 溢位）
    #   ③ astype(np.float32) 此時不會再產生新 inf
    #   ④ 保險再 replace 一次（避免 1.0 / 0.0 之類 ufunc 在 cast 階段又產生 inf）
    F32_MAX = float(np.finfo(np.float32).max)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.clip(lower=-F32_MAX, upper=F32_MAX)
    df = df.astype(np.float32)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# ──────────────────────────────────────────────────────────────────
# 對外 API（保留原版函式簽名，並補強實作）
# ──────────────────────────────────────────────────────────────────

def run_data_audit(
    raw_df: pd.DataFrame,
    target_col: str,
    export_html_path: Optional[str] = None, # 🆕 新增參數
    export_json_path: Optional[str] = None  # 🆕 新增參數
) -> Dict[str, Any]:
    """
    [提供給 UI 組] 產生預處理審查報告 (Preprocessing Audit Report)。

    在正式訓練之前，讓前端顯示資料品質問題，
    告知使用者哪些欄位有缺失、哪些可能是 ID、哪些有洩漏風險等。

    ※ 這個函式只「診斷」，不修改資料，也不觸發任何 sklearn 管線。

    Parameters
    ----------
    raw_df : pd.DataFrame
        待診斷的原始資料（必須包含目標欄位）。
    target_col : str
        預測目標的欄位名稱。
    export_html_path : str, optional
        匯出 HTML 視覺化報告的檔案路徑（例如："audit_report.html"）。若未提供則不產出 HTML。
    export_json_path : str, optional
        匯出 JSON 格式報告的檔案路徑（例如："audit_report.json"）。供自動化腳本讀取，若未提供則不產出。

    Returns
    -------
    dict
        JSON 友善的資料健康診斷報告字典（包含 summary, warnings, info 等鍵值），可直接序列化後傳遞給前端 UI 渲染。
    """
    print(f"\n[健檢中心] 正在針對 {raw_df.shape[0]} 筆資料進行全身健康檢查...")
    report = generate_health_report(raw_df, target_col)

    # 2. 在終端機印出漂亮摘要 (呼叫你剛寫的 print_health_report)
    print_health_report(report)

    # 3. 如果有指定路徑，就匯出成實體檔案
    if export_html_path:
        export_report_html(report, export_html_path)
    if export_json_path:
        export_report_json(report, export_json_path)

    if report["warnings"]:
        print(f"[警告] 發現 {len(report['warnings'])} 個潛在問題！")
    else:
        print("[OK] 資料健康狀況良好！")

    return report


def preprocess_for_training(
    data_source: Union[pd.DataFrame, str, List[str]],  
    target_col: str,
    test_data_source: Optional[Union[pd.DataFrame, str, List[str]]] = None, # 🆕 新增：獨立測試集
    test_size: float = 0.2,
    schema_override: Optional[Dict[str, str]] = None,
    main_file_index: int = 0, 
    enable_adv_val: bool = True # 🆕 新增：對抗驗證開關
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    [提供給 模型組] 執行端對端的預處理管線。
    包含多檔智慧載入、偵測有害資料、處理缺失值以及特徵提取。
    Parameters
    ----------
    data_source : Union[pd.DataFrame, str, List[str]]
        資料來源。可接受：
        1. 已讀取的 DataFrame
        2. 單一 CSV 檔案路徑字串
        3. 多個 CSV 檔案路徑清單 (將自動執行 Join)
    target_col : str
        目標欄位名稱
    test_data_source : Union[pd.DataFrame, str, List[str]], optional
        獨立的測試集資料來源（格式同 data_source）。
        若提供此參數，系統將能執行對抗驗證 (Adversarial Validation) 防護機制。
    test_size : float
        驗證集比例，預設 0.2。
        (僅在未提供 test_data_source 時生效，系統會自動從 data_source 中切分)
    schema_override : dict, optional
        手動覆蓋 Router 的自動判斷。
    main_file_index : int
        當傳入多個檔案時，指定哪一個是主表 (預設 0)。
    enable_adv_val : bool
        是否啟用對抗驗證防護盾，預設 True。
        用以自動偵測並剔除導致 Train/Test 分佈漂移的間諜特徵（僅在提供 test_data_source 時生效）。

    Returns
    -------
    (X_train_clean, X_test_clean, y_train, y_test, fitted_preprocessor)

    """
    print(">>> [Phase 1] 資料載入與整合 (Data Ingestion)")
    raw_df = load_and_merge_data(data_source, main_file_index=main_file_index)

    print(">>> [Phase 2] 特徵預處理管線 (Feature Engineering)")

    if target_col not in raw_df.columns:
        raise ValueError(f"找不到目標欄位 '{target_col}'，現有欄位：{list(raw_df.columns)}")

    clean_df = _clean_raw_data(raw_df)

    # 1. 切割特徵與標籤
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    valid_mask = y.notna()
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  [前處理] 目標欄有 {n_invalid:,} 筆缺失，對應列已刪除")
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    if len(X) == 0:
        raise ValueError("清理後資料集為空（0 列），無法訓練。")

    # 2. 嚴格時間與空間切割
    _is_clf_target = (
        pd.api.types.is_bool_dtype(y) or
        y.dtype == object or
        str(y.dtype) == 'category' or
        (pd.api.types.is_integer_dtype(y) and y.nunique() <= 20)
    )
    stratify = y if (_is_clf_target and y.nunique() >= 2) else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    # 🚀 新增這行：立刻超渡舊的肥大變數，釋放記憶體！
    del raw_df, clean_df, X, y
    import gc; gc.collect()

    # 3. 🛡️ 測試集處理與切割邏輯
    print(">>> [Phase 1] 訓練資料載入與整合 (Train Ingestion)")
    raw_df = load_and_merge_data(data_source, main_file_index=main_file_index)

    if target_col not in raw_df.columns:
        raise ValueError(f"找不到目標欄位 '{target_col}'")

    clean_df = _clean_raw_data(raw_df)

    # 1. 拆分特徵與標籤 (暫存為 X, y)
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    valid_mask = y.notna()
    if not valid_mask.all():
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    # 2. 🛡️ 測試集處理與切割邏輯
    y_test = None
    adv_result = None  # 對抗驗證結果，無測試集時保持 None
    if test_data_source is not None:
        # ── 情況 A：使用者有提供真實獨立 Test 集 ──
        print(">>> [Phase 1.5] 測試資料載入與整合 (Test Ingestion)")
        raw_test_df = load_and_merge_data(test_data_source, main_file_index=main_file_index)
        clean_test_df = _clean_raw_data(raw_test_df)
        
        X_train_raw = X
        y_train = y
        X_test_raw = clean_test_df
        
        # 確保 Test 集如果混到了 target 欄位，要把它拔掉
        if target_col in X_test_raw.columns:
            X_test_raw = X_test_raw.drop(columns=[target_col])
        
        # 🚀 啟動對抗驗證 (Adversarial Validation)
        if enable_adv_val:
            print("\n>>> [Phase 1.8] 啟動對抗驗證 (Adversarial Validation)")
            try:
                adv_result = run_adversarial_validation(
                    train_df=X_train_raw,
                    test_df=X_test_raw,
                    target_col=None,
                    auc_threshold_warn=0.7,
                    auc_threshold_danger=0.85
                )
                print_adversarial_report(adv_result)
                n_before = X_train_raw.shape[1]
                X_train_raw, X_test_raw = drop_adversarial_features(X_train_raw, X_test_raw, adv_result)
                adv_result["n_dropped"] = n_before - X_train_raw.shape[1]
            except Exception as e:
                print(f"⚠️ [預處理模組] 對抗驗證執行失敗，已跳過: {e}")
                adv_result = {"verdict": "error", "message": str(e)}

        # 🧹 情況 A 記憶體清理
        try:
            del raw_df, clean_df, raw_test_df, clean_test_df, X, y
        except NameError:
            pass

    else:
        # ── 情況 B：使用者只給了 Train 集，啟動傳統的 80/20 分割 ──
        print(f">>> [Phase 1.5] 無獨立測試集，自動進行 {test_size*100}% 驗證集隨機切割")
        _is_clf_target2 = (
            pd.api.types.is_bool_dtype(y) or
            y.dtype == object or
            str(y.dtype) == 'category' or
            (pd.api.types.is_integer_dtype(y) and y.nunique() <= 20)
        )
        stratify = y if (_is_clf_target2 and y.nunique() >= 2) else None

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )
        
        # 🧹 情況 B 記憶體清理
        try:
            del raw_df, clean_df, X, y
        except NameError:
            pass

    # 強制回收被刪除的記憶體
    import gc; gc.collect()

    print(f"\n[預處理模組] 正在針對 {len(X_train_raw):,} 筆訓練資料進行分析...")

    # ---------------------------------------------------------
    # 🛡️ 修改核心 1: 讓盾牌先幫 Router 探路 (⚡ 記憶體極限優化版)
    # ---------------------------------------------------------
    print("[預處理模組] 啟動 Phase 0 探路 (僅抽樣 5,000 筆防 OOM)...")
    
    # 💡 我們只抽 5000 筆去給 Router 看，省記憶體
    sample_size = min(5000, len(X_train_raw))
    X_train_sample = X_train_raw.sample(n=sample_size, random_state=42)

    temp_cleaner = RobustDataCleaner()
    # 這裡的運算對記憶體來說輕如鴻毛
    X_sample_phase0_array = temp_cleaner.fit_transform(X_train_sample)
    
    X_sample_phase0_df = pd.DataFrame(
        X_sample_phase0_array, 
        columns=temp_cleaner.get_feature_names_out(),
        index=X_train_sample.index
    )

    # 3. 啟動分類大腦 (Router) 掃描【清洗後的抽樣集】
    router = AutoRouter(
        categorical_threshold=50,
        text_length_threshold=20,
        schema_override=schema_override or {},
    )
    feature_groups = router.fit_predict(X_sample_phase0_df)

    # 🧹 探路完畢，任務達成，立刻銷毀這些臨時的抽樣資料！
    del X_train_sample, X_sample_phase0_array, X_sample_phase0_df
    import gc; gc.collect()

    # 4. 啟動組裝廠 (Assembler)
    print("[預處理模組] 啟動雙軌制管線 (Dual-Track Pipeline) 組裝...")
    assembler = PipelineAssembler(feature_groups)

    # 判斷任務類型 — 浮點數型別強制視為回歸，防止 mutual_info_classif 拋 "Unknown label type: continuous"
    is_classification = (
        pd.api.types.is_bool_dtype(y_train) or
        y_train.dtype == object or
        str(y_train.dtype) == 'category' or
        (pd.api.types.is_integer_dtype(y_train) and y_train.nunique() <= 50)
    )

    # ==========================================
    # 🌳 第一軌：樹狀模型專用 (Tree Track - 生肉)
    # 不補值、不縮放。直接交給 XGBoost 自己挖寶，不需經過 MI 篩選！
    # ==========================================
    tree_preprocessor = Pipeline([
        ('phase0', RobustDataCleaner()),
        ('phase1', assembler.build(track="tree"))
        # 🚨 刪除 phase2_mi_selector！讓生肉原汁原味進入模型。
    ])

    # ==========================================
    # 🧠 第二軌：深度學習/線性模型專用 (DL Track - 熟肉)
    # 精緻補值、標準化，嚴格壓在 300 個特徵防 NAS/DL 梯度爆炸與 OOM
    # ==========================================
    dl_preprocessor = Pipeline([
        ('phase0', RobustDataCleaner()),
        ('phase1', assembler.build(track="dl")),
        # ✅ DL 軌道非常需要 MI 篩選，因為神經網路對無用特徵（雜訊）非常敏感！
        ('phase2_mi_selector', MIFeatureSelector(top_k=300, is_classification=is_classification))
    ])

    

    # 5. 正式擬合 (Fit) 與轉換 (Transform) 訓練集
    # ── 1. 生產線啟動：先產生兩個大型陣列 (Numpy Array) ──
    print("[預處理模組] 正在處理 Tree 軌道資料 (Phase 0 -> Phase 2)...")
    # =========================================================
    # 🛡️ 終極防線：確保 Scikit-Learn 引擎絕對不會吃到 pd.NA 或 Int64
    # =========================================================
    if isinstance(X_train_raw, pd.DataFrame):
        X_train_raw = X_train_raw.copy() # 避免 SettingWithCopyWarning
        for col in X_train_raw.columns:
            # 只要是 Pandas 的擴充型態 (Int64, Float32 等)
            if pd.api.types.is_extension_array_dtype(X_train_raw[col]):
                if pd.api.types.is_numeric_dtype(X_train_raw[col]):
                    X_train_raw[col] = X_train_raw[col].astype(np.float64)
                else:
                    X_train_raw[col] = X_train_raw[col].astype(object)
        # 把所有殘存的 pd.NA 徹底抹殺成 np.nan
        X_train_raw = X_train_raw.replace({pd.NA: np.nan})
    # =========================================================
    X_train_tree_array = tree_preprocessor.fit_transform(X_train_raw, y_train)

    print("[預處理模組] 正在處理 DL 軌道資料 (Phase 0 -> Phase 2)...")
    X_train_dl_array = dl_preprocessor.fit_transform(X_train_raw, y_train)

    # ── 2. 🚀 強制烙印法 2.0：趁陣列剛做完，把原始欄位章蓋上去 ──
    tree_preprocessor.named_steps['phase0'].feature_names_in_ = np.array(X_train_raw.columns)
    dl_preprocessor.named_steps['phase0'].feature_names_in_ = np.array(X_train_raw.columns)

    # ── 3. 產品包裝：提取名稱並轉回 DataFrame ──
    tree_feat_names = [name.split('__')[-1] for name in tree_preprocessor.get_feature_names_out()]
    X_train_tree = _array_to_dataframe(X_train_tree_array, tree_feat_names)

    dl_feat_names = [name.split('__')[-1] for name in dl_preprocessor.get_feature_names_out()]
    X_train_dl = _array_to_dataframe(X_train_dl_array, dl_feat_names)

    # ── 4. 統一丟棄廢料：一次性清除所有過渡變數 ──
    try:
        del X_train_tree_array, X_train_dl_array, X_train_raw
    except NameError:
        pass
    import gc; gc.collect()

    # 6. 僅轉換 (Transform) 測試集
    print("[預處理模組] 正在轉換測試集資料...")
    X_test_tree = _array_to_dataframe(tree_preprocessor.transform(X_test_raw), tree_feat_names)
    X_test_dl = _array_to_dataframe(dl_preprocessor.transform(X_test_raw), dl_feat_names)

    # 7. 打包雙軌資料
    X_train_dict = {"tree": X_train_tree, "dl": X_train_dl}
    X_test_dict = {"tree": X_test_tree, "dl": X_test_dl}
    preprocessors = {"tree": tree_preprocessor, "dl": dl_preprocessor}

    print("[預處理模組] 雙軌處理完成！")
    return X_train_dict, X_test_dict, y_train, y_test, preprocessors, adv_result

def preprocess_for_timeseries(
    df: pd.DataFrame,
    time_col: str = None,
    target_col: str = None,
    n_train: int = None
) -> pd.DataFrame:
    """
    [時序專用入口] 
    提供給 run_pipeline_time.py 呼叫的安全時序前處理 API。
    包含：強制時間排序、安全補值 (ffill)、萃取週期特徵、類別編碼。
    嚴格禁止：全域平均補值、打亂順序 (防 Data Leakage)。
    
    Parameters
    ----------
    df : pd.DataFrame
        準備進行時序預處理的資料表 (建議 Train 與 Test 垂直合併後一起傳入，確保編碼一致)。
    time_col : str, optional
        時間欄位的名稱 (例如 'date', 'timestamp')。
    target_col : str, optional
        預測目標欄位的名稱 (例如 'sales', 'pressure')。
    n_train : int, optional
        訓練段列數。若呼叫端是 concat([train, test]) 後傳入，請帶 len(train)，
        類別編碼便只用 train 段 fit，杜絕 test 分布洩漏回 train。

    Returns
    -------
    clean_df : pd.DataFrame
        全部轉為純數值 (Float32)、無缺失值、按時間排序完畢的乾淨資料表。
    """
    print("\n>>> [Interface] 啟動時序專屬預處理管線...")

    from preprocessing.core.ts_preprocessor import TSDataProcessor

    # 1. 呼叫我們剛剛寫好的「時序戰術指揮官」
    processor = TSDataProcessor(time_col=time_col, target_col=target_col, n_train=n_train)
    
    # 2. 執行端到端 (End-to-End) 的安全處理
    clean_df = processor.process(df)
    
    print(">>> [Interface] 時序預處理完成！準備進入訓練管線。")
    return clean_df

def preprocess_for_inference(
    data_source: Union[pd.DataFrame, str, List[str]],
    fitted_preprocessor: Any,
    training_features: List[str],
    main_file_index: int = 0,
) -> pd.DataFrame:
    """
    [提供給 上線部署/預測組] 執行推論期的預處理管線。
    負責載入測試資料、執行特徵對齊，並進行純轉換 (Transform Only)。
    
    ※ 只能 transform，絕對不能 fit。
    ※ 推論階段絕對不能刪列（每一列都是要預測的客戶資料）。

    Parameters
    ----------
    data_source : Union[pd.DataFrame, str, List[str]]
        測試資料來源 (如 test_transaction.csv 與 test_identity.csv)。
    fitted_preprocessor : Any
        在 preprocess_for_training 中訓練好的管線物件。
    training_features : List[str]
        訓練時「進管線前」的原始特徵清單 (用來對齊)。
    main_file_index : int
        多表合併時的主表索引。

    Returns
    -------
    pd.DataFrame
        可以直接餵給 XGBoost/LightGBM 的乾淨測試特徵矩陣。
    """
    if fitted_preprocessor is None:
        raise ValueError(
            "fitted_preprocessor 是 None。\n"
            "請先執行 preprocess_for_training() 並保存回傳的 preprocessor，\n"
            "再傳入此函式。"
        )

    print(">>> [推論 Phase 1] 測試資料載入與整合")
    # 1. 智慧載入器 (支援多表 Join 與 記憶體壓縮)
    raw_df = load_and_merge_data(data_source, main_file_index=main_file_index)

    print(">>> [推論 Phase 2] 基礎清理 (不刪除任何資料列)")
    # ⚠️ 注意：這裡不能呼叫 _clean_raw_data，因為推論階段絕對不能刪除重複列！
    # 我們只手動替換 inf -> NaN
    clean_df = raw_df.copy()
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(clean_df[numeric_cols]).sum().sum()
    if n_inf > 0:
        clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"  [推論清理] 替換了 {n_inf:,} 個 inf / -inf 值為 NaN")
    
    X_new = clean_df

    print(">>> [推論 Phase 3] 特徵強制對齊")
    # 2. 特徵對齊裝甲 (Feature Alignment)
    missing_cols = set(training_features) - set(X_new.columns)
    extra_cols = set(X_new.columns) - set(training_features)
    
    if missing_cols:
        print(f"  [推論對齊] 警告：測試資料缺少 {len(missing_cols)} 個訓練欄位 (將自動補 NaN)。")
    if extra_cols:
        print(f"  [推論對齊] 提示：測試資料多出 {len(extra_cols)} 個未知欄位 (已自動捨棄)。")
        
    # 一行搞定補齊與捨棄，並確保順序與訓練時完全一致
    X_aligned = X_new.reindex(columns=training_features)

    print(">>> [推論 Phase 4] 執行純轉換 (Transform Only)")
    # =========================================================
    # 🛡️ 終極防線 (推論專用版)：確保 Scikit-Learn 引擎絕對不會吃到 pd.NA
    # =========================================================
    # 註：pd / np 已在模組頂端 import；此處不可再區域 import，
    #     否則會讓 np 在整個函式變成區域變數，導致前面 (Phase 2) 的 np.* 觸發 UnboundLocalError。
    if isinstance(X_aligned, pd.DataFrame):
        X_aligned = X_aligned.copy() # 避免 SettingWithCopyWarning
        for col in X_aligned.columns:
            # 只要是 Pandas 的擴充型態 (Int64, Float32 等)
            if pd.api.types.is_extension_array_dtype(X_aligned[col]):
                if pd.api.types.is_numeric_dtype(X_aligned[col]):
                    X_aligned[col] = X_aligned[col].astype(np.float64)
                else:
                    X_aligned[col] = X_aligned[col].astype(object)
        # 把所有殘存的 pd.NA 徹底抹殺成 np.nan
        X_aligned = X_aligned.replace({pd.NA: np.nan})
    # =========================================================
    # 3. 絕對只能用 transform！
    X_clean_array = fitted_preprocessor.transform(X_aligned)
    
    feature_names = fitted_preprocessor.get_feature_names_out()
    clean_feature_names = [name.split('__')[-1] for name in feature_names]
    X_test_clean = _array_to_dataframe(X_clean_array, clean_feature_names)
    
    print("[推論模組] 測試資料轉換完成，準備預測！")
    return X_test_clean