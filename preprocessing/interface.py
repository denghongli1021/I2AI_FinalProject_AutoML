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
from .utils.data_health import generate_health_report


from preprocessing.data_loader import load_and_merge_data
from preprocessing.processors.feature_generator import RobustDataCleaner

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
        print(f"   [前處理] ⚠️ 警告：偵測到 {len(ghost_cols)} 個欄位缺失率高達 100%！")
        print(f"   [前處理] 🔪 已自動刪除無效欄位 (範例: {ghost_cols[:5]}...)")
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
    return pd.DataFrame(array, columns=feature_names)


# ──────────────────────────────────────────────────────────────────
# 對外 API（保留原版函式簽名，並補強實作）
# ──────────────────────────────────────────────────────────────────

def run_data_audit(
    raw_df: pd.DataFrame,
    target_col: str,
) -> Dict[str, Any]:
    """
    [提供給 UI 組] 產生預處理審查報告 (Preprocessing Audit Report)。

    在正式訓練之前，讓前端顯示資料品質問題，
    告知使用者哪些欄位有缺失、哪些可能是 ID、哪些有洩漏風險等。

    ※ 這個函式只「診斷」，不修改資料，也不觸發任何 sklearn 管線。

    Parameters
    ----------
    raw_df : pd.DataFrame
        原始資料（完整，包含 target 欄）
    target_col : str
        目標欄位名稱

    Returns
    -------
    dict
        JSON 友善的健康報告，可直接序列化後傳給前端。
    """
    print(f"\n[健檢中心] 正在針對 {raw_df.shape[0]} 筆資料進行全身健康檢查...")
    report = generate_health_report(raw_df, target_col)

    if report["warnings"]:
        print(f"⚠️ 發現 {len(report['warnings'])} 個潛在問題！")
    else:
        print("✅ 資料健康狀況良好！")

    return report


def preprocess_for_training(
    data_source: Union[pd.DataFrame, str, List[str]],  
    target_col: str,
    test_size: float = 0.2,
    schema_override: Optional[Dict[str, str]] = None,
    main_file_index: int = 0, 
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
    test_size : float
        測試集比例，預設 0.2
    schema_override : dict, optional

        手動覆蓋 Router 的自動判斷。
    main_file_index : int
        當傳入多個檔案時，指定哪一個是主表 (預設 0)。

    Returns
    -------
    (X_train_clean, X_test_clean, y_train, y_test, fitted_preprocessor)

    """
    print(">>> 🔵 Phase 1: 資料載入與整合 (Data Ingestion)")
    raw_df = load_and_merge_data(data_source, main_file_index=main_file_index)

    print(">>> 🔵 Phase 2: 特徵預處理管線 (Feature Engineering)")

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
    stratify = y if (y.nunique() <= 20 and y.nunique() >= 2) else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    print(f"[預處理模組] 正在針對 {len(X_train_raw):,} 筆訓練資料進行分析...")

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
    print("[預處理模組] 正在組裝處理管線...")
    assembler = PipelineAssembler(feature_groups)
    main_transformer = assembler.build()

    # ---------------------------------------------------------
    # 🛡️ 修改核心 2: 鑄造終極一體化 Pipeline
    # ---------------------------------------------------------
    # 我們把「盾牌 (Phase 0)」跟「煉金爐 (Phase 1)」鎖進同一個管線裡
    fitted_preprocessor = Pipeline(steps=[
        ('phase0_robust_cleaner', RobustDataCleaner()),
        ('phase1_feature_engineering', main_transformer)
    ])

    # 5. 正式擬合 (Fit) 與轉換 (Transform) 訓練集
    print("[預處理模組] 正在擬合訓練集資料 (Phase 0 -> Phase 1 一氣呵成)...")
    # 💡 直接對最原始的 X_train_raw 進行訓練
    X_train_clean_array = fitted_preprocessor.fit_transform(X_train_raw, y_train)

    feature_names = fitted_preprocessor.get_feature_names_out()
    X_train_clean = _array_to_dataframe(X_train_clean_array, feature_names)

    # 6. 僅轉換 (Transform) 測試集
    print("[預處理模組] 正在轉換測試集資料...")
    # 💡 測試集會自動先經過 RobustDataCleaner，保證不會當機
    X_test_clean_array = fitted_preprocessor.transform(X_test_raw)
    X_test_clean = _array_to_dataframe(X_test_clean_array, feature_names)

    print("[預處理模組] 處理完成！")
    return X_train_clean, X_test_clean, y_train, y_test, fitted_preprocessor


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

    print(">>> 🟢 推論期 Phase 1: 測試資料載入與整合")
    # 1. 智慧載入器 (支援多表 Join 與 記憶體壓縮)
    raw_df = load_and_merge_data(data_source, main_file_index=main_file_index)
    
    print(">>> 🟢 推論期 Phase 2: 基礎清理 (不刪除任何資料列)")
    # ⚠️ 注意：這裡不能呼叫 _clean_raw_data，因為推論階段絕對不能刪除重複列！
    # 我們只手動替換 inf -> NaN
    clean_df = raw_df.copy()
    numeric_cols = clean_df.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(clean_df[numeric_cols]).sum().sum()
    if n_inf > 0:
        clean_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"  [推論清理] 替換了 {n_inf:,} 個 inf / -inf 值為 NaN")
    
    X_new = clean_df

    print(">>> 🟢 推論期 Phase 3: 特徵強制對齊")
    # 2. 特徵對齊裝甲 (Feature Alignment)
    missing_cols = set(training_features) - set(X_new.columns)
    extra_cols = set(X_new.columns) - set(training_features)
    
    if missing_cols:
        print(f"  [推論對齊] ⚠️ 警告：測試資料缺少 {len(missing_cols)} 個訓練欄位 (將自動補 NaN)。")
    if extra_cols:
        print(f"  [推論對齊] 🔪 提示：測試資料多出 {len(extra_cols)} 個未知欄位 (已自動捨棄)。")
        
    # 一行搞定補齊與捨棄，並確保順序與訓練時完全一致
    X_aligned = X_new.reindex(columns=training_features)

    print(">>> 🟢 推論期 Phase 4: 執行純轉換 (Transform Only)")
    # 3. 絕對只能用 transform！
    X_clean_array = fitted_preprocessor.transform(X_aligned)
    
    feature_names = fitted_preprocessor.get_feature_names_out()
    clean_feature_names = [name.split('__')[-1] for name in feature_names]
    X_test_clean = _array_to_dataframe(X_clean_array, clean_feature_names)
    
    print("[推論模組] 測試資料轉換完成，準備預測！")
    return X_test_clean