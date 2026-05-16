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
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import train_test_split

# 保留原版 import 方式，依賴 core/__init__.py 的 export 設定
from .core import AutoRouter, PipelineAssembler
from .utils.data_health import generate_health_report


# ──────────────────────────────────────────────────────────────────
# 私有輔助函式（原版沒有，新增用來處理 inf 值與稀疏矩陣）
# ──────────────────────────────────────────────────────────────────

def _clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    進場前清理（在進入 sklearn 管線之前完成）。

    處理兩件 sklearn Pipeline 無法自動處理的事：
    1. inf / -inf → NaN
       StandardScaler / KNNImputer 碰到 inf 會直接拋 ValueError。
    2. 刪除完全重複列
       重複列若跨越 train/test split，模型等於偷看過測試集答案，評估虛高。

    為什麼不在 Processor 裡做？
       Processor 的 fit / transform 不能刪列（維度必須一致）。
       這兩件事必須在「進管線之前」完成，interface 層最合適。
    """
    cleaned = df.copy()

    # inf → NaN
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(cleaned[numeric_cols]).sum().sum()
    if n_inf > 0:
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"  [前處理] 替換了 {n_inf:,} 個 inf / -inf 值為 NaN")

    # 刪重複列
    n_before = len(cleaned)
    cleaned.drop_duplicates(inplace=True)
    n_dropped = n_before - len(cleaned)
    if n_dropped > 0:
        cleaned.reset_index(drop=True, inplace=True)
        print(f"  [前處理] 刪除了 {n_dropped:,} 筆完全重複列")

    return cleaned


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
    raw_df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    schema_override: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    """
    [提供給 模型組] 執行端對端的預處理管線。
    包含偵測有害資料、處理缺失值以及特徵提取。

    Parameters
    ----------
    raw_df : pd.DataFrame
        原始資料（含 target 欄）
    target_col : str
        目標欄位名稱
    test_size : float
        測試集比例，預設 0.2（與原版一致）
    schema_override : dict, optional
        手動覆蓋 Router 的自動判斷。例：{"zipcode": "high_cardinality"}
        原版沒有此參數，此為新增，預設 None 不影響原有行為。

    Returns
    -------
    (X_train_clean, X_test_clean, y_train, y_test, fitted_preprocessor)
    """
    print("[預處理模組] 開始執行...")

    # 驗證 target_col 存在
    if target_col not in raw_df.columns:
        raise ValueError(
            f"找不到目標欄位 '{target_col}'，現有欄位：{list(raw_df.columns)}"
        )

    # 進場清理：inf → NaN、刪重複列（原版沒有此步驟，新增以防崩潰）
    clean_df = _clean_raw_data(raw_df)

    # 1. 切割特徵 (X) 與標籤 (y)
    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    # 移除 target 本身有缺失值的列（無法訓練）
    valid_mask = y.notna()
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"  [前處理] 目標欄有 {n_invalid:,} 筆缺失，對應列已刪除")
        X = X[valid_mask].reset_index(drop=True)
        y = y[valid_mask].reset_index(drop=True)

    if len(X) == 0:
        raise ValueError("清理後資料集為空（0 列），無法訓練。")

    # 2. 嚴格執行時間與空間的切割，防止資料洩漏
    # 分類問題自動啟用 Stratified split，保持類別比例一致
    stratify = y if (y.nunique() <= 20 and y.nunique() >= 2) else None
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    print(f"[預處理模組] 正在針對 {len(X_train_raw)} 筆訓練資料進行分析...")

    # 3. 啟動分類大腦 (Router) 掃描訓練集
    # 使用我們設定好的參數：超過 50 種算文字，平均字串長度大於 20 算 NLP 文字
    router = AutoRouter(
        categorical_threshold=50,
        text_length_threshold=20,
        schema_override=schema_override or {},
    )
    feature_groups = router.fit_predict(X_train_raw)

    # 4. 啟動組裝廠 (Assembler)
    print("[預處理模組] 正在組裝處理管線...")
    assembler = PipelineAssembler(feature_groups)
    fitted_preprocessor = assembler.build()

    # 5. 正式擬合 (Fit) 與轉換 (Transform) 訓練集
    print("[預處理模組] 正在擬合訓練集資料...")
    # 確保這裡是呼叫 fit_transform，這會同時進行「學習」與「轉換」
    X_train_clean_array = fitted_preprocessor.fit_transform(X_train_raw, y_train)

    # 取得欄位名稱 (ColumnTransformer 的特有方法)
    feature_names = fitted_preprocessor.get_feature_names_out()
    # 使用輔助函式安全轉換（處理 TF-IDF 可能產生的 sparse matrix）
    X_train_clean = _array_to_dataframe(X_train_clean_array, feature_names)

    # 6. 僅轉換 (Transform) 測試集
    print("[預處理模組] 正在轉換測試集資料...")
    # 這裡絕對不能再寫 fit，只能寫 transform
    X_test_clean_array = fitted_preprocessor.transform(X_test_raw)
    X_test_clean = _array_to_dataframe(X_test_clean_array, feature_names)

    print("[預處理模組] 處理完成！")
    return X_train_clean, X_test_clean, y_train, y_test, fitted_preprocessor


def preprocess_for_inference(
    new_data_df: pd.DataFrame,
    fitted_preprocessor: Any,
) -> pd.DataFrame:
    """
    [提供給 系統推論/預測使用]
    使用已經擬合好的參數處理新資料，確保結果的一致性。

    原版此函式是空殼（直接回傳原始資料），已補上完整實作。

    ※ 只能 transform，絕對不能 fit。
    ※ 推論階段不能刪列（每一列都是要預測的資料）。

    Parameters
    ----------
    new_data_df : pd.DataFrame
        需要預測的新資料（不含 target 欄）
    fitted_preprocessor : sklearn ColumnTransformer
        由 preprocess_for_training() 回傳的第五個值

    Returns
    -------
    pd.DataFrame
        轉換後的特徵矩陣，欄位名稱與訓練集輸出完全一致
    """
    if fitted_preprocessor is None:
        raise ValueError(
            "fitted_preprocessor 是 None。\n"
            "請先執行 preprocess_for_training() 並保存回傳的 preprocessor，\n"
            "再傳入此函式。"
        )

    # inf → NaN（推論階段不刪列，只清 inf）
    cleaned = new_data_df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    n_inf = np.isinf(cleaned[numeric_cols]).sum().sum()
    if n_inf > 0:
        cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
        print(f"  [推論前處理] 替換了 {n_inf:,} 個 inf / -inf 值")

    # 直接套用之前存下來的轉換規則（只 transform，不 fit）
    result_array  = fitted_preprocessor.transform(cleaned)
    feature_names  = fitted_preprocessor.get_feature_names_out()
    return _array_to_dataframe(result_array, feature_names)
