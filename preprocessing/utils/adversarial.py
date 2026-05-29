# preprocessing/utils/adversarial.py
"""
對抗驗證 (Adversarial Validation) — Kaggle 神技。

核心概念：
  把 Train 標為 0、Test 標為 1，混在一起丟給 LightGBM 分類。
  若 LightGBM 能輕易分辨兩邊（AUC 高），代表 Train/Test 分布差異極大，
  模型在訓練集上學到的模式不會泛化到測試集。
  這時應把「最能區分兩邊」的特徵刪掉，消除分布偏差。

AUC 判讀：
  ≈ 0.5        → 完美，Train/Test 分布幾乎一樣
  0.5 ~ 0.7    → 輕微差異，可接受
  0.7 ~ 0.85   → 警告，建議刪除 top 貢獻特徵
  > 0.85       → 危險，分布嚴重不一致，模型泛化能力堪慮
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────
# 主要入口函式
# ──────────────────────────────────────────────────────────────────

def run_adversarial_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: Optional[str] = None,
    n_splits: int = 5,
    auc_threshold_warn: float = 0.7,
    auc_threshold_danger: float = 0.85,
    top_n_drop: int = 10,
    lgb_params: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    執行對抗驗證，評估 Train / Test 的分布差異程度。

    Parameters
    ----------
    train_df : pd.DataFrame
        訓練集原始資料（含或不含 target_col 皆可）。
    test_df : pd.DataFrame
        測試集原始資料。
    target_col : str, optional
        目標欄位名稱。若提供，會在建立特徵矩陣前排除此欄。
    n_splits : int
        交叉驗證折數，預設 5。
    auc_threshold_warn : float
        AUC 超過此值發出警告，預設 0.7。
    auc_threshold_danger : float
        AUC 超過此值發出危險警告，預設 0.85。
    top_n_drop : int
        建議刪除的最高重要性特徵數量，預設 10。
    lgb_params : dict, optional
        自訂 LightGBM 參數。若未提供則使用預設值。

    Returns
    -------
    dict
        {
            "auc_mean"           : float,      ← 各折 AUC 的平均值
            "auc_std"            : float,      ← 各折 AUC 的標準差
            "auc_per_fold"       : list[float],
            "verdict"            : "ok" | "warning" | "danger",
            "message"            : str,        ← 人類可讀的結論
            "n_train"            : int,
            "n_test"             : int,
            "n_features"         : int,
            "feature_importances": list[dict], ← 依重要性降序排列
            "features_to_drop"   : list[str],  ← 建議刪除的欄位清單
        }
    """
    try:
        import lightgbm as lgb
    except ImportError as e:
        raise ImportError(
            "對抗驗證需要 lightgbm。請執行：pip install lightgbm"
        ) from e

    print("\n[對抗驗證] 開始執行 Adversarial Validation...")

    # ── 1. 準備特徵矩陣 ───────────────────────────────────────────
    feature_cols = _get_common_feature_cols(train_df, test_df, target_col)
    print(f"[對抗驗證] 共同特徵數：{len(feature_cols)}")

    X_train_feat = train_df[feature_cols].copy()
    X_test_feat  = test_df[feature_cols].copy()

    # 類別欄位 → LabelEncode（LGB 也可以 native，但統一做比較乾淨）
    X_train_feat, X_test_feat = _encode_categoricals(X_train_feat, X_test_feat, feature_cols)

    # 合併並貼上偽標籤（train=0, test=1）
    X_combined = pd.concat([X_train_feat, X_test_feat], axis=0, ignore_index=True)
    y_combined = np.concatenate([
        np.zeros(len(X_train_feat), dtype=int),
        np.ones(len(X_test_feat),  dtype=int),
    ])

    print(f"[對抗驗證] 合併後資料集大小：{X_combined.shape[0]:,} 列")

    # ── 2. 交叉驗證訓練 LightGBM ──────────────────────────────────
    params = _default_lgb_params()
    if lgb_params:
        params.update(lgb_params)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auc_scores: List[float] = []
    importance_sum = np.zeros(len(feature_cols))

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_combined, y_combined)):
        X_tr, X_val = X_combined.iloc[train_idx], X_combined.iloc[val_idx]
        y_tr, y_val = y_combined[train_idx],       y_combined[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )

        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        fold_auc = roc_auc_score(y_val, y_pred)
        auc_scores.append(fold_auc)
        importance_sum += model.feature_importance(importance_type="gain")

        print(f"  Fold {fold_idx + 1}/{n_splits} — AUC: {fold_auc:.4f}")

    # ── 3. 整合結果 ────────────────────────────────────────────────
    auc_mean = float(np.mean(auc_scores))
    auc_std  = float(np.std(auc_scores))

    # 依重要性排序特徵
    importance_avg = importance_sum / n_splits
    sorted_idx = np.argsort(importance_avg)[::-1]
    feature_importances = [
        {
            "feature":    feature_cols[i],
            "importance": round(float(importance_avg[i]), 4),
        }
        for i in sorted_idx
    ]

    # 判定危險等級
    if auc_mean > auc_threshold_danger:
        verdict = "danger"
        message = (
            f"AUC = {auc_mean:.4f}（> {auc_threshold_danger}）— 危險！"
            "Train / Test 分布嚴重不一致，模型泛化能力堪慮。"
            f"強烈建議刪除 features_to_drop 中的前 {top_n_drop} 個特徵。"
        )
    elif auc_mean > auc_threshold_warn:
        verdict = "warning"
        message = (
            f"AUC = {auc_mean:.4f}（> {auc_threshold_warn}）— 警告！"
            "Train / Test 存在明顯分布差異。"
            f"建議檢視並考慮刪除 features_to_drop 中的特徵。"
        )
    else:
        verdict = "ok"
        message = (
            f"AUC = {auc_mean:.4f}（≈ 0.5）— 良好！"
            "Train / Test 分布幾乎一致，無需額外處理。"
        )

    # 🚀 升級：精準打擊邏輯 (建議刪除清單)
    features_to_drop = []
    if verdict != "ok":
        # 計算所有特徵的總重要性
        total_importance = sum(fi["importance"] for fi in feature_importances)
        
        if total_importance > 0:
            for fi in feature_importances:
                # 🛡️ 核心防護：重要性必須大於 0，且貢獻度大於 1% 才算是「作弊主謀」
                if fi["importance"] > 0 and (fi["importance"] / total_importance) > 0.01:
                    features_to_drop.append(fi["feature"])
            
            # 保險機制：就算作弊主謀很多，最多也只砍設定的 top_n_drop 數量，保護資料集不被砍殘
            features_to_drop = features_to_drop[:top_n_drop]

    result = {
        "auc_mean":            round(auc_mean, 4),
        "auc_std":             round(auc_std, 4),
        "auc_per_fold":        [round(s, 4) for s in auc_scores],
        "verdict":             verdict,
        "message":             message,
        "n_train":             len(train_df),
        "n_test":              len(test_df),
        "n_features":          len(feature_cols),
        "feature_importances": feature_importances,
        "features_to_drop":    features_to_drop,
    }

    print(f"\n[對抗驗證] {message}")
    return result


# ──────────────────────────────────────────────────────────────────
# 實用工具函式
# ──────────────────────────────────────────────────────────────────

def drop_adversarial_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    result: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    根據 run_adversarial_validation() 的回傳結果，從 Train / Test 中刪除偏差特徵。

    Parameters
    ----------
    train_df : pd.DataFrame
        原始訓練集。
    test_df : pd.DataFrame
        原始測試集。
    result : dict
        run_adversarial_validation() 的回傳值。

    Returns
    -------
    (train_clean, test_clean) : Tuple[pd.DataFrame, pd.DataFrame]
        已刪除偏差欄位的 Train 和 Test。
    """
    to_drop = result.get("features_to_drop", [])
    if not to_drop:
        print("[對抗驗證] 無需刪除任何特徵（分布差異在可接受範圍內）。")
        return train_df.copy(), test_df.copy()

    # 只刪除實際存在的欄位（防禦性編碼）
    train_cols_to_drop = [c for c in to_drop if c in train_df.columns]
    test_cols_to_drop  = [c for c in to_drop if c in test_df.columns]

    train_clean = train_df.drop(columns=train_cols_to_drop)
    test_clean  = test_df.drop(columns=test_cols_to_drop)

    print(f"[對抗驗證] 已從 Train / Test 中刪除 {len(train_cols_to_drop)} 個偏差特徵：")
    print(f"  {train_cols_to_drop}")
    return train_clean, test_clean


def print_adversarial_report(result: Dict[str, Any]) -> None:
    """
    將 run_adversarial_validation() 的回傳結果以易讀格式印出。
    """
    SEP = "═" * 62
    verdict_icon = {"ok": "✅", "warning": "⚠️ ", "danger": "🚨"}.get(result["verdict"], "❓")

    print(f"\n{SEP}")
    print("  對抗驗證報告 (Adversarial Validation Report)")
    print(SEP)
    print(f"  規模：Train {result['n_train']:,} 列 | Test {result['n_test']:,} 列 | {result['n_features']} 個共同特徵")
    print(f"  AUC  : {result['auc_mean']:.4f} ± {result['auc_std']:.4f}")

    fold_str = "  ".join(f"Fold{i+1}={s:.3f}" for i, s in enumerate(result["auc_per_fold"]))
    print(f"  各折  : {fold_str}")

    # AUC 視覺化量表（0.5 ~ 1.0 映射到 25 格）
    bar_fill = max(0, int((result["auc_mean"] - 0.5) / 0.5 * 25))
    bar_empty = 25 - bar_fill
    print(f"  量表  : |{'█' * bar_fill}{'░' * bar_empty}| 0.5 ~ 1.0")
    print(f"\n  {verdict_icon} {result['message']}")

    if result.get("feature_importances"):
        print("\n  ── Top 特徵重要性（能區分 Train/Test 的主謀）──")
        top_feats = result["feature_importances"][:15]
        max_imp = top_feats[0]["importance"] if top_feats else 1.0
        for rank, fi in enumerate(top_feats, 1):
            bar_len = int(fi["importance"] / max_imp * 20) if max_imp > 0 else 0
            flag = " ← 建議刪除" if fi["feature"] in result.get("features_to_drop", []) else ""
            print(
                f"  {rank:2d}. {fi['feature']:<25} {'█' * bar_len:<20} "
                f"{fi['importance']:,.1f}{flag}"
            )

    if result.get("features_to_drop"):
        print(f"\n  ── 建議刪除特徵（共 {len(result['features_to_drop'])} 個）──")
        for feat in result["features_to_drop"]:
            print(f"    • {feat}")

    print(f"{SEP}\n")


# ──────────────────────────────────────────────────────────────────
# 私有輔助函式
# ──────────────────────────────────────────────────────────────────

def _get_common_feature_cols(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: Optional[str],
) -> List[str]:
    """取得 Train 和 Test 的共同特徵欄位（排除 target）。"""
    common = list(set(train_df.columns) & set(test_df.columns))
    if target_col and target_col in common:
        common.remove(target_col)
    if not common:
        raise ValueError(
            "Train 和 Test 之間沒有任何共同欄位，無法執行對抗驗證。"
        )
    return sorted(common)  # 固定排序，確保可重現


def _encode_categoricals(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    將 object / category 欄位做 LabelEncoding。

    Train 和 Test 一起 fit encoder（避免 unseen label 問題），
    unknown 值填入 -1。
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return X_train, X_test

    X_train = X_train.copy()
    X_test  = X_test.copy()

    for col in cat_cols:
        # 合併 unique 值來 fit encoder
        combined_vals = pd.concat([
            X_train[col].fillna("__NaN__").astype(str),
            X_test[col].fillna("__NaN__").astype(str),
        ], ignore_index=True)

        le = LabelEncoder()
        le.fit(combined_vals)

        def _safe_transform(series: pd.Series) -> np.ndarray:
            filled = series.fillna("__NaN__").astype(str)
            # 處理 unseen label（理論上不會發生，因為已合併 fit）
            known = set(le.classes_)
            filled = filled.apply(lambda x: x if x in known else "__NaN__")
            return le.transform(filled)

        X_train[col] = _safe_transform(X_train[col])
        X_test[col]  = _safe_transform(X_test[col])

    return X_train, X_test


def _default_lgb_params() -> Dict:
    """LightGBM 預設參數，適合快速對抗驗證（不需要最優表現，只需要穩定）。"""
    return {
        "objective":      "binary",
        "metric":         "auc",
        "boosting_type":  "gbdt",
        "learning_rate":  0.05,
        "num_leaves":     31,
        "max_depth":      -1,
        "min_child_samples": 20,
        "feature_fraction":  0.8,
        "bagging_fraction":  0.8,
        "bagging_freq":      5,
        "reg_alpha":      0.1,
        "reg_lambda":     0.1,
        "random_state":   42,
        "n_jobs":         -1,
        "verbose":        -1,
    }
