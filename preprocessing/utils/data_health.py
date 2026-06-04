# preprocessing/utils/data_health.py
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from typing import Any, Dict, List, Optional

# 💡 新增功能 2：稽核報告輸出（JSON + HTML）
import json
from pathlib import Path
from datetime import datetime

# 💡 新增功能 3：進階異常偵測（Ensemble Outlier + 標籤雜訊）
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


# ──────────────────────────────────────────────────────────────────
# Cramér's V：衡量兩個類別欄位的關聯強度
# ──────────────────────────────────────────────────────────────────

def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    計算 Cramér's V 統計量（0 = 無關，1 = 完全相關）。
    使用偏差修正版本（Bias-Corrected Cramér's V），對小樣本更可靠。

    為什麼要用 Cramér's V？
    Pearson 相關係數只能計算「數值 vs 數值」。
    如果目標欄或特徵欄是類別型（字串），Pearson 無法使用。
    Cramér's V 基於卡方統計量，可以衡量任意兩個類別欄位的關聯程度，
    適合偵測「類別特徵 vs 類別目標」的洩漏風險。
    """
    try:
        confusion = pd.crosstab(x, y)
        chi2, _, _, _ = chi2_contingency(confusion)
        n = len(x)
        r, k = confusion.shape
        if n == 0 or min(r - 1, k - 1) <= 0:
            return 0.0
        phi2     = chi2 / n
        phi2_c   = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r_c      = r - (r - 1) ** 2 / (n - 1)
        k_c      = k - (k - 1) ** 2 / (n - 1)
        denom    = min(k_c - 1, r_c - 1)
        return float(np.sqrt(phi2_c / denom)) if denom > 0 else 0.0
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────
# 主要健檢函式
# ──────────────────────────────────────────────────────────────────

def generate_health_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    掃描 DataFrame，產出完整的資料健康診斷報告。

    報告結構設計為 JSON 友善格式，可直接串接前端 UI 儀表板。
    開發階段可用 print_health_report(report) 在終端機查看。

    Parameters
    ----------
    df : pd.DataFrame
        待診斷的原始資料（唯讀，不修改）
    target_col : str, optional
        目標（標籤）欄位名稱

    Returns
    -------
    dict
        {
            total_rows, total_columns, perfect_columns,
            missing_summary, duplicate_rows,
            constant_columns, suspected_id_columns, inf_columns,
            outlier_summary,          ← 新增：IQR 離群值掃描
            target_info,
            leakage_candidates,       ← 增強：含 Cramér's V 類別洩漏偵測
            info, warnings
        }
    """
    n_rows, n_cols = df.shape

    report: Dict[str, Any] = {
        "total_rows":           n_rows,
        "total_columns":        n_cols,
        "perfect_columns":      0,
        "missing_summary":      {},
        "duplicate_rows":       0,
        "constant_columns":     [],
        "suspected_id_columns": [],
        "inf_columns":          [],
        "outlier_summary":      [],   # 新增
        "ensemble_outlier_summary": [],   # 新增功能 3：IQR+IsolationForest+LOF 投票
        "label_noise_candidates":   [],   # 新增功能 3：疑似標籤錯誤樣本
        "target_info":          {},
        "leakage_candidates":   [],
        "info":                 [],
        "warnings":             [],
    }

    if n_rows == 0:
        report["warnings"].append("🚨 資料集是空的（0 列），無法進行健康檢查。")
        return report

    _check_missing_values(df, n_rows, report)
    _check_inf_values(df, report)
    _check_duplicates(df, n_rows, report)
    _check_constant_columns(df, target_col, report)
    _check_suspected_id_columns(df, n_rows, target_col, report)
    _check_outliers(df, target_col, report)        # 新增
    _check_outliers_ensemble(df, target_col, report)  # 新增功能 3：集成異常偵測
    _check_label_noise(df, target_col, report)        # 新增功能 3：標籤雜訊偵測
    _check_target_column(df, target_col, report)
    _check_target_leakage(df, target_col, report)  # 增強（加入 Cramér's V）

    return report


# ──────────────────────────────────────────────────────────────────
# 私有檢查函式（單一職責，方便維護與擴充）
# ──────────────────────────────────────────────────────────────────

def _check_missing_values(
    df: pd.DataFrame, n_rows: int, report: dict
) -> None:
    """缺失值掃描，三個嚴重程度給予不同警告等級"""
    missing_counts = df.isnull().sum()
    missing_only   = missing_counts[missing_counts > 0]
    report["perfect_columns"] = df.shape[1] - len(missing_only)

    for col, count in missing_only.items():
        ratio = count / n_rows
        report["missing_summary"][col] = {
            "count": int(count),
            "ratio": round(float(ratio), 4),
        }
        if ratio > 0.6:
            report["warnings"].append(
                f"⚠️  欄位 '{col}' 缺失率高達 {ratio:.1%}，系統將自動整欄刪除。"
            )
        elif ratio > 0.3:
            report["warnings"].append(
                f"⚠️  欄位 '{col}' 缺失率 {ratio:.1%}，系統將自動填補，請確認填補結果合理。"
            )
        else:
            report["info"].append(
                f"ℹ️  欄位 '{col}' 有 {count:,} 筆缺失（{ratio:.1%}），系統自動填補。"
            )

    if missing_only.empty:
        report["info"].append("✅ 所有欄位均無缺失值。")
    else:
        report["info"].append(f"ℹ️  共 {len(missing_only)} 個欄位有缺失值，系統將自動處理。")


def _check_inf_values(df: pd.DataFrame, report: dict) -> None:
    """
    掃描 inf / -inf 值。
    StandardScaler / KNNImputer 碰到 inf 會直接 ValueError 崩潰。
    """
    for col in df.select_dtypes(include=[np.number]).columns:
        if np.isinf(df[col]).any():
            cnt = int(np.isinf(df[col]).sum())
            report["inf_columns"].append(col)
            report["warnings"].append(
                f"⚠️  欄位 '{col}' 含有 {cnt:,} 個 inf / -inf 值，"
                "系統將自動替換為 NaN 再填補。"
            )


def _check_duplicates(
    df: pd.DataFrame, n_rows: int, report: dict
) -> None:
    """
    掃描完全重複列。
    重複列若跨越 train/test split，模型等於偷看過測試集的答案，評估指標虛高。
    """
    dup_count = int(df.duplicated().sum())
    report["duplicate_rows"] = dup_count
    if dup_count > 0:
        report["warnings"].append(
            f"⚠️  發現 {dup_count:,} 筆完全重複列（{dup_count/n_rows:.1%}），系統將自動刪除。"
        )
    else:
        report["info"].append("✅ 無重複列。")


def _check_constant_columns(
    df: pd.DataFrame, target_col: Optional[str], report: dict
) -> None:
    """
    掃描常數欄（唯一值 ≤ 1）。
    常數欄對模型無資訊量，且 StandardScaler 計算標準差=0 → 除以零崩潰。
    """
    for col in df.columns:
        if target_col and col == target_col:
            continue
        if df[col].nunique(dropna=True) <= 1:
            report["constant_columns"].append(col)
    if report["constant_columns"]:
        report["warnings"].append(
            f"⚠️  常數欄位（唯一值 ≤ 1），將自動剔除：{report['constant_columns']}"
        )


def _check_suspected_id_columns(
    df: pd.DataFrame, n_rows: int, target_col: Optional[str], report: dict
) -> None:
    """
    掃描疑似整數型 ID 欄（整數型且唯一值比例 ≥ 95%）。
    ID 欄會讓模型「記住」每個樣本的 ID，嚴重 overfitting。
    注意：只偵測整數型，float 連續欄天生幾乎每值不同，不能用唯一比例判斷。
    """
    int_dtypes = ["int8","int16","int32","int64","uint8","uint16","uint32","uint64"]
    for col in df.select_dtypes(include=int_dtypes).columns:
        if target_col and col == target_col:
            continue
        if df[col].nunique(dropna=True) / n_rows >= 0.95:
            report["suspected_id_columns"].append(col)
    if report["suspected_id_columns"]:
        report["warnings"].append(
            f"⚠️  疑似整數 ID 欄位，將自動剔除：{report['suspected_id_columns']}"
        )


def _check_outliers(
    df: pd.DataFrame, target_col: Optional[str], report: dict
) -> None:
    """
    IQR 法掃描數值欄位的離群值。

    使用 3×IQR 門檻（比標準 1.5×IQR 更保守，減少誤報）：
        下界 = Q1 - 3 × IQR
        上界 = Q3 + 3 × IQR

    為什麼離群值是問題？
    StandardScaler 的均值和標準差會被極端值嚴重扭曲。
    例如一個欄位有 99 個值在 [0, 10]，只有 1 個值是 10000，
    均值會被拉到遠離大多數資料的位置，縮放後的資料完全失去意義。
    此時應改用 RobustScaler（基於中位數和 IQR，對離群值不敏感）。
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if target_col and col == target_col:
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper   = q1 - 3 * iqr, q3 + 3 * iqr
        outlier_mask   = (series < lower) | (series > upper)
        outlier_ratio  = float(outlier_mask.mean())
        outlier_count  = int(outlier_mask.sum())
        if outlier_ratio > 0.02:   # 超過 2% 的值是離群值才警告
            report["outlier_summary"].append({
                "column":        col,
                "outlier_count": outlier_count,
                "outlier_ratio": round(outlier_ratio, 4),
                "q1":  round(float(q1), 4),
                "q3":  round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            })
            report["warnings"].append(
                f"⚠️  欄位 '{col}' 有 {outlier_count:,} 個離群值（{outlier_ratio:.1%}），"
                "StandardScaler 可能被扭曲，建議評估是否改用 RobustScaler。"
            )


def _check_target_column(
    df: pd.DataFrame, target_col: Optional[str], report: dict
) -> None:
    """
    分析目標欄位，自動判斷分類 vs 迴歸任務，並偵測類別不平衡。
    """
    if not target_col:
        return
    if target_col not in df.columns:
        report["warnings"].append(
            f"🚨 致命錯誤：找不到目標欄位 '{target_col}'！現有欄位：{list(df.columns)}"
        )
        return

    target  = df[target_col]
    n_unique = target.nunique()
    is_clf   = n_unique <= 50

    report["target_info"] = {
        "column":        target_col,
        "dtype":         str(target.dtype),
        "unique_values": int(n_unique),
        "missing_count": int(target.isnull().sum()),
        "task_type":     "classification" if is_clf else "regression",
    }

    if target.isnull().any():
        report["warnings"].append(
            f"🚨 目標欄 '{target_col}' 有 {target.isnull().sum():,} 筆缺失值，"
            "這些列將在訓練前被自動刪除。"
        )

    if is_clf:
        vc = target.value_counts(normalize=True)
        report["target_info"]["class_distribution"] = {
            str(k): round(float(v), 4) for k, v in vc.items()
        }
        max_r, min_r = float(vc.iloc[0]), float(vc.iloc[-1])
        if max_r >= 0.85:
            report["warnings"].append(
                f"⚠️  目標欄 '{target_col}' 嚴重不平衡：最多類 {max_r:.1%}，最少類 {min_r:.1%}。"
                "建議使用 class_weight='balanced' 或 SMOTE。"
            )
        elif max_r >= 0.70:
            report["info"].append(
                f"ℹ️  目標欄 '{target_col}' 輕微不平衡（最多 {max_r:.1%}，最少 {min_r:.1%}）。"
            )
        else:
            report["info"].append(
                f"✅ 目標欄 '{target_col}' 類別分布均衡（最多 {max_r:.1%}）。"
            )
    else:
        num = pd.to_numeric(target, errors="coerce")
        report["target_info"]["numeric_stats"] = {
            "min":    round(float(num.min()), 4),
            "max":    round(float(num.max()), 4),
            "mean":   round(float(num.mean()), 4),
            "median": round(float(num.median()), 4),
            "std":    round(float(num.std()), 4),
        }
        report["info"].append(
            f"ℹ️  目標欄 '{target_col}' 判定為迴歸任務（唯一值 {n_unique} 種）。"
        )


def _check_target_leakage(
    df: pd.DataFrame, target_col: Optional[str], report: dict
) -> None:
    """
    偵測目標洩漏（Target Leakage）候選欄位。

    什麼是目標洩漏？
    某個特徵欄位與目標欄高度相關，不是因為它有真實預測力，
    而是因為它本質上就是「答案的另一種呈現」
    （例如：purchase_date 只有購買後才有，直接洩漏了答案）。

    檢查策略（雙管齊下）：
    1. 數值特徵 vs 數值目標 → Pearson 相關係數（快速、直觀）
    2. 類別特徵 vs 類別目標 → Cramér's V（偵測字串欄位洩漏，Pearson 無法處理）

    門檻：
    - 相關性 > 0.9 → 高風險警告（🚨）
    - 相關性 > 0.8 → 中風險提示（ℹ️）
    """
    if not target_col or target_col not in df.columns:
        return

    target   = df[target_col]
    features = df.drop(columns=[target_col])
    n_unique_target = target.nunique()
    is_clf   = n_unique_target <= 50

    def _add_candidate(col: str, score: float, method: str) -> None:
        risk = "high" if score > 0.9 else "medium"
        report["leakage_candidates"].append({
            "column":      col,
            "correlation": round(score, 4),
            "method":      method,
            "risk":        risk,
        })
        icon = "🚨" if risk == "high" else "ℹ️ "
        label = "強烈懷疑目標洩漏" if risk == "high" else "建議人工確認是否洩漏"
        report["warnings" if risk == "high" else "info"].append(
            f"{icon} 欄位 '{col}' 與目標欄相關性 {score:.2f}（{method}），{label}。"
        )

    # ── 1. 數值特徵 vs 數值目標：Pearson 相關係數 ─────────────────
    try:
        target_num = pd.to_numeric(target, errors="coerce")
        if not target_num.isna().all():
            num_feats = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
            corrs = num_feats.corrwith(target_num).abs().dropna().sort_values(ascending=False)
            for col, corr in corrs.items():
                if corr > 0.8:
                    _add_candidate(col, float(corr), "Pearson")
    except Exception:
        pass

    # ── 2. 類別特徵 vs 類別目標：Cramér's V ───────────────────────
    # Pearson 無法計算字串欄位的相關性，
    # Cramér's V 基於卡方統計量，可以衡量任意兩個類別變數的關聯強度。
    if is_clf:
        try:
            target_str = target.fillna("__MISSING__").astype(str)
            cat_feats  = features.select_dtypes(include=["object", "category"])
            for col in cat_feats.columns:
                x = cat_feats[col].fillna("__MISSING__").astype(str)
                v = _cramers_v(x, target_str)
                if v > 0.8:
                    _add_candidate(col, v, "Cramér's V")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────
# 新增功能 3：進階異常偵測（Ensemble Outlier + 標籤雜訊）
# ──────────────────────────────────────────────────────────────────

def _check_outliers_ensemble(
    df: pd.DataFrame, target_col: Optional[str], report: dict
) -> None:
    """
    集成異常偵測：IQR（現有） + Isolation Forest + LOF 三種方法投票。

    為什麼要 ensemble？
    每種方法都有盲點：
    - IQR 只看單欄，偵測不到多變數離群值
    - Isolation Forest 擅長全域異常（整體上與眾不同的點）
    - LOF 擅長局部異常（在自己的鄰域中格格不入的點）
    三者投票，被 2 種以上方法標記才算「高信心異常」，降低誤報率。

    結果存入 report["ensemble_outlier_summary"]：
    一個 list，每個元素是：
    {"row_index": int, "votes": int, "methods": [str]}
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col and target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
    if numeric_df.empty or len(numeric_df) < 10:
        return

    # 填補 NaN（模型需要完整資料）
    X = numeric_df.fillna(numeric_df.median())

    # 🚀 拆彈：如果資料大於 5萬筆，隨機抽樣，否則 LOF 會 OOM
    if len(X) > 50000:
        X = X.sample(n=50000, random_state=42)
    
    n   = len(X)
    votes = np.zeros(n, dtype=int)
    methods_per_row = [[] for _ in range(n)]

    # 方法一：IQR（已在 _check_outliers 跑過，這裡重新取結果）
    for col in X.columns:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        mask = (X[col] < q1 - 3 * iqr) | (X[col] > q3 + 3 * iqr)
        for idx in X.index[mask]:
            votes[X.index.get_loc(idx)] += 1
            methods_per_row[X.index.get_loc(idx)].append("IQR")

    # 方法二：Isolation Forest
    try:
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        iso_pred = iso.fit_predict(X)          # -1 = 異常
        for i, pred in enumerate(iso_pred):
            if pred == -1:
                votes[i] += 1
                methods_per_row[i].append("IsolationForest")
    except Exception:
        pass

    # 方法三：LOF
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, n - 1), contamination=0.05)
        lof_pred = lof.fit_predict(X)          # -1 = 異常
        for i, pred in enumerate(lof_pred):
            if pred == -1:
                votes[i] += 1
                methods_per_row[i].append("LOF")
    except Exception:
        pass

    # 整合：2 票以上才記錄
    ensemble_result = []
    for i in range(n):
        if votes[i] >= 2:
            ensemble_result.append({
                "row_index": int(X.index[i]),
                "votes":     int(votes[i]),
                "methods":   methods_per_row[i],
            })

    report["ensemble_outlier_summary"] = ensemble_result
    if ensemble_result:
        report["warnings"].append(
            f"⚠️  集成異常偵測（IQR+IsolationForest+LOF）發現 "
            f"{len(ensemble_result)} 筆高信心離群樣本（≥2 種方法同時標記）。"
        )


def _check_label_noise(
    df: pd.DataFrame, target_col: Optional[str], report: dict
) -> None:
    """
    標籤雜訊偵測：用交叉驗證預測信心度抓出疑似被標錯的樣本。

    原理：
    訓練一個簡單的 RandomForest，用 5-fold CV 取得每筆樣本的預測機率。
    若「模型對該樣本真實標籤的預測機率」< confidence_threshold，
    代表這筆樣本的標籤與其特徵的模式不符，可能是標籤錯誤。

    （這是 Cleanlab 背後使用的核心概念之一，
    此處用更輕量的方式實作，不需要額外安裝套件。）

    只對分類任務執行（target 唯一值 <= 50）。
    只使用數值欄進行檢測（避免 OHE 等預處理複雜度）。
    結果存入 report["label_noise_candidates"]：
    {"row_index": int, "true_label": ..., "predicted_proba": float}
    """
    if not target_col or target_col not in df.columns:
        return

    y = df[target_col].dropna()
    if y.nunique() > 50 or y.nunique() < 2:
        return      # 迴歸任務或只有一個類別，跳過

    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=[target_col], errors="ignore"
    )
    if numeric_df.empty:
        return

    # 對齊 index
    common_idx = numeric_df.index.intersection(y.index)
    if len(common_idx) < 20:
        return      # 樣本太少，結果不可靠

    X_check = numeric_df.loc[common_idx].fillna(numeric_df.median())
    y_check = y.loc[common_idx]

    # 🚀 更高級的抽樣：分層抽樣 (確保每個類別都有被抽到)
    if len(X_check) > 50000:
        # 將特徵與標籤合併，以利 groupby 抽樣
        temp_df = X_check.copy()
        temp_df['__target__'] = y_check
        
        # 依照 target 群組，等比例抽出 20000 筆
        sampled_df = temp_df.groupby('__target__', group_keys=False).apply(
            lambda x: x.sample(int(np.rint(20000 * len(x) / len(temp_df))), random_state=42)
        )
        
        X_check = sampled_df.drop(columns=['__target__'])
        y_check = sampled_df['__target__']

    try:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_check)
        clf   = RandomForestClassifier(
            n_estimators=50, random_state=42, n_jobs=-1, max_depth=6
        )
        confidence_threshold = 0.25   # 低於此機率視為疑似標籤錯誤

        proba = cross_val_predict(
            clf, X_check, y_enc, cv=min(5, y_check.nunique()),
            method="predict_proba", n_jobs=-1,
        )
        # 取每筆樣本「真實標籤」那個類別的預測機率
        true_class_idx = [list(le.classes_).index(label)
                          for label in y_check]
        true_proba = proba[np.arange(len(proba)), true_class_idx]

        noise_candidates = []
        for i, (idx, prob) in enumerate(zip(common_idx, true_proba)):
            if prob < confidence_threshold:
                noise_candidates.append({
                    "row_index":      int(idx),
                    "true_label":     str(y_check.iloc[i]),
                    "predicted_proba": round(float(prob), 4),
                })

        report["label_noise_candidates"] = noise_candidates
        if noise_candidates:
            report["warnings"].append(
                f"⚠️  標籤雜訊偵測發現 {len(noise_candidates)} 筆疑似標錯的樣本"
                f"（模型對其真實標籤預測信心度 < {confidence_threshold:.0%}）。"
                "建議人工複查 report['label_noise_candidates']。"
            )
    except Exception as e:
        report["info"].append(f"ℹ️  標籤雜訊偵測跳過（{e}）")


# ──────────────────────────────────────────────────────────────────
# 終端機易讀輸出
# ──────────────────────────────────────────────────────────────────

def print_health_report(report: Dict[str, Any]) -> None:
    """
    將 generate_health_report() 的回傳值以易讀格式印出。
    適合開發 / debug 時快速確認資料品質。
    """
    SEP = "═" * 62
    print(f"\n{SEP}")
    print("  🏥 資料健康報告")
    print(SEP)
    print(f"  規模：{report['total_rows']:,} 列 × {report['total_columns']} 欄")
    print(f"  完整：{report['perfect_columns']} / {report['total_columns']} 欄無缺失")
    if report["duplicate_rows"] > 0:
        print(f"  重複：{report['duplicate_rows']:,} 筆")

    if report["missing_summary"]:
        print("\n  ── 缺失值摘要 ──")
        for col, info in sorted(
            report["missing_summary"].items(), key=lambda x: x[1]["ratio"], reverse=True
        ):
            bar = "█" * int(info["ratio"] * 25) + "░" * (25 - int(info["ratio"] * 25))
            icon = "🔴" if info["ratio"] > 0.6 else ("🟡" if info["ratio"] > 0.3 else "🟢")
            print(f"  {icon} {col:<22} {bar} {info['ratio']:5.1%} ({info['count']:,}筆)")

    if report.get("inf_columns"):
        print(f"\n  ⚠ inf/-inf 欄位    : {report['inf_columns']}")
    if report.get("constant_columns"):
        print(f"  ⚠ 常數欄（刪）     : {report['constant_columns']}")
    if report.get("suspected_id_columns"):
        print(f"  ⚠ 疑似 ID 欄（刪） : {report['suspected_id_columns']}")

    if report.get("outlier_summary"):
        print("\n  ── 離群值摘要 ──")
        for o in report["outlier_summary"]:
            print(
                f"  ⚠ {o['column']:<20} 離群值 {o['outlier_ratio']:.1%} "
                f"({o['outlier_count']}筆) | 合理範圍 [{o['lower_bound']}, {o['upper_bound']}]"
            )

    if report.get("leakage_candidates"):
        print("\n  ── 目標洩漏候選 ──")
        for c in report["leakage_candidates"]:
            icon = "🚨" if c["risk"] == "high" else "⚠️ "
            print(f"  {icon} {c['column']:<20} {c['method']:10} = {c['correlation']:.3f}")

    if report.get("target_info"):
        ti = report["target_info"]
        print(f"\n  ── 目標欄：{ti['column']} ({ti['task_type']}) ──")
        print(f"     型別={ti['dtype']}  唯一值={ti['unique_values']}")
        if ti.get("class_distribution"):
            for cls, r in list(ti["class_distribution"].items())[:8]:
                print(f"     {str(cls):<15} {'█'*int(r*20):<20} {r:.1%}")

    if report["info"]:
        print("\n  ── 資訊 ──")
        for m in report["info"]:
            print(f"  {m}")
    if report["warnings"]:
        print("\n  ── 警告 ──")
        for m in report["warnings"]:
            print(f"  {m}")
    print(f"{SEP}\n")


# ──────────────────────────────────────────────────────────────────
# 新增功能 2：稽核報告輸出（JSON + HTML）
# ──────────────────────────────────────────────────────────────────

def export_report_json(report: dict, filepath: str) -> None:
    """
    將 generate_health_report() 的回傳結果輸出成 JSON 檔。

    用途：
    - 讓 CI/CD pipeline 可以用程式讀取稽核結果（判斷是否有 warning 決定要不要中止訓練）
    - 方便版本控制（可以 diff 兩次訓練的資料健康狀態）

    Parameters
    ----------
    report   : generate_health_report() 的回傳值
    filepath : 輸出路徑，例如 "audit_report.json"
    """
    output = {
        "generated_at": datetime.now().isoformat(),
        "report": report,
    }
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"[data_health] 稽核報告（JSON）已儲存至：{filepath}")


def export_report_html(report: dict, filepath: str) -> None:
    """
    將 generate_health_report() 的回傳結果輸出成人類可讀的 HTML 報告。

    用途：
    - 開完會後可以直接把 HTML 附件寄給組員或 PM
    - 不需要跑 Python 就能看到資料品質摘要

    HTML 結構：
    - 標題 + 生成時間
    - 綠底：info 訊息
    - 黃底：warnings
    - 表格：missing_summary / outlier_summary / leakage_candidates
    """
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows  = report.get("total_rows", "?")
    cols  = report.get("total_columns", "?")
    warns = report.get("warnings", [])
    infos = report.get("info", [])

    def _table_rows(items: list, keys: list) -> str:
        if not items:
            return "<tr><td colspan='99'>（無）</td></tr>"
        header = "<tr>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
        body   = ""
        for item in items:
            body += "<tr>" + "".join(
                f"<td>{item.get(k, '')}</td>" for k in keys
            ) + "</tr>"
        return header + body

    warn_html = "".join(
        f"<li style='color:#b45309;background:#fef9c3;padding:4px 8px;"
        f"border-radius:4px;margin:3px 0'>{w}</li>"
        for w in warns
    ) or "<li style='color:green'>無警告</li>"

    info_html = "".join(f"<li style='color:#166534'>{i}</li>" for i in infos)

    missing_keys = ["column", "missing_count", "missing_ratio"]
    miss_data = [
        {"column": col, "missing_count": v.get("count", ""), "missing_ratio": f"{v.get('ratio',0):.1%}"}
        for col, v in report.get("missing_summary", {}).items()
    ]
    outlier_keys = ["column", "outlier_count", "outlier_ratio"]
    leakage_keys = ["column", "correlation", "method", "risk"]

    html = f"""<!DOCTYPE html>
<html lang="zh-Hant"><head>
<meta charset="UTF-8">
<title>資料健康稽核報告</title>
<style>
  body{{font-family:'Segoe UI',Arial,sans-serif;margin:32px;color:#1e293b}}
  h1{{color:#0f172a}} h2{{color:#334155;margin-top:28px}}
  table{{border-collapse:collapse;width:100%;margin-top:8px}}
  th{{background:#e2e8f0;padding:8px 12px;text-align:left}}
  td{{padding:7px 12px;border-bottom:1px solid #e2e8f0}}
  tr:hover{{background:#f8fafc}}
  .badge{{padding:2px 10px;border-radius:12px;font-size:13px}}
  .warn{{background:#fef9c3;color:#92400e}}
  .ok{{background:#dcfce7;color:#166534}}
</style>
</head><body>
<h1>📋 資料健康稽核報告</h1>
<p>生成時間：{now}　｜　資料規模：{rows:,} 列 × {cols} 欄</p>

<h2>⚠️ 警告（{len(warns)} 項）</h2>
<ul>{warn_html}</ul>

<h2>ℹ️ 一般資訊</h2>
<ul>{info_html}</ul>

<h2>缺失值摘要</h2>
<table>{_table_rows(miss_data, missing_keys)}</table>

<h2>離群值摘要</h2>
<table>{_table_rows(report.get("outlier_summary", []), outlier_keys)}</table>

<h2>目標洩漏候選</h2>
<table>{_table_rows(report.get("leakage_candidates", []), leakage_keys)}</table>

</body></html>"""

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[data_health] 稽核報告（HTML）已儲存至：{filepath}")
