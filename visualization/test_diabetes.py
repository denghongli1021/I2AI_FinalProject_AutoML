"""
壓力測試：Diabetes 130-US Hospitals Dataset
-----------------------------------------------
資料來源：UCI Machine Learning Repository（OpenML ID: 4541）
任務：預測糖尿病病患出院後 30 天內是否再次入院（分類）
規模：~101,766 筆 × 50 個特徵
難點：
  - 大量缺失值（race, payer_code, medical_specialty 等欄位）
  - 混合型別（數值 + 類別 + 有序類別）
  - 類別不平衡（再入院率約 11%）
  - 高基數類別特徵（診斷碼 diag_1/2/3 有 700+ 種值）
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
from visualizer import AutoMLVisualizer


def preprocess_diabetes(X, y):
    """
    針對 Diabetes 130 的前處理。
    這裡示範的是基本版，實際系統會由前處理組的模組負責。
    """
    X = X.copy()

    # 1. 刪除無預測價值的 ID 欄位
    drop_cols = ['encounter_id', 'patient_nbr']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])

    # 2. 處理明確的缺失值標記（'?' 在這個資料集代表缺失）
    X = X.replace('?', np.nan)

    # 3. 刪除缺失率過高的欄位（>40%）
    missing_rate = X.isnull().mean()
    high_missing = missing_rate[missing_rate > 0.4].index.tolist()
    print(f"  刪除高缺失率欄位（>40%）：{high_missing}")
    X = X.drop(columns=high_missing)

    # 4. 處理 age：把區間字串轉為數值中位數
    #    原始值長這樣：'[0-10)', '[10-20)', ..., '[90-100)'
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    if 'age' in X.columns:
        X['age'] = X['age'].map(age_map)

    # 5. 處理診斷碼（diag_1/2/3）：700+ 種值，簡化為大類
    #    ICD-9 編碼前三碼決定疾病大類
    def simplify_diag(val):
        if pd.isnull(val) or val == '?':
            return -1
        try:
            code = float(str(val).split('.')[0].replace('V', '0').replace('E', '0'))
            if 390 <= code <= 459 or code == 785: return 1   # 循環系統
            elif 460 <= code <= 519 or code == 786: return 2  # 呼吸系統
            elif 520 <= code <= 579 or code == 787: return 3  # 消化系統
            elif 250 <= code <= 250.99: return 4              # 糖尿病本身
            elif 800 <= code <= 999: return 5                 # 外傷
            elif 710 <= code <= 739: return 6                 # 肌肉骨骼
            elif 580 <= code <= 629 or code == 788: return 7  # 泌尿系統
            elif 140 <= code <= 239: return 8                 # 腫瘤
            else: return 0                                    # 其他
        except:
            return -1

    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in X.columns:
            X[col] = X[col].apply(simplify_diag)

    # 6. 類別欄位：Label Encoding
    #    注意：OpenML 載入的資料部分欄位是 'category' dtype，
    #    需要同時處理 'object' 和 'category'，否則 XGBoost 會報錯
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype(str).fillna('Unknown').replace('nan', 'Unknown')
        X[col] = LabelEncoder().fit_transform(X[col])

    # 7. 數值欄位：缺失值填中位數
    num_cols = X.select_dtypes(include=['number']).columns.tolist()
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    # 8. 強制把所有欄位轉成 float（確保 XGBoost 能接受）
    X = X.astype(float)

    # 8. 目標變數：轉為二元分類
    #    原始值：'NO', '>30', '<30'
    #    '<30' = 30天內再入院（高風險）= 1，其餘 = 0
    y_binary = (y == '<30').astype(int)

    return X, y_binary


def run_diabetes():
    print("Step 1: 透過 OpenML API 載入 Diabetes 130 資料集...")
    print("  （首次載入需要下載，約需 30 秒）")

    # OpenML dataset ID = 4541
    data = fetch_openml(data_id=4541, as_frame=True, parser='auto')
    X_raw = data.data
    y_raw = data.target

    print(f"  原始資料：{X_raw.shape[0]:,} 筆 × {X_raw.shape[1]} 個特徵")
    print(f"  目標分布：\n{y_raw.value_counts().to_string()}")

    print("\nStep 2: 前處理（缺失值、類別編碼、診斷碼簡化）...")
    X, y = preprocess_diabetes(X_raw, y_raw)
    print(f"  前處理後：{X.shape[0]:,} 筆 × {X.shape[1]} 個特徵")
    print(f"  再入院（<30天）比例：{y.mean():.1%}")

    print("\nStep 3: Train/Test 切分（80/20）...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} 筆，Test: {len(X_test):,} 筆")

    print("\nStep 4: 訓練 XGBoost 分類模型...")
    # scale_pos_weight 處理類別不平衡
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale = neg / pos
    print(f"  類別不平衡比例：{scale:.1f}，套用 scale_pos_weight")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale,
        random_state=42,
        eval_metric='auc',
        verbosity=0
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    from sklearn.metrics import roc_auc_score, classification_report
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"  Test AUC-ROC: {auc:.4f}")

    print("\nStep 5: 產出 SHAP 視覺化圖表...")
    viz = AutoMLVisualizer(
        model=model,
        X_test=X_test,
        output_dir="diabetes_results"
    )

    # number_inpatient = 過去一年住院次數，是預測再入院的關鍵特徵
    viz.generate_all_plots(
        sample_index=0,
        target_feature='number_inpatient',
        prefix="diabetes"
    )

    print("\n完成！圖表輸出至 diabetes_results/ 目錄")
    print("  - diabetes_global.png    : 全局特徵重要性（哪些因素最影響再入院）")
    print("  - diabetes_waterfall.png : 局部瀑布圖（第 0 筆病患的預測拆解）")
    print("  - diabetes_dependence.png: 依賴散佈圖（住院次數 × 交互特徵）")


if __name__ == "__main__":
    run_diabetes()