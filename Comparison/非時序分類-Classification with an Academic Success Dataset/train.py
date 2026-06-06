"""
Kaggle Playground Series S4E6 - Classification with an Academic Success Dataset
==============================================================================
Task        : Multiclass Classification (Dropout / Enrolled / Graduate)
Metric      : Accuracy (Kaggle official), + R2 / F1 reported internally
Strategy    : LightGBM + XGBoost ensemble with Optuna tuning,
              Stratified K-Fold CV, feature engineering, SMOTE oversampling
Author      : Auto-generated training pipeline
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

# ── Sklearn ──────────────────────────────────────────────────────────────────
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    r2_score,
)
from sklearn.inspection import permutation_importance

# ── Imbalanced-learn ─────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("[WARN] imbalanced-learn not installed; SMOTE disabled. "
          "Install with: pip install imbalanced-learn")

# ── Gradient boosting ────────────────────────────────────────────────────────
import lightgbm as lgb
import xgboost as xgb

# ── Optuna (hyperparameter search) ───────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[WARN] Optuna not installed; using default hyperparameters. "
          "Install with: pip install optuna")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
# 路徑自動切換:Kaggle 環境 vs 本機
#   - Kaggle 上 /kaggle/input/ 唯讀,輸出必須寫 /kaggle/working/
#   - 自動掃 /kaggle/input/ 找出同時包含 train.csv + test.csv 的目錄,
#     比賽資料夾名換了也不用改
#   - Notebook 沒有 __file__,所以本機 fallback 用 cwd
def _find_kaggle_data_dir():
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    for dirpath, _dirs, files in os.walk(root):
        if "train.csv" in files and "test.csv" in files:
            return Path(dirpath)
    return None

_kaggle = _find_kaggle_data_dir()
if _kaggle:
    DATA_DIR   = _kaggle
    OUTPUT_DIR = Path("/kaggle/working")
else:
    try:
        DATA_DIR = Path(__file__).resolve().parent
    except NameError:
        DATA_DIR = Path.cwd()
    OUTPUT_DIR = DATA_DIR

TRAIN_CSV       = DATA_DIR / "train.csv"
TEST_CSV        = DATA_DIR / "test.csv"
SUBMISSION_CSV  = OUTPUT_DIR / "submission.csv"
print(f"[path] DATA_DIR   = {DATA_DIR}")
print(f"[path] SUBMISSION = {SUBMISSION_CSV}")

N_FOLDS         = 5                  # Stratified K-Fold 折數
OPTUNA_TRIALS   = 50                 # Optuna 搜索次數 (越多越好，但越慢)
RANDOM_STATE    = 42
USE_SMOTE       = True               # 是否用 SMOTE 處理類別不平衡
ENSEMBLE_WEIGHT = {"lgb": 0.55, "xgb": 0.45}  # 最終集成權重

TARGET_COL      = "Target"
ID_COL          = "id"

# ── GPU 自動偵測 ────────────────────────────────────────────────────────
# Kaggle 上要先在 Settings → Accelerator 選 GPU(T4 x2 或 P100)。
# 沒 GPU 時自動 fallback CPU,不會 crash。
def _detect_gpu() -> bool:
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi"], capture_output=True, timeout=3)
        return r.returncode == 0
    except Exception:
        return False

USE_GPU = _detect_gpu()
XGB_DEVICE = "cuda" if USE_GPU else "cpu"
# LightGBM 退回 CPU:Kaggle 的 LightGBM GPU build 對某些超參數組合(num_leaves > 2^max_depth,
# 或 min_child_samples 較小)會踩 assertion bug —
#   "Check failed: (best_split_info.left_count) > (0)"
# 這個資料集規模又小(~80k 列),CPU 跑一折 <5s,GPU overhead 反而拖慢。
# 若日後資料集 ≥ 100 萬列 + LightGBM GPU build 穩定再切回 "gpu"。
LGB_DEVICE = "cpu"
print(f"[GPU] detected = {USE_GPU}  (xgb device={XGB_DEVICE}, lgb device={LGB_DEVICE})")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1 │ Loading data")
print("=" * 65)

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

print(f"Train shape : {train_df.shape}")
print(f"Test  shape : {test_df.shape}")
print(f"\nTarget distribution:\n{train_df[TARGET_COL].value_counts()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. LABEL-ENCODE TARGET
# ═══════════════════════════════════════════════════════════════════════════════
le = LabelEncoder()
train_df[TARGET_COL] = le.fit_transform(train_df[TARGET_COL])
CLASSES = le.classes_          # ['Dropout', 'Enrolled', 'Graduate']
NUM_CLASSES = len(CLASSES)
print(f"\nEncoded classes: {dict(enumerate(CLASSES))}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2 │ Feature engineering")
print("=" * 65)

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 學業表現比率特徵 ─────────────────────────────────────────────────────
    # 第一學期通過率
    df["pass_rate_1st"] = (
        df["Curricular units 1st sem (approved)"] /
        (df["Curricular units 1st sem (enrolled)"] + 1e-6)
    )
    # 第二學期通過率
    df["pass_rate_2nd"] = (
        df["Curricular units 2nd sem (approved)"] /
        (df["Curricular units 2nd sem (enrolled)"] + 1e-6)
    )
    # 兩學期合計通過率
    total_enrolled = (
        df["Curricular units 1st sem (enrolled)"] +
        df["Curricular units 2nd sem (enrolled)"] + 1e-6
    )
    total_approved = (
        df["Curricular units 1st sem (approved)"] +
        df["Curricular units 2nd sem (approved)"]
    )
    df["pass_rate_total"] = total_approved / total_enrolled

    # ── 成績特徵 ────────────────────────────────────────────────────────────
    df["avg_grade"] = (
        df["Curricular units 1st sem (grade)"] +
        df["Curricular units 2nd sem (grade)"]
    ) / 2
    df["grade_diff"] = (
        df["Curricular units 2nd sem (grade)"] -
        df["Curricular units 1st sem (grade)"]
    )
    df["grade_improvement"] = (df["grade_diff"] > 0).astype(int)

    # ── 評估參與率 ─────────────────────────────────────────────────────────
    df["eval_rate_1st"] = (
        df["Curricular units 1st sem (evaluations)"] /
        (df["Curricular units 1st sem (enrolled)"] + 1e-6)
    )
    df["eval_rate_2nd"] = (
        df["Curricular units 2nd sem (evaluations)"] /
        (df["Curricular units 2nd sem (enrolled)"] + 1e-6)
    )

    # ── 家庭背景特徵 ────────────────────────────────────────────────────────
    df["parent_edu_avg"] = (
        df["Mother's qualification"] + df["Father's qualification"]
    ) / 2
    df["parent_occ_avg"] = (
        df["Mother's occupation"] + df["Father's occupation"]
    ) / 2

    # ── 財務風險指標 ────────────────────────────────────────────────────────
    df["financial_risk"] = (
        df["Debtor"].astype(int) +
        (1 - df["Tuition fees up to date"].astype(int)) +
        (1 - df["Scholarship holder"].astype(int))
    )

    # ── 總修課數 ───────────────────────────────────────────────────────────
    df["total_units_enrolled"] = (
        df["Curricular units 1st sem (enrolled)"] +
        df["Curricular units 2nd sem (enrolled)"]
    )
    df["total_units_approved"] = (
        df["Curricular units 1st sem (approved)"] +
        df["Curricular units 2nd sem (approved)"]
    )

    # ── 宏觀經濟複合指標 ─────────────────────────────────────────────────
    df["econ_stress"] = df["Unemployment rate"] - df["GDP"]

    return df


train_df = feature_engineering(train_df)
test_df  = feature_engineering(test_df)

# ── 刪除低重要性或高噪音欄位 ─────────────────────────────────────────────
# 根據原始論文與社群分析：Nacionality、International 對模型貢獻低，
# 且 Educational special needs 樣本過少，增加雜訊
DROP_COLS = [ID_COL]   # id 欄位一定要刪；其他欄位在後面用重要性剔除

FEATURE_COLS = [c for c in train_df.columns
                if c not in DROP_COLS + [TARGET_COL]]

X = train_df[FEATURE_COLS].values
y = train_df[TARGET_COL].values
X_test_final = test_df[FEATURE_COLS].values
test_ids = test_df[ID_COL].values

print(f"Feature count : {len(FEATURE_COLS)}")
print(f"Features      : {FEATURE_COLS[:10]} ...")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. HYPERPARAMETER SEARCH WITH OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3 │ Hyperparameter optimisation (Optuna)")
print("=" * 65)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

# ── Apply SMOTE on the search subset ────────────────────────────────────────
if USE_SMOTE and HAS_SMOTE:
    sm = SMOTE(random_state=RANDOM_STATE)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
    print(f"[SMOTE] Resampled train size: {X_tr_sm.shape[0]}")
else:
    X_tr_sm, y_tr_sm = X_tr, y_tr

# ─── LightGBM objective ────────────────────────────────────────────────────
def lgb_objective(trial):
    params = {
        "objective"       : "multiclass",
        "num_class"       : NUM_CLASSES,
        "metric"          : "multi_logloss",
        "verbosity"       : -1,
        "boosting_type"   : "gbdt",
        "device"          : LGB_DEVICE,
        "n_estimators"    : trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves"      : trial.suggest_int("num_leaves", 31, 256),
        "max_depth"       : trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state"    : RANDOM_STATE,
        "n_jobs"          : -1,
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr_sm, y_tr_sm,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

# ─── XGBoost objective ─────────────────────────────────────────────────────
def xgb_objective(trial):
    params = {
        "objective"       : "multi:softmax",
        "num_class"       : NUM_CLASSES,
        "eval_metric"     : "mlogloss",
        "n_estimators"    : trial.suggest_int("n_estimators", 500, 2000),
        "learning_rate"   : trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma"           : trial.suggest_float("gamma", 1e-8, 5.0, log=True),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state"    : RANDOM_STATE,
        "n_jobs"          : -1,
        "tree_method"     : "hist",
        "device"          : XGB_DEVICE,
        # XGBoost 2.0+ 把 early_stopping_rounds 從 fit() 移到 constructor
        "early_stopping_rounds": 50,
    }
    model = xgb.XGBClassifier(**params, verbosity=0)
    model.fit(
        X_tr_sm, y_tr_sm,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)


if HAS_OPTUNA:
    # LightGBM search
    print("  Tuning LightGBM ...")
    lgb_study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    lgb_study.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_lgb_params = lgb_study.best_params
    best_lgb_params.update({
        "objective": "multiclass", "num_class": NUM_CLASSES,
        "metric": "multi_logloss", "verbosity": -1,
        "device": LGB_DEVICE,
        "random_state": RANDOM_STATE, "n_jobs": -1,
    })
    print(f"  Best LGB accuracy : {lgb_study.best_value:.5f}")

    # XGBoost search
    print("  Tuning XGBoost ...")
    xgb_study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    xgb_study.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_xgb_params = xgb_study.best_params
    best_xgb_params.update({
        "objective": "multi:softmax", "num_class": NUM_CLASSES,
        "eval_metric": "mlogloss", "random_state": RANDOM_STATE,
        "n_jobs": -1, "tree_method": "hist", "device": XGB_DEVICE,
        "verbosity": 0,
        "early_stopping_rounds": 100,    # XGBoost 2.0+ 必須在 constructor
    })
    print(f"  Best XGB accuracy : {xgb_study.best_value:.5f}")

else:
    # Fallback: 預設超參數
    best_lgb_params = {
        "objective": "multiclass", "num_class": NUM_CLASSES,
        "metric": "multi_logloss", "verbosity": -1,
        "device": LGB_DEVICE,
        "n_estimators": 1000, "learning_rate": 0.05,
        "num_leaves": 127, "max_depth": 8,
        "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.1,
        "reg_lambda": 1.0, "random_state": RANDOM_STATE, "n_jobs": -1,
    }
    best_xgb_params = {
        "objective": "multi:softmax", "num_class": NUM_CLASSES,
        "eval_metric": "mlogloss", "n_estimators": 1000,
        "learning_rate": 0.05, "max_depth": 6,
        "min_child_weight": 5, "subsample": 0.8,
        "colsample_bytree": 0.8, "gamma": 0.1,
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": RANDOM_STATE, "n_jobs": -1,
        "tree_method": "hist", "device": XGB_DEVICE, "verbosity": 0,
        "early_stopping_rounds": 100,    # XGBoost 2.0+ 必須在 constructor
    }
    print("  [Optuna unavailable] Using default hyperparameters.")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. STRATIFIED K-FOLD CROSS-VALIDATION & OOF PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4 │ Stratified K-Fold CV training")
print("=" * 65)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_lgb  = np.zeros((len(X), NUM_CLASSES))
oof_xgb  = np.zeros((len(X), NUM_CLASSES))
test_lgb = np.zeros((len(X_test_final), NUM_CLASSES))
test_xgb = np.zeros((len(X_test_final), NUM_CLASSES))

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n  ── Fold {fold}/{N_FOLDS} ──────────────────────────────")
    X_tr_f, X_val_f = X[train_idx], X[val_idx]
    y_tr_f, y_val_f = y[train_idx], y[val_idx]

    # SMOTE on each fold's training data
    if USE_SMOTE and HAS_SMOTE:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_tr_f, y_tr_f = sm.fit_resample(X_tr_f, y_tr_f)

    # ── LightGBM ──────────────────────────────────────────────────────────
    lgb_model = lgb.LGBMClassifier(**best_lgb_params)
    lgb_model.fit(
        X_tr_f, y_tr_f,
        eval_set=[(X_val_f, y_val_f)],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    lgb_probs = lgb_model.predict_proba(X_val_f)
    oof_lgb[val_idx] = lgb_probs
    test_lgb += lgb_model.predict_proba(X_test_final) / N_FOLDS

    # ── XGBoost ───────────────────────────────────────────────────────────
    xgb_model = xgb.XGBClassifier(**best_xgb_params)
    xgb_model.fit(
        X_tr_f, y_tr_f,
        eval_set=[(X_val_f, y_val_f)],
        verbose=False,
    )
    xgb_probs = xgb_model.predict_proba(X_val_f)
    oof_xgb[val_idx] = xgb_probs
    test_xgb += xgb_model.predict_proba(X_test_final) / N_FOLDS

    # ── Fold metrics ──────────────────────────────────────────────────────
    lgb_preds = np.argmax(lgb_probs, axis=1)
    xgb_preds = np.argmax(xgb_probs, axis=1)

    # Ensemble fold predictions
    ens_probs = (
        ENSEMBLE_WEIGHT["lgb"] * lgb_probs +
        ENSEMBLE_WEIGHT["xgb"] * xgb_probs
    )
    ens_preds = np.argmax(ens_probs, axis=1)

    fold_acc = accuracy_score(y_val_f, ens_preds)
    fold_f1  = f1_score(y_val_f, ens_preds, average="weighted")
    fold_metrics.append({"fold": fold, "accuracy": fold_acc, "f1": fold_f1})
    print(f"  Fold {fold} │ Accuracy: {fold_acc:.5f} │ F1 (weighted): {fold_f1:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. OOF EVALUATION (自切割驗證集的三項指標)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 5 │ OOF (Out-Of-Fold) evaluation metrics")
print("=" * 65)

# 融合 OOF 機率
oof_ensemble = (
    ENSEMBLE_WEIGHT["lgb"] * oof_lgb +
    ENSEMBLE_WEIGHT["xgb"] * oof_xgb
)
oof_preds = np.argmax(oof_ensemble, axis=1)

# ── 1. Accuracy ──────────────────────────────────────────────────────────────
oof_accuracy = accuracy_score(y, oof_preds)

# ── 2. F1 Score (weighted，適合多分類不平衡問題) ──────────────────────────
oof_f1 = f1_score(y, oof_preds, average="weighted")

# ── 3. R² Score（對多分類使用 label 值計算，反映預測的系統性偏差）
#       多分類的 R2 通常以 label integer 計算，是一種補充指標
oof_r2 = r2_score(y, oof_preds)

print(f"\n  ┌─────────────────────────────────────────┐")
print(f"  │  OOF Metrics (Self-Split Validation)    │")
print(f"  ├─────────────────────────────────────────┤")
print(f"  │  Accuracy       : {oof_accuracy:.6f}             │")
print(f"  │  F1 (weighted)  : {oof_f1:.6f}             │")
print(f"  │  R² Score       : {oof_r2:.6f}             │")
print(f"  └─────────────────────────────────────────┘")

# ── Per-class F1 ─────────────────────────────────────────────────────────────
f1_per_class = f1_score(y, oof_preds, average=None)
print("\n  Per-class F1:")
for cls_name, f1_val in zip(CLASSES, f1_per_class):
    print(f"    {cls_name:<12}: {f1_val:.5f}")

# ── Fold 統計 ────────────────────────────────────────────────────────────────
fold_df = pd.DataFrame(fold_metrics)
print(f"\n  Fold-level stats:")
print(f"    Accuracy  mean={fold_df['accuracy'].mean():.5f}  std={fold_df['accuracy'].std():.5f}")
print(f"    F1        mean={fold_df['f1'].mean():.5f}  std={fold_df['f1'].std():.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 7. GENERATE TEST PREDICTIONS & SUBMISSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 6 │ Generating test predictions")
print("=" * 65)

test_ensemble = (
    ENSEMBLE_WEIGHT["lgb"] * test_lgb +
    ENSEMBLE_WEIGHT["xgb"] * test_xgb
)
test_preds_idx   = np.argmax(test_ensemble, axis=1)
test_preds_label = le.inverse_transform(test_preds_idx)

submission = pd.DataFrame({
    ID_COL    : test_ids,
    TARGET_COL: test_preds_label,
})
submission.to_csv(SUBMISSION_CSV, index=False)
print(f"  Submission saved → {SUBMISSION_CSV}")
print(f"  Prediction distribution:\n{pd.Series(test_preds_label).value_counts()}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8. FEATURE IMPORTANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 7 │ Feature importance (LightGBM gain, last fold)")
print("=" * 65)

importance_df = pd.DataFrame({
    "feature"   : FEATURE_COLS,
    "importance": lgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

print(importance_df.head(20).to_string(index=False))

# ── 自動剔除 importance=0 的特徵建議 ────────────────────────────────────
zero_imp = importance_df[importance_df["importance"] == 0]["feature"].tolist()
if zero_imp:
    print(f"\n  [INFO] Features with zero importance (consider dropping): {zero_imp}")

# ═══════════════════════════════════════════════════════════════════════════════
# 9. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"  Model         : LightGBM ({ENSEMBLE_WEIGHT['lgb']*100:.0f}%) + "
      f"XGBoost ({ENSEMBLE_WEIGHT['xgb']*100:.0f}%)")
print(f"  CV Strategy   : {N_FOLDS}-Fold Stratified K-Fold + SMOTE")
print(f"  Optuna Trials : {OPTUNA_TRIALS if HAS_OPTUNA else 'N/A (default params)'}")
print(f"\n  ╔════════════════════════════════════╗")
print(f"  ║  OOF Accuracy  : {oof_accuracy:.6f}        ║")
print(f"  ║  OOF F1 Score  : {oof_f1:.6f}        ║")
print(f"  ║  OOF R² Score  : {oof_r2:.6f}        ║")
print(f"  ╚════════════════════════════════════╝")
print(f"\n  Output → {SUBMISSION_CSV}")
print("=" * 65)