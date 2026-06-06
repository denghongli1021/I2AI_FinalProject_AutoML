# -*- coding: utf-8 -*-
"""
時序分類 — Detecting Reversal Points in US Equities
3-class (H / L / None) + 高維 (~68k feats) + 小樣本 (2684) + 極不平衡 (94/3/3)

策略:
  - LightGBM multi-class,feature_fraction=0.3 抽列降噪 (68k 特徵)
  - class_weight='balanced' 抵消 None 主導
  - StratifiedKFold 5 折 + early stopping,OOF 評估 macro-F1
  - 預測平均 5 折 → submission.csv (id, class_label)

跑法:
  python train_and_predict.py
"""

import gc
import os
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight


# ── 路徑自動切換:Kaggle 環境 vs 本機 ────────────────────────
# Kaggle 上 /kaggle/input/ 是唯讀,output 必須寫到 /kaggle/working/。
# 自動掃 /kaggle/input/ 找出真實 dataset 目錄,避免比賽資料夾名變動寫死失準。
def _find_kaggle_data_dir() -> str | None:
    """掃 /kaggle/input 找同時有 train.csv + test.csv 的資料夾。
    多個候選時,優先順序:
      1) 名字含 'new'(舊比賽常見 new_xxx 為最新版資料)
      2) train.csv 檔案最大者(通常是更完整 / 更新的版本)
    """
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    candidates = []
    for dirpath, _dirs, files in os.walk(root):
        if "train.csv" in files and "test.csv" in files:
            size = os.path.getsize(os.path.join(dirpath, "train.csv"))
            candidates.append((dirpath, size))
    if not candidates:
        return None
    # sort: 含 'new' 優先,再按 train.csv 大小降冪
    candidates.sort(key=lambda x: (0 if "new" in os.path.basename(x[0]).lower() else 1, -x[1]))
    chosen = candidates[0][0]
    if len(candidates) > 1:
        print(f"[path] 多個候選 → 選 {chosen}")
        for c, sz in candidates:
            print(f"        - {c}  ({sz/1e6:.1f} MB)")
    return chosen

_kaggle = _find_kaggle_data_dir()
if _kaggle:
    DATA_DIR = _kaggle
    OUT_DIR = "/kaggle/working"
else:
    # Jupyter / IPython 沒有 __file__;用 cwd 當保底
    try:
        DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        DATA_DIR = os.getcwd()
    OUT_DIR = DATA_DIR
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUB_PATH = os.path.join(OUT_DIR, "submission.csv")
print(f"[path] DATA_DIR={DATA_DIR}")
print(f"[path] SUB_PATH={SUB_PATH}")

# 5-class → 3-class 映射(資料描述定義)
FIVE_TO_THREE = {
    "HH": "H", "LH": "H",   # High patterns
    "HL": "L", "LL": "L",   # Low patterns
}

N_FOLDS = 5
SEED = 42


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data():
    log("讀取 train.csv ...")
    # low_memory=False 讓 pandas 整欄一次推斷 dtype,避免「Columns have mixed types」warning
    tr = pd.read_csv(TRAIN_PATH, low_memory=False)
    log(f"  train shape={tr.shape}")
    log("讀取 test.csv ...")
    te = pd.read_csv(TEST_PATH, low_memory=False)
    log(f"  test shape={te.shape}")

    # ── target & id ────────────────────────────────────────────
    y_raw = tr["class_label"].astype(str)
    train_ids = tr["train_id"].values
    test_ids = te["id"].values

    # ── feature 欄 (扣掉 metadata) ─────────────────────────────
    meta = {"train_id", "id", "ticker_id", "t", "class_label"}
    feat_cols = [c for c in tr.columns if c not in meta and c in te.columns]
    log(f"  feature 欄數 = {len(feat_cols)}")

    # ── 降記憶體:全部 float32 ──────────────────────────────────
    log("壓縮 dtype → float32 (省一半記憶體) ...")
    X_tr = tr[feat_cols].astype(np.float32)
    X_te = te[feat_cols].astype(np.float32)
    del tr, te
    gc.collect()
    log(f"  X_tr={X_tr.shape}, X_te={X_te.shape}")

    # ── 動態 label 處理 ──────────────────────────────────────
    # Kaggle 上 train.csv 的 class_label 可能是:
    #   (a) 已轉好的 3-class: H / L / None
    #   (b) 原版 5-class:    HH / LH / HL / LL / None  ← 需要先映射
    # 先印 raw 看到的類別,再套 5→3,確保 fail-safe
    log(f"  class_label 原始類別: {sorted(y_raw.unique().tolist())}")
    # 1) 統一 NaN / 'nan' / 空 → 'None'
    y_raw = y_raw.replace({"nan": "None", "NaN": "None", "": "None"}).fillna("None")
    # 2) 5→3 映射(若已是 3-class,replace 無事發生)
    y_raw = y_raw.replace(FIVE_TO_THREE)
    log(f"  映射後類別: {sorted(y_raw.unique().tolist())}")

    # 3) 動態 label encoding —— 只用實際出現的類別,避免 compute_class_weight 報錯
    labels = sorted(y_raw.unique().tolist())
    lab2idx = {v: i for i, v in enumerate(labels)}
    y = y_raw.map(lab2idx).astype(int).values
    log(f"  lab2idx = {lab2idx}")

    return X_tr, y, X_te, train_ids, test_ids, feat_cols, labels


def train_cv(X, y, X_test, num_classes: int):
    """5 折 stratified CV,OOF + 5 折平均預測 test。"""
    n_train = len(X)
    n_test = len(X_test)
    oof = np.zeros((n_train, num_classes), dtype=np.float32)
    preds = np.zeros((n_test, num_classes), dtype=np.float32)

    # class weight — classes 用實際 y 出現過的,不寫死 np.arange(3),避免「classes not in y」
    classes_present = np.unique(y)
    cw = compute_class_weight("balanced", classes=classes_present, y=y)
    class_w = {int(c): float(w) for c, w in zip(classes_present, cw)}
    log(f"class_weight = {class_w}")

    params = dict(
        objective="multiclass",
        num_class=num_classes,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=10,
        feature_fraction=0.3,    # 68k 特徵高度冗餘,每樹只看 30%
        bagging_fraction=0.8,
        bagging_freq=5,
        reg_alpha=0.1,
        reg_lambda=0.1,
        metric="multi_logloss",
        verbosity=-1,
        class_weight=class_w,
        n_estimators=2000,
        n_jobs=-1,
        random_state=SEED,
    )

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (tri, vai) in enumerate(skf.split(X, y)):
        t0 = time.time()
        log(f"── Fold {fold + 1}/{N_FOLDS} ─────────────────")
        log(f"  train={len(tri)} val={len(vai)}")
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X.iloc[tri], y[tri],
            eval_set=[(X.iloc[vai], y[vai])],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        best_iter = getattr(m, "best_iteration_", None)
        log(f"  best_iter={best_iter}, 耗時 {time.time()-t0:.1f}s")
        oof[vai] = m.predict_proba(X.iloc[vai])
        preds += m.predict_proba(X_test) / N_FOLDS

        del m
        gc.collect()

    return oof, preds


def main():
    log("=" * 60)
    log("Detecting Reversal Points — LightGBM CV")
    log("=" * 60)

    X_tr, y, X_te, train_ids, test_ids, feat_cols, labels = load_data()
    num_classes = len(labels)
    idx2lab = {i: v for i, v in enumerate(labels)}

    oof, preds = train_cv(X_tr, y, X_te, num_classes=num_classes)

    # ── OOF 評估 ─────────────────────────────────────────────
    y_pred = np.argmax(oof, axis=1)
    macro_f1 = f1_score(y, y_pred, average="macro")
    log(f"\n[OOF] macro-F1 = {macro_f1:.4f}")
    print(classification_report(y, y_pred, target_names=labels, zero_division=0))

    # ── submission ─────────────────────────────────────────
    final = np.argmax(preds, axis=1)
    final_lab = [idx2lab[i] for i in final]
    sub = pd.DataFrame({"id": test_ids, "class_label": final_lab})
    sub.to_csv(SUB_PATH, index=False)
    log(f"submission 已寫入: {SUB_PATH}")
    log(f"  shape={sub.shape}, 類別分佈:")
    print(sub["class_label"].value_counts().to_string())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("使用者中斷")
        sys.exit(130)
