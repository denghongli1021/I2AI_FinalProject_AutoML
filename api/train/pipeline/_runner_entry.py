# -*- coding: utf-8 -*-
"""
_runner_entry.py — 子行程入口

兩種模式 (擇一):
  A. --csv  PATH                       單 CSV,自己做 80/20 split (測試模式 + 實驗室 raw 來源)
  B. --train-csv PATH --test-csv PATH  pre-split,跳過內部 split (實驗室 preprocessed 來源)

通用 args:
  --target / --metric / --fast / --time-limit / --ts / --skip-tabular / --skip-dl / --no-nas

stdout 兩種行:
  普通 log 行           → 父行程當 log 串出去
  __RESULT_JSON__:{...} → 最終結果 (ok=bool + scores/perModel/...)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _detect_target_col(df, requested):
    if requested and requested in df.columns:
        return requested
    for cand in ("target", "label", "class", "y", "c"):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def _detect_task(y_raw):
    """object/bool → classification;整數且 nunique 小 → classification;否則 regression。"""
    if y_raw.dtype == object or y_raw.dtype == bool:
        return "classification"
    n_unique = y_raw.nunique()
    return "classification" if (n_unique <= 50 and n_unique / max(len(y_raw), 1) < 0.30) else "regression"


def _prepare_xy(df, target_col, np, le=None):
    """從 DataFrame 抽出 X (數值矩陣) + y (LabelEncoder 編碼)。回 (X, y_encoded, classes, le)。"""
    X = (df.drop(columns=[target_col])
           .select_dtypes(include=[np.number])
           .fillna(0).values.astype(np.float32))
    y_raw = df[target_col]
    if le is None:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y_raw.astype(str).values)
    else:
        y_enc = le.transform(y_raw.astype(str).values)
    return X, y_enc, le


def _dump_ensemble_bundle(result, *, task_type, n_classes, target_col, label_encoder,
                          global_cfg, feature_names_raw, bundle_path,
                          y_center=None, y_scale=None):
    """
    把 PipelineResult / PipelineRegResult 組成完整 ensemble bundle pickle
    (含 fold models + blender + stacker),dump 到 bundle_path。
    回傳 None 表示無法 persist (某些 config cache hit 沒 fold artifacts)。
    回傳 bundle_path 表示成功。

    回歸用法:傳 task_type="regression"、n_classes=1、label_encoder=None,
            並用 y_center / y_scale 帶 RobustScaler 的中位數 + IQR 參數
            (推論時要 inverse-transform 才能還原原 y 尺度)。

    Bundle 結構見 [設計] 文件:configs 是每 config 的 fold list,blender/stacker 是 fitted 物件。
    """
    import pickle as _pickle

    # 任何一個 config 沒 fold artifacts → blender/stacker 在訓練時用全部 config 的 OOF fit,
    # 推論時若少 config,輸入維度對不上 → 拒絕 persist 比較安全。
    if any(folds is None for folds in result.per_config_folds):
        missing = [i for i, f in enumerate(result.per_config_folds) if f is None]
        print(f"[bundle] config {missing} 來自快取沒有 fold artifacts → 跳過 ensemble persist "
              f"(清掉 artifacts/ 後重訓可解決)")
        return None

    bundle = {
        "version": 1,
        "task_type": task_type,
        "n_classes": int(n_classes),
        "target": target_col,
        "label_encoder": label_encoder,
        "global_cfg": dict(global_cfg or {}),
        "ts_preprocessor": None,    # 未來 preprocessed source 用
        "configs": [
            {"tag": tag, "folds": folds}
            for tag, folds in zip(result.model_tags, result.per_config_folds)
        ],
        "blender": result.blender,
        "stacker": result.stacker,
        "model_tags": list(result.model_tags),
        "feature_names_raw": list(feature_names_raw),
        # 回歸:RobustScaler 的中位數 + IQR (推論時 y_pred = y_pred_scaled * y_scale + y_center)
        "y_center": float(y_center) if y_center is not None else None,
        "y_scale": float(y_scale) if y_scale is not None else None,
    }
    os.makedirs(os.path.dirname(bundle_path) or ".", exist_ok=True)
    with open(bundle_path, "wb") as f:
        _pickle.dump(bundle, f, protocol=_pickle.HIGHEST_PROTOCOL)
    return bundle_path


def _extract_feature_importance(result, per_model_meta):
    """
    從 ensemble 裡 OOF 分數最高的 tabular config 抽 feature_importance,
    回傳 list of [name, score] tuple (依 score 降冪)。沒抓到回 []。
    """
    TABULAR = {"lgbm", "xgb", "catboost", "rf", "extra_trees"}
    # extra_trees 等含底線的 tag 用 longest-prefix 匹配,不能 split("_")[0] (會切成 "extra")
    _SORTED_TAB = sorted(TABULAR, key=len, reverse=True)
    def _tag_model_type(tag: str) -> str:
        # 回歸的 tag 會有 "reg_" 前綴,先剝掉再看
        t = tag[4:] if tag.startswith("reg_") else tag
        for n in _SORTED_TAB:
            if t.startswith(n + "_"):
                return n
        return t.split("_")[0]

    candidates = []
    for i, (tag, meta) in enumerate(zip(result.model_tags, per_model_meta)):
        if _tag_model_type(tag) not in TABULAR:
            continue
        if meta.get("oofScore") is None:
            continue
        folds = result.per_config_folds[i] if i < len(result.per_config_folds) else None
        if not folds:
            continue
        candidates.append((meta["oofScore"], i, folds))
    if not candidates:
        return []

    candidates.sort(reverse=True)   # 高分排前
    _, _best_i, best_folds = candidates[0]
    first_fold = best_folds[0]
    model = first_fold.get("model")
    fb = first_fold.get("fb")
    if model is None or not hasattr(model, "feature_importances_"):
        return []
    imps = list(model.feature_importances_)
    # 取 transformed 後的特徵名稱;沒有就 f0/f1/...
    try:
        if fb is not None and hasattr(fb, "get_feature_names_out"):
            names = list(fb.get_feature_names_out())
        else:
            names = [f"f{i}" for i in range(len(imps))]
    except Exception:
        names = [f"f{i}" for i in range(len(imps))]
    if len(names) != len(imps):
        names = [f"f{i}" for i in range(len(imps))]
    pairs = sorted(zip(names, imps), key=lambda p: float(p[1]), reverse=True)
    return [[str(n), float(s)] for n, s in pairs]


def _prepare_xy_reg(df, target_col, np):
    """回歸專用:X (數值矩陣) + y (float 連續值,不編碼)。"""
    X = (df.drop(columns=[target_col])
           .select_dtypes(include=[np.number])
           .fillna(0).values.astype(np.float32))
    y = df[target_col].astype(float).fillna(df[target_col].astype(float).mean()).values.astype(np.float32)
    return X, y


def _run_regression(X_tr, y_tr, X_te, y_te, target_col, source_tag,
                    test_has_label, args, t0, *, is_ts: bool,
                    feature_names_raw=None):
    """回歸 pipeline (用 daniel 的 pipeline_time.run_regression)。
    支援 TS / 非 TS — is_ts 只影響 splitMode 標籤,run_regression 內部一律走 TS folds。
    自己輸出 __RESULT_JSON__,回傳 exit code。
    feature_names_raw:訓練時的原始欄名(給 bundle / SHAP / batch predict 對齊用)。
    """
    import json as _json
    import numpy as _np
    from src.config import ARTIFACTS_DIR, DEVICE
    import pipeline_time as _pt
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    budget = _pt.TimeBudget(limit_sec=args.time_limit, t_start=t0)
    cfg = _pt.get_cfg_time(args.fast, n_samples=len(y_tr))

    # 每次跑用獨立 artifacts dir (timestamp 為後綴),防止快取命中跳過重訓
    run_artifacts_dir = os.path.join(ARTIFACTS_DIR, "api", source_tag, str(int(t0)))
    result = _pt.run_regression(
        X_tr, y_tr, X_te, cfg, budget,
        skip_tabular=args.skip_tabular,
        skip_dl=args.skip_dl,
        artifacts_dir=run_artifacts_dir,
        metric="rmse",
    )

    # test_blend / test_stack 是回歸單值預測 (1D)
    blend = _np.asarray(result.test_blend).ravel()
    stack = _np.asarray(result.test_stack).ravel()

    def _reg_scores(y_true, y_pred):
        y_true = _np.asarray(y_true, dtype=float).ravel()
        y_pred = _np.asarray(y_pred, dtype=float).ravel()
        rmse = float(_np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        return rmse, r2, mae

    rmse = r2 = mae = None
    rmse_blend = r2_blend = rmse_stack = r2_stack = None
    best_pred = stack  # 預設用 stack
    best_ensemble = "stack"
    if test_has_label and y_te is not None:
        rmse_blend, r2_blend, _ = _reg_scores(y_te, blend)
        rmse_stack, r2_stack, _ = _reg_scores(y_te, stack)
        # RMSE 越小越好
        if rmse_blend < rmse_stack:
            best_pred, best_ensemble = blend, "blend"
            rmse, r2, mae = _reg_scores(y_te, blend)
        else:
            best_pred, best_ensemble = stack, "stack"
            rmse, r2, mae = _reg_scores(y_te, stack)

    def _r(v): return round(v, 4) if v is not None else None

    # ── per_model OOF score (給前端 Pipeline 詳細卡列各模型分數) ─────────────
    # OOF 已 inverse-transform 回原 y 尺度,直接拿 y_tr 比即可
    per_model = []
    y_tr_arr = _np.asarray(y_tr, dtype=float).ravel()
    for tag, oof_pred in zip(result.model_tags, result.all_oof):
        try:
            # OOF 可能有 NaN (mask=0 的位置);只用 valid 部分
            oof_arr = _np.asarray(oof_pred, dtype=float).ravel()
            valid = ~_np.isnan(oof_arr)
            if valid.sum() >= 2:
                _s = float(r2_score(y_tr_arr[valid], oof_arr[valid]))
            else:
                _s = None
        except Exception as _e:
            print(f"[OOF] R² 計算失敗 ({tag}): {_e}", flush=True)
            _s = None
        # 前端 perModel 卡顯示 tag 不要 "reg_" 前綴
        clean_tag = tag[4:] if tag.startswith("reg_") else tag
        per_model.append({"tag": clean_tag, "oofScore": _r(_s)})

    # ── Ensemble bundle 持久化 (給 batch predict / re-login 用) ──────────
    bundle_path = _dump_ensemble_bundle(
        result,
        task_type="regression",
        n_classes=1,
        target_col=target_col,
        label_encoder=None,
        global_cfg=cfg,
        feature_names_raw=feature_names_raw or [],
        bundle_path=os.path.join(run_artifacts_dir, "ensemble_bundle.pkl"),
        y_center=getattr(result, "y_center", None),
        y_scale=getattr(result, "y_scale", None),
    )
    print(f"[bundle] _dump_ensemble_bundle 回傳: {bundle_path!r}", flush=True)

    # ── Insights / SHAP 用的圖表資料 ───────────────────────────────────────
    fi_pairs = _extract_feature_importance(result, per_model)[:200]
    _CAP = 5000
    if test_has_label and y_te is not None:
        test_true_out = [float(v) for v in _np.asarray(y_te[:_CAP]).tolist()]
    else:
        test_true_out = None
    test_pred_out = [float(v) for v in best_pred[:_CAP].tolist()]
    _SHAP_CAP = 50
    x_test_sample = _np.asarray(X_te[:_SHAP_CAP]).tolist()

    split_mode = "Chronological (TS regression)" if is_ts else "Random (regression)"

    out = {
        "ok": True,
        "taskType": "regression",
        "metric": "RMSE",
        "bestEnsemble": best_ensemble,
        "rmse": _r(rmse),
        "r2": _r(r2),
        "mae": _r(mae),
        # Pipeline 詳細卡片:Blend / Stack 各自的 R² (跟 bestScore 同單位,直接可比)
        "scoreBlend": _r(r2_blend),
        "scoreStack": _r(r2_stack),
        "rmseBlend": _r(rmse_blend),
        "rmseStack": _r(rmse_stack),
        # 給前端排行榜用:回歸用 R² 當 testScore (越大越好),沒 label 時 None
        "bestScore": _r(r2),
        "scoreSource": "test" if test_has_label else "none",
        "nTrain": int(len(y_tr)),
        "nTest": int(X_te.shape[0]),
        "nClasses": 1,  # 給前端 hydration 統一介面 (分類版有這欄)
        "nFeatures": int(X_tr.shape[1]),
        "elapsedSec": round(time.time() - t0, 2),
        "splitMode": split_mode,
        "isTimeSeries": bool(is_ts),
        "device": DEVICE,
        "perModel": per_model,
        "target": target_col,
        "testHasLabel": bool(test_has_label),
        "predictions": [float(v) for v in best_pred.tolist()],
        "predictionsBlend": [float(v) for v in blend.tolist()],
        "predictionsStack": [float(v) for v in stack.tolist()],
        # 新增:ensemble bundle + Insights 圖表資料
        "ensembleBundlePath": bundle_path,
        "featureNames": list(feature_names_raw or []),
        "featureImportance": fi_pairs,
        "testTrueDecoded": test_true_out,
        "testPredDecoded": test_pred_out,
        "xTestSample": x_test_sample,
    }
    print(f"__RESULT_JSON__:{_json.dumps(out, ensure_ascii=False)}")
    return 0


# ── 兼容舊呼叫者:_run_ts_regression 是 _run_regression(is_ts=True) 的別名 ───
def _run_ts_regression(X_tr, y_tr, X_te, y_te, target_col, source_tag,
                       test_has_label, args, t0, feature_names_raw=None):
    return _run_regression(X_tr, y_tr, X_te, y_te, target_col, source_tag,
                           test_has_label, args, t0, is_ts=True,
                           feature_names_raw=feature_names_raw)


def main():
    parser = argparse.ArgumentParser()
    # 模式 A
    parser.add_argument("--csv", default=None)
    # 模式 B (pre-split)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    # 通用
    parser.add_argument("--target", default=None)
    parser.add_argument("--ts", action="store_true")
    parser.add_argument("--metric", default="f1", choices=["f1", "accuracy"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--time-limit", type=float, default=0)
    parser.add_argument("--skip-tabular", action="store_true")
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--no-nas", action="store_true")
    args = parser.parse_args()

    # Windows cp950 → utf-8 (Daniel 的 emoji/中文 print 才不會炸)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, HERE)

    # 確認模式
    presplit = bool(args.train_csv and args.test_csv)
    if not presplit and not args.csv:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '需提供 --csv 或 (--train-csv + --test-csv)'})}")
        return 1

    try:
        import numpy as np
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from src.config import DEVICE, SEED, ARTIFACTS_DIR
        import pipeline as _pl
        from src.metrics import calculate_score, get_metric_name
    except Exception as e:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': f'import 失敗: {e}', 'trace': traceback.format_exc()}, ensure_ascii=False)}")
        return 1

    t0 = time.time()

    try:
        if presplit:
            # ── 模式 B: pre-split CSV ─────────────────────────────────────
            if not (os.path.isfile(args.train_csv) and os.path.isfile(args.test_csv)):
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': 'train-csv 或 test-csv 不存在'})}")
                return 1
            df_tr = pd.read_csv(args.train_csv)
            df_te = pd.read_csv(args.test_csv)
            target_col = _detect_target_col(df_tr, args.target)

            # test CSV 是否有 target 欄? 沒有就是 Kaggle 風的「無 label predict」模式
            test_has_label = target_col in df_te.columns

            # 任務判斷:用 train (有時加 test) 的 y 集合
            from sklearn.preprocessing import LabelEncoder
            if test_has_label:
                y_combined = pd.concat([df_tr[target_col], df_te[target_col]], ignore_index=True)
                task = _detect_task(y_combined)
                fit_target_series = y_combined
            else:
                task = _detect_task(df_tr[target_col])
                fit_target_series = df_tr[target_col]

            if task == "regression":
                # TS / 非 TS 回歸都走 pipeline_time.run_regression (daniel 架構設計)
                X_tr_r, y_tr_r = _prepare_xy_reg(df_tr, target_col, np)
                if test_has_label:
                    X_te_r, y_te_r = _prepare_xy_reg(df_te, target_col, np)
                else:
                    X_te_r = (df_te.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32))
                    y_te_r = None
                if X_tr_r.shape[1] != X_te_r.shape[1]:
                    print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': f'train ({X_tr_r.shape[1]}) / test ({X_te_r.shape[1]}) 特徵數不一致'})}")
                    return 1
                # 原始欄名 (給 SHAP / batch predict 對齊用) — 排除 target 後的數值欄
                feat_names_reg = list(df_tr.drop(columns=[target_col])
                                            .select_dtypes(include=[np.number]).columns)
                src_tag = os.path.splitext(os.path.basename(args.train_csv))[0]
                _kind = "TS" if args.ts else "non-TS"
                print(f"[Pipeline] target={target_col} n_train={len(y_tr_r)} n_test={X_te_r.shape[0]} "
                      f"task=regression({_kind}) device={DEVICE} test_has_label={test_has_label}")
                sys.stdout.flush()
                return _run_regression(X_tr_r, y_tr_r, X_te_r, y_te_r, target_col, src_tag,
                                       test_has_label, args, t0,
                                       is_ts=bool(args.ts),
                                       feature_names_raw=feat_names_reg)

            le = LabelEncoder()
            le.fit(fit_target_series.astype(str).values)
            X_tr, y_tr, _ = _prepare_xy(df_tr, target_col, np, le=le)
            feature_cols_classify = list(df_tr.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns)
            if test_has_label:
                X_te, y_te, _ = _prepare_xy(df_te, target_col, np, le=le)
            else:
                # 沒 label — 只抽特徵 X,y_te 設成 dummy (反正不會用到分數計算)
                X_te = (df_te.select_dtypes(include=[np.number])
                              .fillna(0).values.astype(np.float32))
                y_te = None
            n_classes = len(le.classes_)

            if X_tr.shape[1] == 0:
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '預處理後沒有數值欄位'})}")
                return 1
            if X_tr.shape[1] != X_te.shape[1]:
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': f'train ({X_tr.shape[1]}) / test ({X_te.shape[1]}) 特徵數不一致'})}")
                return 1

            split_mode = "Pre-split (從預處理模組)" if test_has_label else "Pre-split (Kaggle predict)"
            source_tag = os.path.splitext(os.path.basename(args.train_csv))[0]
            is_forecasting = bool(args.ts)
        else:
            # ── 模式 A: 單 CSV,自己 split ──────────────────────────────────
            if not os.path.isfile(args.csv):
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': f'CSV 不存在: {args.csv}'})}")
                return 1
            df = pd.read_csv(args.csv)
            target_col = _detect_target_col(df, args.target)
            y_raw = df[target_col]
            task = _detect_task(y_raw)

            if task == "regression":
                # 共用:抽特徵 + 原始欄名
                X_all_r, y_all_r = _prepare_xy_reg(df, target_col, np)
                if X_all_r.shape[1] == 0:
                    print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '沒有可用的數值欄位'})}")
                    return 1
                feat_names_reg = list(df.drop(columns=[target_col])
                                        .select_dtypes(include=[np.number]).columns)
                if args.ts:
                    # 時序回歸 → Chronological 80/20 (尊重時間順序,不能 shuffle)
                    idx = int(len(X_all_r) * 0.8)
                    X_tr_r, X_te_r = X_all_r[:idx], X_all_r[idx:]
                    y_tr_r, y_te_r = y_all_r[:idx], y_all_r[idx:]
                    split_label = "Chronological"
                else:
                    # 非時序回歸 → Random 80/20 (daniel 架構設計)
                    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
                        X_all_r, y_all_r, test_size=0.2, random_state=SEED)
                    split_label = "Random"
                src_tag = os.path.splitext(os.path.basename(args.csv))[0]
                _kind = "TS" if args.ts else "non-TS"
                print(f"[Pipeline] target={target_col} n_train={len(y_tr_r)} n_test={len(y_te_r)} "
                      f"task=regression({_kind}) device={DEVICE} split={split_label}")
                sys.stdout.flush()
                return _run_regression(X_tr_r, y_tr_r, X_te_r, y_te_r, target_col, src_tag,
                                       True, args, t0,
                                       is_ts=bool(args.ts),
                                       feature_names_raw=feat_names_reg)

            X_all, y_all, le = _prepare_xy(df, target_col, np)
            feature_cols_classify = list(df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns)
            n_classes = len(le.classes_)
            if X_all.shape[1] == 0:
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '沒有可用的數值欄位 (pipeline 僅讀數值型,請先做 one-hot 或選只含數值欄的資料集)'})}")
                return 1

            # NOTE: 不要再用 `args.ts and task != "classification"`,那會讓「單一 CSV + 勾時序 +
            # 分類 target」的情境跑成 stratified random fold,破壞時序回測 → label leakage。
            # Mode B 一律用 bool(args.ts),這裡跟它對齊。
            is_forecasting = bool(args.ts)
            try:
                if not is_forecasting:
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)
                    split_mode = "Random Stratified"
                else:
                    idx = int(len(X_all) * 0.8)
                    X_tr, X_te = X_all[:idx], X_all[idx:]
                    y_tr, y_te = y_all[:idx], y_all[idx:]
                    split_mode = "Chronological"
            except ValueError:
                X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=SEED)
                split_mode = "Random (no stratify)"
            source_tag = os.path.splitext(os.path.basename(args.csv))[0]
            test_has_label = True  # 模式 A 永遠是 self-split,test 一定有 label

        n_test_for_log = len(y_te) if y_te is not None else X_te.shape[0]
        print(f"[Pipeline] target={target_col} "
              f"n_train={len(y_tr)} n_test={n_test_for_log} n_classes={n_classes} "
              f"device={DEVICE} split={split_mode} ts={is_forecasting} test_has_label={test_has_label}")
        sys.stdout.flush()

        # ── 呼叫 Daniel pipeline ──────────────────────────────────────────
        budget = _pl.TimeBudget(limit_sec=args.time_limit, t_start=t0)
        cfg = _pl.get_cfg(args.fast, n_samples=len(y_tr))
        cfg["is_timeseries"] = is_forecasting

        # 每次跑用獨立 artifacts dir (timestamp 為後綴),避免同 CSV 第二次跑時
        # OOF/.npy 快取命中導致 fold model 沒重訓 → ensemble bundle 拒絕 persist
        run_artifacts_dir = os.path.join(ARTIFACTS_DIR, "api", source_tag, str(int(t0)))

        result = _pl.run(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            is_ts=is_forecasting,
            artifacts_dir=run_artifacts_dir,
            metric=args.metric,
        )

        # ── 評估 + 預測 ────────────────────────────────────────────────────
        # test_blend / test_stack 是 N×C 機率矩陣 (分類);取 argmax 拿 class index。
        m_name = get_metric_name(args.metric)

        def _decode_preds(proba):
            """N×C 機率 → 原始類別 label (用 LabelEncoder.inverse_transform)。"""
            import numpy as _np
            arr = _np.asarray(proba)
            idx = arr.argmax(axis=1) if arr.ndim == 2 else arr.astype(int)
            return le.inverse_transform(idx)

        preds_blend_labels = _decode_preds(result.test_blend)
        preds_stack_labels = _decode_preds(result.test_stack)

        # 計分 — 只在 test 有 label 時才有意義
        if test_has_label and y_te is not None:
            score_blend = float(calculate_score(y_te, result.test_blend, metric=args.metric))
            score_stack = float(calculate_score(y_te, result.test_stack, metric=args.metric))
            best_preds_proba = result.test_stack if score_stack >= score_blend else result.test_blend
            best_score = float(max(score_stack, score_blend))
            acc = float(calculate_score(y_te, best_preds_proba, metric="accuracy"))
            f1 = float(calculate_score(y_te, best_preds_proba, metric="f1"))
            best_ensemble = "stack" if score_stack >= score_blend else "blend"
        else:
            # Kaggle 風:test 沒 label,只回預測,不評分
            score_blend = score_stack = best_score = acc = f1 = None
            best_ensemble = "stack"  # 預設用 stack (通常較強)

        elapsed = round(time.time() - t0, 2)

        # 守門:pipeline 沒訓練到任何 model → 拋明確錯誤,不要回 0% 假成功
        # (常見於 Render free tier 記憶體不足,subprocess 被 OOM kill 後 daniel pipeline 回空殼)
        if not result.model_tags or len(result.model_tags) == 0:
            print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': 'pipeline 跑完但 0 個模型 — 可能是 OOM / 環境缺套件 / 預算太小被全跳過。請檢查 server logs。', 'modelTags': list(result.model_tags or []), 'elapsedSec': round(time.time() - t0, 2)}, ensure_ascii=False)}")
            return 1

        # 各模型 OOF score (train 一定有 label,所以這個一定能算)
        # 注意:result.all_oof 是 N×C 機率矩陣,calculate_score (= sklearn f1/acc) 只吃 1D label,
        # 所以先 argmax 把機率轉成 class index 再算。
        import numpy as _np
        def _oof_to_labels(oof_arr):
            arr = _np.asarray(oof_arr)
            return arr.argmax(axis=1) if arr.ndim == 2 else arr

        per_model = []
        per_model_oof_labels = []
        for tag, oof in zip(result.model_tags, result.all_oof):
            try:
                oof_labels = _oof_to_labels(oof)
                s = float(calculate_score(y_tr, oof_labels, metric=args.metric))
            except Exception as _e:
                print(f"[OOF] calculate_score 失敗 ({tag}): {_e}")
                s = None
                oof_labels = None
            per_model.append({"tag": tag, "oofScore": s})
            per_model_oof_labels.append(oof_labels)

        # OOF-based fallback:test 沒 label 時,把各 model OOF 的最大 score 當 reference
        oof_best_score = None
        oof_acc = oof_f1 = None
        if not test_has_label:
            try:
                valid_oof_scores = [m["oofScore"] for m in per_model if m["oofScore"] is not None]
                if valid_oof_scores:
                    oof_best_score = float(max(valid_oof_scores))
                # 對 train 整體用最佳 OOF 算 acc/f1
                valid_indices = [i for i, m in enumerate(per_model) if m["oofScore"] is not None]
                if valid_indices:
                    best_idx = max(valid_indices, key=lambda i: per_model[i]["oofScore"])
                    best_labels = per_model_oof_labels[best_idx]
                    if best_labels is not None:
                        oof_acc = float(calculate_score(y_tr, best_labels, metric="accuracy"))
                        oof_f1  = float(calculate_score(y_tr, best_labels, metric="f1"))
            except Exception as _e:
                print(f"[OOF] fallback 計算失敗: {_e}")

        # 把最佳 ensemble 的預測 (decoded label) 一起回 — 給 Option B 寫 submission.csv 用
        best_preds_labels = preds_stack_labels if best_ensemble == "stack" else preds_blend_labels

        def _r(v): return round(v, 4) if v is not None else None

        # ── Ensemble bundle 持久化 (給 batch predict / re-login 用) ──────────
        # 診斷:印出每 config 的 fold artifact 狀態
        _folds_status = []
        for _i, _f in enumerate(getattr(result, "per_config_folds", []) or []):
            _tag = result.model_tags[_i] if _i < len(result.model_tags) else f"c{_i}"
            if _f is None:
                _folds_status.append(f"{_tag}=CACHED(None)")
            else:
                _folds_status.append(f"{_tag}={len(_f)}folds")
        print(f"[bundle] per_config_folds: {' / '.join(_folds_status) if _folds_status else '(empty)'}",
              flush=True)
        bundle_path = _dump_ensemble_bundle(
            result,
            task_type="classification",
            n_classes=n_classes,
            target_col=target_col,
            label_encoder=le,
            global_cfg=cfg,
            feature_names_raw=feature_cols_classify,
            bundle_path=os.path.join(run_artifacts_dir, "ensemble_bundle.pkl"),
        )
        print(f"[bundle] _dump_ensemble_bundle 回傳: {bundle_path!r}", flush=True)
        # 從 ensemble 中最佳 tabular config 抽 feature_importance (cap 200 個)
        fi_pairs = _extract_feature_importance(result, per_model)[:200]
        # testTrue / testPred — 給 Insights 圖表用 (cap 5000 筆,防 SSE done 事件爆)
        _CAP = 5000
        if test_has_label and y_te is not None:
            test_true_decoded = le.inverse_transform(y_te[:_CAP]).tolist()
            test_true_out = [str(v) for v in test_true_decoded]
        else:
            test_true_out = None
        test_pred_out = [str(v) for v in best_preds_labels[:_CAP].tolist()]
        # X_test 樣本 (cap 50 筆) — 給 SHAP 當 background;raw 特徵值 + 對應 featureNames
        _SHAP_CAP = 50
        x_test_sample = _np.asarray(X_te[:_SHAP_CAP]).tolist()

        out = {
            "ok": True,
            "taskType": "classification",
            "metric": m_name,
            "scoreBlend": _r(score_blend),
            "scoreStack": _r(score_stack),
            "bestScore": _r(best_score) if best_score is not None else _r(oof_best_score),
            "bestEnsemble": best_ensemble,
            "accuracy": _r(acc) if acc is not None else _r(oof_acc),
            "f1": _r(f1) if f1 is not None else _r(oof_f1),
            "scoreSource": "test" if test_has_label else "oof",  # 給前端區分顯示
            "oofBestScore": _r(oof_best_score),
            "oofAccuracy": _r(oof_acc),
            "oofF1": _r(oof_f1),
            "nTrain": int(len(y_tr)),
            "nTest": int(X_te.shape[0]),
            "nClasses": int(n_classes),
            "nFeatures": int(X_tr.shape[1]),
            "elapsedSec": elapsed,
            "splitMode": split_mode,
            "isTimeSeries": bool(is_forecasting),
            "device": DEVICE,
            "perModel": per_model,
            "target": target_col,
            "classes": [str(c) for c in le.classes_],
            "testHasLabel": bool(test_has_label),
            # 預測值 (decoded 回原始 label,給前端 / submission.csv 用)
            "predictions": [str(v) for v in best_preds_labels.tolist()],
            "predictionsBlend": [str(v) for v in preds_blend_labels.tolist()],
            "predictionsStack": [str(v) for v in preds_stack_labels.tolist()],
            # 新增:ensemble bundle + Insights 圖表資料
            "ensembleBundlePath": bundle_path,
            "featureNames": feature_cols_classify,
            "featureImportance": fi_pairs,
            "testTrueDecoded": test_true_out,
            "testPredDecoded": test_pred_out,
            "xTestSample": x_test_sample,   # 50 筆 raw 特徵,給 SHAP 用
        }
        print(f"__RESULT_JSON__:{json.dumps(out, ensure_ascii=False)}")
        return 0

    except Exception as e:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': str(e), 'trace': traceback.format_exc()[:2000]}, ensure_ascii=False)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
