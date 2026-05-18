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
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '回歸任務暫不支援 pipeline'})}")
                return 1

            le = LabelEncoder()
            le.fit(fit_target_series.astype(str).values)
            X_tr, y_tr, _ = _prepare_xy(df_tr, target_col, np, le=le)
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
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '回歸任務暫不支援 pipeline (設計上只跑分類)'})}")
                return 1

            X_all, y_all, le = _prepare_xy(df, target_col, np)
            n_classes = len(le.classes_)
            if X_all.shape[1] == 0:
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '沒有可用的數值欄位 (pipeline 僅讀數值型,請先做 one-hot 或選只含數值欄的資料集)'})}")
                return 1

            is_forecasting = args.ts and task != "classification"
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

        result = _pl.run(
            X_tr, y_tr, X_te, n_classes, cfg, budget,
            skip_tabular=args.skip_tabular,
            skip_dl=args.skip_dl,
            no_nas=args.no_nas,
            is_ts=is_forecasting,
            artifacts_dir=os.path.join(ARTIFACTS_DIR, "api", source_tag),
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

        # 各模型 OOF score (train 一定有 label,所以這個一定能算)
        per_model = []
        for tag, oof in zip(result.model_tags, result.all_oof):
            try:
                s = float(calculate_score(y_tr, oof, metric=args.metric))
            except Exception:
                s = None
            per_model.append({"tag": tag, "oofScore": s})

        # 把最佳 ensemble 的預測 (decoded label) 一起回 — 給 Option B 寫 submission.csv 用
        best_preds_labels = preds_stack_labels if best_ensemble == "stack" else preds_blend_labels

        def _r(v): return round(v, 4) if v is not None else None

        out = {
            "ok": True,
            "metric": m_name,
            "scoreBlend": _r(score_blend),
            "scoreStack": _r(score_stack),
            "bestScore": _r(best_score),
            "bestEnsemble": best_ensemble,
            "accuracy": _r(acc),
            "f1": _r(f1),
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
        }
        print(f"__RESULT_JSON__:{json.dumps(out, ensure_ascii=False)}")
        return 0

    except Exception as e:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': str(e), 'trace': traceback.format_exc()[:2000]}, ensure_ascii=False)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
