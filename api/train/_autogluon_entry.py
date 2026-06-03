# -*- coding: utf-8 -*-
"""
_autogluon_entry.py — Autogluon engine 的 subprocess 入口
=========================================================
跟 daniel/_runner_entry.py 同一條 contract:
  · 吃 CSV(單檔自切 80/20,或 pre-split train+test)
  · 輸出單一行 `__RESULT_JSON__:{...}` 給上層 (autogluon_runner.py) parse
  · 其它 print 給 SSE log 即時顯示用

跟 daniel pipeline 的差別:
  · 用 autogluon-tabular 的 TabularPredictor.fit() 一條龍 (內含 HPO + ensemble)
  · 預設沒 time_limit (使用者沒傳就讓 autogluon 自己決定)
  · 訓練產物是一整個資料夾 (~/AutogluonModels/...),persist 時要打包成 tar.gz
  · 整個 dir 重新 load 才能 predict — 不像 daniel 是單一 pickle bundle
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import tempfile
import time
import traceback


def _detect_task(y_raw):
    """跟 _runner_entry._detect_task 同一條 heuristic — 跨 engine 保持一致。"""
    if y_raw.dtype == object or y_raw.dtype == bool:
        return "classification"
    n_unique = y_raw.nunique()
    return "classification" if (n_unique <= 50 and n_unique / max(len(y_raw), 1) < 0.30) else "regression"


def _detect_target_col(df, hint):
    """跟 _runner_entry._detect_target_col 同邏輯。"""
    if hint and hint in df.columns:
        return hint
    for cand in ("target", "label", "class", "y"):
        if cand in df.columns:
            return cand
    return df.columns[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None)
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--target", default=None)
    parser.add_argument("--time-limit", type=float, default=0,
                       help="0 = autogluon default (None = 沒上限)")
    parser.add_argument("--preset", default="medium_quality",
                       help="autogluon presets: best_quality / high_quality / good_quality / medium_quality / experimental_quality")
    args = parser.parse_args()

    # Windows cp950 → utf-8
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    presplit = bool(args.train_csv and args.test_csv)
    if not presplit and not args.csv:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': '需提供 --csv 或 (--train-csv + --test-csv)'})}")
        return 1

    t0 = time.time()

    # Import 抓在 main() 裡 — 沒裝 autogluon 也不會讓整個 module load 失敗
    try:
        import pandas as pd
        import numpy as np
    except Exception as e:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': f'pandas/numpy import 失敗: {e}'})}")
        return 1

    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': 'autogluon 沒裝 — pip install autogluon.tabular 後再試。'})}")
        return 1
    except Exception as e:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': f'autogluon import 失敗: {e}', 'trace': traceback.format_exc()[:2000]}, ensure_ascii=False)}")
        return 1

    try:
        # ── 載資料 ──────────────────────────────────────────────────────
        if presplit:
            if not (os.path.isfile(args.train_csv) and os.path.isfile(args.test_csv)):
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': 'train-csv 或 test-csv 不存在'})}")
                return 1
            train_df = pd.read_csv(args.train_csv)
            test_df = pd.read_csv(args.test_csv)
            target_col = _detect_target_col(train_df, args.target)
            test_has_label = target_col in test_df.columns
            split_mode = "Pre-split"
        else:
            if not os.path.isfile(args.csv):
                print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': f'CSV 不存在: {args.csv}'})}")
                return 1
            df = pd.read_csv(args.csv)
            target_col = _detect_target_col(df, args.target)
            from sklearn.model_selection import train_test_split
            try:
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42,
                                                     stratify=df[target_col] if _detect_task(df[target_col]) == "classification" else None)
                split_mode = "Random Stratified" if _detect_task(df[target_col]) == "classification" else "Random"
            except ValueError:
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                split_mode = "Random (no stratify)"
            test_has_label = True

        # 任務偵測
        task = _detect_task(train_df[target_col])
        if task == "classification":
            problem_type = "binary" if train_df[target_col].nunique() == 2 else "multiclass"
        else:
            problem_type = "regression"

        print(f"[Autogluon] target={target_col} n_train={len(train_df)} n_test={len(test_df)} "
              f"task={task} problem_type={problem_type} split={split_mode} preset={args.preset}")
        sys.stdout.flush()

        # ── 訓練 ──────────────────────────────────────────────────────
        predictor_dir = tempfile.mkdtemp(prefix="autogluon_predictor_")
        predictor = TabularPredictor(
            label=target_col,
            path=predictor_dir,
            problem_type=problem_type,
            verbosity=2,   # 0=silent, 4=verbose;2 給 SSE log 用差不多
        )
        fit_kwargs: dict = {"presets": args.preset}
        if args.time_limit and args.time_limit > 0:
            fit_kwargs["time_limit"] = float(args.time_limit)
        predictor.fit(train_df, **fit_kwargs)

        # ── 評估 ──────────────────────────────────────────────────────
        if test_has_label:
            lb = predictor.leaderboard(test_df, silent=True)
        else:
            lb = predictor.leaderboard(silent=True)
        # leaderboard column 是 score_test / score_val / pred_time_test 等;轉成 dict 給上層
        lb_records = lb.to_dict("records")

        X_test = test_df.drop(columns=[target_col], errors="ignore")
        preds = predictor.predict(X_test)
        preds_list = [v.item() if hasattr(v, "item") else v for v in preds.tolist()]

        rmse = r2 = mae = acc = f1 = best_score = None
        if test_has_label:
            from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
            y_true = test_df[target_col]
            if task == "classification":
                acc = float(accuracy_score(y_true, preds))
                try:
                    f1 = float(f1_score(y_true, preds, average="macro"))
                except Exception:
                    f1 = None
                best_score = f1 if f1 is not None else acc
            else:
                # squared=False 在 newer sklearn 改名 → 用 np.sqrt
                rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
                r2 = float(r2_score(y_true, preds))
                mae = float(mean_absolute_error(y_true, preds))
                best_score = r2

        # ── 持久化 ──────────────────────────────────────────────────
        # autogluon 模型不是單一 pickle 是整個 dir,打 tar.gz 成 bytes,給上層走 file blob
        bundle_path = os.path.join(predictor_dir, "..", f"autogluon_bundle_{int(t0)}.tar.gz")
        bundle_path = os.path.abspath(bundle_path)
        with tarfile.open(bundle_path, mode="w:gz") as tar:
            tar.add(predictor_dir, arcname="autogluon")
        print(f"[Autogluon] tarball 完成: {bundle_path} ({os.path.getsize(bundle_path)/1024/1024:.1f}MB)")

        best_model_name = lb_records[0].get("model") if lb_records else "WeightedEnsemble"

        # Leaderboard 給前端 — autogluon 內部模型樹有 5-20+ 個,展開成 perModel 跟 daniel 對齊
        per_model = []
        for row in lb_records:
            per_model.append({
                "tag": str(row.get("model", "?")),
                "oofScore": float(row.get("score_test")) if row.get("score_test") is not None
                            else (float(row.get("score_val")) if row.get("score_val") is not None else None),
                "predTimeS": float(row.get("pred_time_test")) if row.get("pred_time_test") is not None else None,
                "fitTimeS": float(row.get("fit_time")) if row.get("fit_time") is not None else None,
            })

        def _r(v): return round(v, 4) if v is not None else None

        out = {
            "ok": True,
            "engine": "autogluon",
            "taskType": task,
            "metric": ("F1" if task == "classification" else "R²"),
            "bestModel": str(best_model_name),
            "bestScore": _r(best_score),
            "rmse": _r(rmse),
            "r2": _r(r2),
            "mae": _r(mae),
            "accuracy": _r(acc),
            "f1": _r(f1),
            "presetUsed": args.preset,
            "nTrain": int(len(train_df)),
            "nTest": int(len(test_df)),
            "nClasses": int(train_df[target_col].nunique()) if task == "classification" else 1,
            "nFeatures": int(X_test.shape[1]),
            "elapsedSec": round(time.time() - t0, 2),
            "splitMode": split_mode,
            "device": "cpu",   # autogluon 預設 CPU
            "target": target_col,
            "testHasLabel": bool(test_has_label),
            "predictions": preds_list[:5000],   # cap
            "predictionsBlend": None,           # autogluon 沒這概念
            "predictionsStack": None,
            "scoreBlend": None,
            "scoreStack": None,
            "perModel": per_model,
            "bundlePath": bundle_path,
            # autogluon 的 leaderboard — 給前端顯示完整內部模型樹
            "leaderboardRaw": lb_records[:50],
            "testTrueDecoded": [str(v) for v in y_true[:5000]] if test_has_label else None,
            "testPredDecoded": [str(v) for v in preds[:5000]],
        }
        print(f"__RESULT_JSON__:{json.dumps(out, ensure_ascii=False, default=str)}")
        return 0

    except Exception as e:
        print(f"__RESULT_JSON__:{json.dumps({'ok': False, 'error': str(e), 'trace': traceback.format_exc()[:3000]}, ensure_ascii=False)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
