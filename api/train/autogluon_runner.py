# -*- coding: utf-8 -*-
"""
autogluon_runner.py — 跟 daniel_runner 同樣的 subprocess 包裝
=============================================================
為什麼用 subprocess (跟 daniel 同理):
  · autogluon import 慢 (~5-10s),又會在 sys.path 灌一堆 module → in-process 會
    污染主 API namespace
  · TabularPredictor.fit() 跑 N 分鐘,subprocess 比較好掐死 / 取消
  · autogluon 內部 print 很多,subprocess 才能乾淨 redirect 到 SSE log
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Optional


_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNNER_SCRIPT = os.path.join(_HERE, "_autogluon_entry.py")


# autogluon 的 log 階段判斷 — 把進度條 0 ~ 100 串出來給前端
_STAGE_PATTERNS = [
    (re.compile(r"Loading data|train_data is provided"),          3,  "資料載入"),
    (re.compile(r"Beginning AutoGluon training|Performing"),      8,  "啟動 AutoGluon"),
    (re.compile(r"Inferring problem_type|Train Data"),            12, "資料分析"),
    (re.compile(r"AutoGluon training beginning|Fitting"),          18, "開始 Fit"),
    (re.compile(r"Fitting model:\s*KNeighbors"),                   25, "KNN"),
    (re.compile(r"Fitting model:\s*LightGBM"),                     35, "LightGBM"),
    (re.compile(r"Fitting model:\s*XGBoost"),                      45, "XGBoost"),
    (re.compile(r"Fitting model:\s*CatBoost"),                     55, "CatBoost"),
    (re.compile(r"Fitting model:\s*RandomForest"),                 62, "RandomForest"),
    (re.compile(r"Fitting model:\s*ExtraTrees"),                   68, "ExtraTrees"),
    (re.compile(r"Fitting model:\s*NeuralNet"),                    78, "NeuralNet"),
    (re.compile(r"Fitting model:\s*WeightedEnsemble|Fitting model: Ensemble"), 92, "Ensemble"),
    (re.compile(r"AutoGluon training complete"),                   97, "訓練完成"),
    (re.compile(r"tarball 完成"),                                  98, "持久化"),
]


def _classify_log_line(line: str) -> tuple[str, Optional[tuple[int, str]]]:
    lo = line.strip()
    if not lo:
        return ("info", None)
    progress = None
    for patt, pct, step in _STAGE_PATTERNS:
        if patt.search(lo):
            progress = (pct, step)
            break
    if "Error" in lo or "error" in lo or "Traceback" in lo or "錯誤" in lo:
        level = "error"
    elif "Warning" in lo or "warn" in lo or "警告" in lo:
        level = "warning"
    elif "completed" in lo or "complete" in lo or "Best" in lo or "score" in lo.lower():
        level = "success"
    elif lo.startswith("[") or lo.startswith("===") or lo.startswith("──"):
        level = "info"
    else:
        level = "muted"
    return (level, progress)


def run_autogluon(
    csv_bytes: bytes | None = None,
    file_name: str | None = None,
    options: dict[str, Any] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    *,
    train_csv_bytes: bytes | None = None,
    test_csv_bytes: bytes | None = None,
    train_file_name: str | None = None,
    csv_path: str | None = None,
    train_csv_path: str | None = None,
    test_csv_path: str | None = None,
    cancel_token=None,
) -> dict[str, Any]:
    """跟 daniel_runner.run_pipeline 介面對齊 — caller 跟 SSE 包裝幾乎可共用。

    options keys (都可省):
      target / timeLimit / preset
    """
    options = options or {}

    def _emit(ev):
        if on_progress:
            try: on_progress(ev)
            except Exception: pass

    have_single = csv_bytes is not None or csv_path is not None
    have_split = (train_csv_bytes is not None or train_csv_path is not None) and \
                 (test_csv_bytes is not None or test_csv_path is not None)
    if not have_single and not have_split:
        return {"ok": False, "error": "需提供 csv_bytes/csv_path 或 (train + test)"}

    tmp_files: list[str] = []
    def _bytes_to_tmp(data: bytes, prefix: str, suffix: str = ".csv") -> str:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False, prefix=prefix) as tf:
            tf.write(data)
            path = tf.name
        tmp_files.append(path)
        return path

    if have_single:
        tmp_csv = csv_path if csv_path is not None else _bytes_to_tmp(
            csv_bytes, "autogluon_", os.path.splitext(file_name or "input.csv")[1] or ".csv",
        )
        tmp_train = tmp_test = None
    else:
        base = os.path.splitext(train_file_name or "preprocessed.csv")[0]
        tmp_train = train_csv_path if train_csv_path is not None \
            else _bytes_to_tmp(train_csv_bytes, f"autogluon_{base}_train_")
        tmp_test = test_csv_path if test_csv_path is not None \
            else _bytes_to_tmp(test_csv_bytes, f"autogluon_{base}_test_")
        tmp_csv = None

    cmd = [sys.executable, "-u", _RUNNER_SCRIPT]
    if tmp_csv:
        cmd += ["--csv", tmp_csv]
    else:
        cmd += ["--train-csv", tmp_train, "--test-csv", tmp_test]
    if options.get("target"):
        cmd += ["--target", str(options["target"])]
    # 預設沒 time_limit (使用者沒傳就讓 autogluon 自己決定);傳 0 也視為沒設
    _tl = options.get("timeLimit", 0)
    if _tl and float(_tl) > 0:
        cmd += ["--time-limit", str(float(_tl))]
    if options.get("preset"):
        cmd += ["--preset", str(options["preset"])]

    _emit({"type": "log", "msg": f"啟動 autogluon (preset={options.get('preset','medium_quality')}, "
           f"time_limit={'autogluon default' if not _tl else f'{_tl}s'})", "level": "info"})
    _emit({"type": "progress", "pct": 1, "step": "啟動中"})

    t0 = time.time()
    result_dict: dict[str, Any] | None = None

    from api.bootstrap import subprocess_env
    env = subprocess_env()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
    )

    cancelled = False
    try:
        assert proc.stdout is not None
        while True:
            if cancel_token is not None and cancel_token.is_set():
                cancelled = True
                _emit({"type": "log", "msg": "收到取消訊號,終止 autogluon subprocess...", "level": "warning"})
                try: proc.terminate()
                except Exception: pass
                try: proc.wait(timeout=1.5)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
                break
            raw_line = proc.stdout.readline()
            if not raw_line:
                if proc.poll() is not None:
                    break
                continue
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("__RESULT_JSON__:"):
                try:
                    result_dict = json.loads(line[len("__RESULT_JSON__:"):])
                except Exception as e:
                    result_dict = {"ok": False, "error": f"無法解析 result line: {e}"}
                continue
            level, progress = _classify_log_line(line)
            _emit({"type": "log", "msg": line, "level": level})
            if progress is not None:
                pct, step = progress
                _emit({"type": "progress", "pct": pct, "step": step})
        if not cancelled:
            proc.wait()
    finally:
        for p in tmp_files:
            try: os.unlink(p)
            except Exception: pass

    if cancelled:
        return {"ok": False, "cancelled": True, "error": "使用者取消訓練",
                "elapsedSec": round(time.time() - t0, 2)}

    elapsed = round(time.time() - t0, 2)
    if result_dict is None:
        rc = proc.returncode
        return {"ok": False,
                "error": f"autogluon subprocess 結束 (returncode={rc}) 但沒回結果。可能 OOM / autogluon 沒裝 / Python crash。",
                "elapsedSec": elapsed}

    result_dict["elapsedSec"] = result_dict.get("elapsedSec", elapsed)
    return result_dict
