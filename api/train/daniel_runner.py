# -*- coding: utf-8 -*-
"""
daniel_runner.py — 在 API 端呼叫 Daniel pipeline 的 helper
============================================================
為什麼用 subprocess:
  - Daniel 的 pipeline.py / run_pipeline.py 在 sys.path 上動手腳,in-process 會
    跟主 API 的 import 命名空間打架
  - 大量 print() 在多執行緒 FastAPI 裡 redirect stdout 危險 (其他 thread 也會被攔)
  - pipeline 跑很久 (分鐘級),subprocess 比較好控制與終止

對外:
  run_pipeline(csv_bytes, file_name, options, on_progress) -> dict
    on_progress(ev): ev = {'type': 'log', 'msg': str, 'level': str}
                       | {'type': 'progress', 'pct': int, 'step': str}
                       | {'type': 'done', 'result': dict}
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
_RUNNER_SCRIPT = os.path.join(_HERE, "pipeline", "_runner_entry.py")


# Daniel 的 print 帶有方框符號等格式字元 — 解析 pipeline 階段給前端做 progress bar
_STAGE_PATTERNS = [
    (re.compile(r"\[2a\] Tabular Scout"),            5,   "Scout HPO"),
    (re.compile(r"\[2b\] Tabular Full HPO"),         15,  "Full HPO"),
    (re.compile(r"\[3\] MLP NAS"),                    35,  "MLP NAS"),
    (re.compile(r"\[3\] TS-?Net NAS"),                35,  "TSNet NAS"),
    (re.compile(r"\[4\] MLP Training HPO"),           50,  "MLP Training HPO"),
    (re.compile(r"\[4\] TS-?Net Training HPO"),       50,  "TSNet Training HPO"),
    (re.compile(r"\[5\] (CNN1D|TCN)"),                65,  "CNN/TCN HPO"),
    (re.compile(r"\[6\] (Transformer|PatchTST)"),     78,  "Transformer HPO"),
    (re.compile(r"\[7\] (CV|5-?Fold)"),               88,  "5-Fold CV"),
    (re.compile(r"\[8\] Ensemble A|Nelder"),          93,  "Nelder-Mead Blend"),
    (re.compile(r"\[9\] Ensemble B|Meta-?Learner"),   97,  "Meta-Learner Stacking"),
]


def _classify_log_line(line: str) -> tuple[str, Optional[tuple[int, str]]]:
    """根據一行 log 內容判斷 level,以及是否要更新 progress。"""
    lo = line.strip()
    if not lo:
        return ("info", None)
    # 進度推進 — 找第一個符合的階段
    progress = None
    for patt, pct, step in _STAGE_PATTERNS:
        if patt.search(lo):
            progress = (pct, step)
            break
    # log level
    if "錯誤" in lo or "Error" in lo or "error" in lo or "Traceback" in lo:
        level = "error"
    elif "Warning" in lo or "警告" in lo or "[SKIP]" in lo:
        level = "warning"
    elif "完成" in lo or "結果" in lo or "✓" in lo or "OOF" in lo or "Done" in lo:
        level = "success"
    elif lo.startswith("[") or lo.startswith("===") or lo.startswith("──"):
        level = "info"
    else:
        level = "muted"
    return (level, progress)


def run_pipeline(
    csv_bytes: bytes | None = None,
    file_name: str | None = None,
    options: dict[str, Any] | None = None,
    on_progress: Callable[[dict], None] | None = None,
    *,
    train_csv_bytes: bytes | None = None,
    test_csv_bytes: bytes | None = None,
    train_file_name: str | None = None,
    csv_path: str | None = None,         # 已寫好的 CSV 檔路徑 (省掉 bytes 來回 → 大矩陣走這條)
    train_csv_path: str | None = None,
    test_csv_path: str | None = None,
    cancel_token=None,   # threading.Event;set 後立刻 terminate subprocess
) -> dict[str, Any]:
    """
    兩種模式擇一 (各自可給 bytes 或「已寫好的檔案路徑」):
      A. csv_bytes/csv_path + file_name              — 單 CSV,Daniel 自己做 80/20 split
      B. (train + test) bytes 或 path               — pre-split (實驗室用預處理結果時)
    給路徑時:caller 自己擁有那些檔,run_pipeline 不會刪;給 bytes 時才寫暫存檔並負責刪除。

    options keys (都可省):
      target / timeSeries / metric / fast / timeLimit / skipTabular / skipDl / noNas
    """
    options = options or {}

    def _emit(ev):
        if on_progress:
            try: on_progress(ev)
            except Exception: pass

    # 驗證:必須擇一 (bytes 或 path 皆可)
    have_single = csv_bytes is not None or csv_path is not None
    have_split = (train_csv_bytes is not None or train_csv_path is not None) and \
                 (test_csv_bytes is not None or test_csv_path is not None)
    if not have_single and not have_split:
        return {"ok": False, "error": "需提供 csv_bytes/csv_path 或 (train + test)"}

    # 準備餵給子行程的 CSV 檔。只把「我們自己寫的暫存檔」記進 tmp_files,最後才刪;
    # caller 直接給的路徑不記、不刪 (那是 caller 的檔)。
    tmp_files: list[str] = []

    def _bytes_to_tmp(data: bytes, prefix: str, suffix: str = ".csv") -> str:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False, prefix=prefix) as tf:
            tf.write(data)
            path = tf.name
        tmp_files.append(path)
        return path

    if have_single:
        if csv_path is not None:
            tmp_csv = csv_path
        else:
            suffix = os.path.splitext(file_name or "input.csv")[1] or ".csv"
            tmp_csv = _bytes_to_tmp(csv_bytes, "daniel_", suffix)
        tmp_train = tmp_test = None
    else:
        base = os.path.splitext(train_file_name or "preprocessed.csv")[0]
        tmp_train = train_csv_path if train_csv_path is not None \
            else _bytes_to_tmp(train_csv_bytes, f"daniel_{base}_train_")
        tmp_test = test_csv_path if test_csv_path is not None \
            else _bytes_to_tmp(test_csv_bytes, f"daniel_{base}_test_")
        tmp_csv = None

    # 組指令
    cmd = [sys.executable, "-u", _RUNNER_SCRIPT, "--metric", str(options.get("metric", "f1"))]
    if tmp_csv:
        cmd += ["--csv", tmp_csv]
    else:
        cmd += ["--train-csv", tmp_train, "--test-csv", tmp_test]
    if options.get("target"):
        cmd += ["--target", str(options["target"])]
    if options.get("timeSeries"):
        cmd.append("--ts")
    if options.get("fast"):
        cmd.append("--fast")
    if options.get("timeLimit"):
        cmd += ["--time-limit", str(float(options["timeLimit"]))]
    if options.get("skipTabular"):
        cmd.append("--skip-tabular")
    if options.get("skipDl"):
        cmd.append("--skip-dl")
    if options.get("noNas"):
        cmd.append("--no-nas")

    mode_label = "pre-split (從預處理)" if tmp_train else "single CSV"
    _emit({"type": "log", "msg": f"啟動 pipeline [{mode_label}], fast={options.get('fast', False)}", "level": "info"})
    _emit({"type": "progress", "pct": 1, "step": "啟動中"})

    t0 = time.time()
    result_dict: dict[str, Any] | None = None

    from api.bootstrap import subprocess_env
    env = subprocess_env()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    cancelled = False
    try:
        assert proc.stdout is not None
        # 用 readline 而非 for loop,才能在每行之間檢查 cancel_token
        while True:
            if cancel_token is not None and cancel_token.is_set():
                cancelled = True
                _emit({"type": "log", "msg": "收到取消訊號,終止 pipeline subprocess...", "level": "warning"})
                try: proc.terminate()
                except Exception: pass
                # 給 1.5 秒收尾,還沒死就 kill
                try: proc.wait(timeout=1.5)
                except Exception:
                    try: proc.kill()
                    except Exception: pass
                break
            raw_line = proc.stdout.readline()
            if not raw_line:
                # EOF (subprocess 已結束)
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
        # 沒收到 __RESULT_JSON__ — 常見原因:OOM kill / Python crash / subprocess timeout
        rc = proc.returncode
        if rc in (-9, 137):
            hint = " (signal 9 / SIGKILL — 多半是 OOM,記憶體不夠)"
        elif rc in (-11, 139):
            hint = " (segfault — 套件衝突,可能是 numpy/torch 版本)"
        elif rc and rc != 0:
            hint = f" (非 0 return code,subprocess 異常退出)"
        else:
            hint = ""
        return {
            "ok": False,
            "error": f"pipeline subprocess 沒回結果 (rc={rc}){hint}",
            "returnCode": rc,
            "elapsedSec": elapsed,
        }
    result_dict.setdefault("elapsedSec", elapsed)
    _emit({"type": "progress", "pct": 100, "step": "完成"})
    return result_dict
