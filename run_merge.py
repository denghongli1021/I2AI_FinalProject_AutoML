"""合併 pipeline_batch_results.csv 與 baseline_batch_results.csv → pipeline_batch_results.csv"""
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
P_PATH = os.path.join(HERE, "pipeline_batch_results.csv")
B_PATH = os.path.join(HERE, "baseline_batch_results.csv")

frames = []

if os.path.exists(P_PATH):
    p = pd.read_csv(P_PATH)
    p.insert(0, "source", "pipeline")
    frames.append(p)
    print(f"[OK] pipeline_batch_results.csv  → {len(p)} rows")
else:
    print("[WARN] pipeline_batch_results.csv not found — pipeline 可能尚未完成")

if os.path.exists(B_PATH):
    b = pd.read_csv(B_PATH)
    b.insert(0, "source", "baseline")
    frames.append(b)
    print(f"[OK] baseline_batch_results.csv   → {len(b)} rows")
else:
    print("[WARN] baseline_batch_results.csv not found — baseline 可能尚未完成")

if not frames:
    print("[ERROR] 無任何結果可合併"); sys.exit(1)

merged = pd.concat(frames, ignore_index=True, sort=False)

# 統一欄位順序（缺少的欄位補 NaN）
COLS = ["source", "dataset", "type", "task", "n_train", "n_test",
        "accuracy", "f1_macro", "score", "elapsed_s"]
for c in COLS:
    if c not in merged.columns:
        merged[c] = None
merged = merged[COLS]

merged.to_csv(P_PATH, index=False, encoding="utf-8-sig")
print(f"\n[合併完成] → {P_PATH}  ({len(merged)} rows total)")
print(merged.to_string(index=False))
