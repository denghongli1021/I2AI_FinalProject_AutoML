"""
串聯執行：pipeline_time → baseline → 合併 combined_ts_results.csv
用法：python run_combined_ts.py
"""
import subprocess
import sys
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

def run(cmd, label):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        print(f"[錯誤] {label} 失敗（exit code {result.returncode}），中止串聯執行。")
        sys.exit(result.returncode)
    print(f"[完成] {label}")

# Step 1
run([PYTHON, "-u", "run_pipeline_time.py",
     "--new-ts-batch", "--cls-top-n", "5", "--reg-top-n", "5",
     "--fast", "--viz", "--result-file", "pipeline_ts_results.csv"],
    "Step 1: run_pipeline_time.py")

# Step 2
run([PYTHON, "-u", "run_baseline.py",
     "--new-ts-batch", "--cls-top-n", "5", "--reg-top-n", "5",
     "--viz", "--time-budget", "300", "--result-file", "baseline_ts_results.csv"],
    "Step 2: run_baseline.py")

# Step 3: merge
print(f"\n{'='*65}")
print("  Step 3: 合併 CSV")
print(f"{'='*65}")
dfs = []
for f in ["pipeline_ts_results.csv", "baseline_ts_results.csv"]:
    path = os.path.join(HERE, f)
    if os.path.exists(path):
        dfs.append(pd.read_csv(path))
    else:
        print(f"[警告] 找不到 {f}")

if dfs:
    out = pd.concat(dfs, ignore_index=True)
    out_path = os.path.join(HERE, "combined_ts_results.csv")
    out.to_csv(out_path, index=False)
    print(f"\n合併完成：{len(out)} 筆 → combined_ts_results.csv")
    print(out[["source", "dataset", "task", "score", "elapsed_s"]].to_string(index=False))
else:
    print("[錯誤] 找不到任何結果檔案")
    sys.exit(1)
