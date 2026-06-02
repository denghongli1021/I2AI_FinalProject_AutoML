"""
merge_final_results.py — 合併三份結果 CSV 到 pipeline_batch_results.csv

來源：
  1. pipeline_batch_results.csv       : 新 pipeline 批次（last-10 OpenML+UCR；REG 為 SKIP）
  2. baseline_batch_results.csv       : 新 baseline 批次（last-10）
  3. pipeline_time_batch_results.csv  : 新 pipeline_time 批次（last-10 REG_*）

輸出 pipeline_batch_results.csv，欄位：
  source, dataset, type, task, n_train, n_test, accuracy, f1_macro, rmse, r2, score, elapsed_s

注意：回歸列的 score 統一為 R²
  - pipeline_time：用 r2 欄位覆蓋 score
  - baseline：score 本身即為 R²（AutoGluon 預設）
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NEW_P  = os.path.join(HERE, "pipeline_batch_results.csv")  # NOT prev — about to be overwritten
NEW_B  = os.path.join(HERE, "baseline_batch_results.csv")
NEW_T  = os.path.join(HERE, "pipeline_time_batch_results.csv")
OUT    = os.path.join(HERE, "pipeline_batch_results.csv")

COLS = ["source", "dataset", "type", "task",
        "n_train", "n_test", "accuracy", "f1_macro",
        "rmse", "r2", "score", "elapsed_s", "fast"]


def _read(path, source):
    if not os.path.exists(path):
        print(f"[WARN] 找不到 {path}，跳過 {source}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "source" not in df.columns:
        df.insert(0, "source", source)
    else:
        df["source"] = df["source"].fillna(source)
    return df


def main():
    # 1. 讀新 pipeline（last-10，含 REG SKIP）
    new_p = _read(NEW_P, "pipeline")
    if not new_p.empty:
        # 移除 REG_* 的 SKIP 列（這些由 pipeline_time 取代）
        before = len(new_p)
        new_p = new_p[~((new_p["source"] == "pipeline") & (new_p["dataset"].str.startswith("REG_")))].copy()
        if len(new_p) < before:
            print(f"  [filter] 移除 {before - len(new_p)} 個 REG_* SKIP 列")
    print(f"[OK] new pipeline → {len(new_p)} rows")

    # 2. 讀新 baseline（last-10）
    new_b = _read(NEW_B, "baseline")
    print(f"[OK] new baseline → {len(new_b)} rows")

    # 3. 讀新 pipeline_time（last-10 REG_*）；score 統一為 R²
    new_t = _read(NEW_T, "pipeline_time")
    if not new_t.empty and "r2" in new_t.columns:
        new_t["score"] = new_t["r2"]
    print(f"[OK] new pipeline_time → {len(new_t)} rows")

    frames = [df for df in (new_p, new_b, new_t) if not df.empty]
    if not frames:
        print("[ERROR] 無資料可合併"); return

    merged = pd.concat(frames, ignore_index=True, sort=False)
    for c in COLS:
        if c not in merged.columns:
            merged[c] = None
    merged = merged[COLS]

    # 去重（同 source + dataset 保留最後一筆，因為通常後到的是更新的結果）
    before = len(merged)
    merged = merged.drop_duplicates(subset=["source", "dataset", "fast"], keep="last").reset_index(drop=True)
    if len(merged) < before:
        print(f"  [dedup] 移除 {before - len(merged)} 個重複 (source, dataset) 列")

    merged.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n[完成] {OUT}  ({len(merged)} rows)")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
