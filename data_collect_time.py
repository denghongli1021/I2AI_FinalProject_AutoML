import os
import pandas as pd
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from aeon.datasets import load_classification, load_regression
from aeon.datasets.tsc_datasets import univariate as univariate_classification_datasets
from aeon.datasets.tser_datasets import tser_soton_clean as tser_datasets

warnings.filterwarnings("ignore", category=FutureWarning)

MAX_WORKERS = 6

def download_task(name, load_func, prefix, output_dir):
    """分別下載並儲存 TRAIN / TEST 兩個 split"""
    train_path = os.path.join(output_dir, f"{prefix}_{name}_TRAIN.csv")
    test_path  = os.path.join(output_dir, f"{prefix}_{name}_TEST.csv")

    # 兩個都存在才算完成，跳過
    if os.path.exists(train_path) and os.path.exists(test_path):
        return True, name

    try:
        # ✅ 關鍵修正：明確指定 split
        X_train, y_train = load_func(name, split="train")
        X_test,  y_test  = load_func(name, split="test")

        for X, y, path in [
            (X_train, y_train, train_path),
            (X_test,  y_test,  test_path),
        ]:
            X_2d = X.reshape(X.shape[0], -1)
            df = pd.DataFrame(X_2d)
            df["target"] = y
            df.to_csv(path, index=False)

        return True, name

    except Exception as e:
        return False, name


def smart_download_80(target_cls=40, target_reg=40, output_dir="ucr_ts_80_new(時序資料)"):
    os.makedirs(output_dir, exist_ok=True)

    configs = [
        {"prefix": "CLS", "func": load_classification, "registry": univariate_classification_datasets, "target": target_cls},
        {"prefix": "REG", "func": load_regression,     "registry": tser_datasets,                      "target": target_reg},
    ]

    for cfg in configs:
        print(f"\n--- 正在準備 {cfg['prefix']} 類型資料集清單 ---")
        candidates = list(cfg["registry"])

        # 計算已完成數：train + test 都存在才算 1 個
        existing_done = sum(
            1 for name in candidates
            if os.path.exists(os.path.join(output_dir, f"{cfg['prefix']}_{name}_TRAIN.csv"))
            and os.path.exists(os.path.join(output_dir, f"{cfg['prefix']}_{name}_TEST.csv"))
        )
        print(f"已完成 {existing_done} 個，目標 {cfg['target']} 個，候選共 {len(candidates)} 個")

        remaining = [
            name for name in candidates
            if not (
                os.path.exists(os.path.join(output_dir, f"{cfg['prefix']}_{name}_TRAIN.csv"))
                and os.path.exists(os.path.join(output_dir, f"{cfg['prefix']}_{name}_TEST.csv"))
            )
        ]

        still_needed = cfg["target"] - existing_done
        if still_needed <= 0:
            print("已達目標，跳過。")
            continue

        success_count = existing_done
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_task, name, cfg["func"], cfg["prefix"], output_dir): name
                for name in remaining[:still_needed + 20]  # 多送幾個以防部分失敗
            }

            pbar = tqdm(total=cfg["target"], desc=f"{cfg['prefix']} 下載進度")
            pbar.update(existing_done)

            for future in as_completed(futures):
                success, name = future.result()
                if success and success_count < cfg["target"]:
                    success_count += 1
                    pbar.update(1)
                if success_count >= cfg["target"]:
                    break
            pbar.close()


if __name__ == "__main__":
    smart_download_80(target_cls=40, target_reg=40)
    print("\n[完成] 80 個精選資料集已就緒！")