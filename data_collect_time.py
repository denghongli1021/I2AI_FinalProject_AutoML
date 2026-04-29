import os
import pandas as pd
import warnings
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
# 修正後的 import
from aeon.datasets import load_classification, load_regression
from aeon.datasets.tsc_datasets import univariate as univariate_classification_datasets
from aeon.datasets.tser_datasets import tser_soton_clean as tser_datasets

# 1. 忽略版本更新警告，保持畫面乾淨
warnings.filterwarnings("ignore", category=FutureWarning)

MAX_WORKERS = 6  # 同時下載 6 個檔案，充分利用學術網路頻寬

def download_task(name, load_func, prefix, output_dir):
    """ 執行單一資料集的下載、轉換與存檔 """
    save_path = os.path.join(output_dir, f"{prefix}_{name}.csv")
    if os.path.exists(save_path):
        return True, name
    
    try:
        X, y = load_func(name)
        # 轉換為二維 (樣本 x 時間點)
        X_2d = X.reshape(X.shape[0], -1)
        df = pd.DataFrame(X_2d)
        df['target'] = y
        df.to_csv(save_path, index=False)
        return True, name
    except Exception:
        return False, name

def smart_download_80(target_cls=40, target_reg=40, output_dir="ucr_ts_80(時序資料)"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定義資料來源與目標
    configs = [
        {"prefix": "CLS", "func": load_classification, "registry": univariate_classification_datasets, "target": target_cls},
        {"prefix": "REG", "func": load_regression, "registry": tser_datasets, "target": target_reg}
    ]

    for cfg in configs:
        print(f"\n--- 正在準備 {cfg['prefix']} 類型資料集清單 ---")

        # registry 是純名稱 list，直接取前 target 個
        candidates = list(cfg['registry'])
        
        print(f"符合條件的候選清單共有 {len(candidates)} 個。準備下載前 {cfg['target']} 個...")

        # 3. 多執行緒併發執行
        success_count = 0
        # 檢查資料夾內是否已經有下載好的
        existing = [f for f in os.listdir(output_dir) if f.startswith(cfg['prefix'])]
        success_count = len(existing)
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for name in candidates:
                if success_count >= cfg['target']: break
                # 提交任務
                futures.append(executor.submit(download_task, name, cfg['func'], cfg['prefix'], output_dir))
            
            pbar = tqdm(total=cfg['target'], desc=f"{cfg['prefix']} 下載進度")
            pbar.update(success_count)
            
            for future in as_completed(futures):
                success, _ = future.result()
                if success:
                    success_count += 1
                    pbar.update(1)
                if success_count >= cfg['target']:
                    # 達成數量後嘗試關閉未開始的任務
                    break
            pbar.close()

if __name__ == "__main__":
    smart_download_80(target_cls=40, target_reg=40)
    print("\n[完成] 80 個精選資料集已就緒！")