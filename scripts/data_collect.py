import openml
import os
import pandas as pd
from tqdm import tqdm

def download_cc18_datasets(output_dir="openml_cc18_data"):
    """
    下載 OpenML-CC18 基準測試套件中的所有 72 個資料集
    """
    # 建立儲存目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立資料夾: {output_dir}")

    print("正在獲取 OpenML-CC18 套件清單 (Suite ID: 99)...")
    try:
        # OpenML-CC18 的 Suite ID 通常是 99 或透過名稱獲取
        benchmark_suite = openml.study.get_suite('OpenML-CC18')
    except Exception as e:
        print(f"無法獲取套件資訊: {e}")
        return

    dataset_ids = benchmark_suite.data
    print(f"預計下載資料集數量: {len(dataset_ids)}")

    # 紀錄下載狀態
    status_list = []

    for did in tqdm(dataset_ids, desc="下載進度"):
        try:
            # 獲取資料集物件 (此步驟會檢查本地快取，若無則下載)
            dataset = openml.datasets.get_dataset(did)
            
            # 取得 DataFrame 格式的資料
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe"
            )
            
            # 儲存為 CSV
            file_name = f"{did}_{dataset.name.replace(' ', '_')}.csv"
            save_path = os.path.join(output_dir, file_name)
            
            # 合併特徵與標籤
            full_df = pd.concat([X, y], axis=1)
            full_df.to_csv(save_path, index=False)
            
            status_list.append({"ID": did, "Name": dataset.name, "Status": "Success"})
            
        except Exception as e:
            print(f"\n[錯誤] 資料集 ID {did} 下載失敗: {e}")
            status_list.append({"ID": did, "Name": "Unknown", "Status": f"Failed: {str(e)}"})

    # 輸出簡易報告
    report_df = pd.DataFrame(status_list)
    report_df.to_csv("download_report.csv", index=False)
    print("\n下載完成！詳細結果請見 download_report.csv")

if __name__ == "__main__":
    # 確保已安裝必要套件: pip install openml pandas tqdm
    download_cc18_datasets()