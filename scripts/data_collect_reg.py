import openml
import os
import pandas as pd


def download_regression_datasets(output_dir="openml_regression_data", target_count=40):
    """
    從 OpenML 搜尋回歸任務（task_type_id=2）並下載非時間序列的回歸資料集
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立資料夾: {output_dir}")

    # 取得已下載的資料集 ID，支援斷點續跑
    existing_ids = set()
    for f in os.listdir(output_dir):
        if f.endswith(".csv"):
            try:
                existing_ids.add(int(f.split("_")[0]))
            except ValueError:
                pass
    if existing_ids:
        print(f"[斷點續跑] 已有 {len(existing_ids)} 個資料集，跳過重複下載")

    print("正在搜尋 OpenML 回歸任務（task_type_id=2）...")
    try:
        tasks_df = openml.tasks.list_tasks(
            task_type=openml.tasks.TaskType.SUPERVISED_REGRESSION,
            output_format="dataframe",
        )
    except Exception as e:
        print(f"無法搜尋任務: {e}")
        return

    print(f"找到 {len(tasks_df)} 個回歸任務，開始篩選...")

    # 過濾：合理樣本數與特徵數
    tasks_df = tasks_df[
        (tasks_df["NumberOfInstances"] >= 200) &
        (tasks_df["NumberOfInstances"] <= 500_000) &
        (tasks_df["NumberOfFeatures"] >= 2) &
        (tasks_df["NumberOfFeatures"] <= 500)
    ].drop_duplicates(subset="did").sort_values("NumberOfInstances").reset_index(drop=True)

    print(f"篩選後剩 {len(tasks_df)} 個候選，將嘗試下載 {target_count} 個...")

    status_list = []
    success_count = len(existing_ids)
    seen_ids = set(existing_ids)
    time_keywords = {"date", "time", "timestamp"}

    for _, row in tasks_df.iterrows():
        if success_count >= target_count:
            print(f"\n[完成] 已成功下載滿 {target_count} 個回歸資料集！")
            break

        did = int(row["did"])
        if did in seen_ids:
            continue
        seen_ids.add(did)

        try:
            dataset = openml.datasets.get_dataset(did, download_data=False)

            # 排除含時間欄位的資料集
            features = dataset.features
            if any(
                any(kw in f.name.lower() for kw in time_keywords)
                for f in features.values()
            ):
                continue

            # 正式下載
            dataset = openml.datasets.get_dataset(did, download_data=True)
            X, y, _, _ = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="dataframe",
            )

            # 確認 y 為連續數值且不是偽裝的分類（唯一值 > 20）
            if y is None or not pd.api.types.is_numeric_dtype(y):
                continue
            if y.nunique() <= 20:
                continue

            full_df = pd.concat([X, y], axis=1)
            file_name = f"{did}_{dataset.name.replace(' ', '_')}.csv"
            full_df.to_csv(os.path.join(output_dir, file_name), index=False)

            success_count += 1
            print(
                f" [{success_count}/{target_count}] 成功下載 ID {did}: {dataset.name} "
                f"(樣本數: {full_df.shape[0]}, 特徵數: {full_df.shape[1] - 1})"
            )
            status_list.append({
                "ID": did, "Name": dataset.name,
                "Rows": full_df.shape[0], "Cols": full_df.shape[1], "Status": "Success",
            })

        except Exception as e:
            status_list.append({
                "ID": did, "Name": "Unknown", "Rows": 0, "Cols": 0,
                "Status": f"Failed: {str(e)[:80]}",
            })

    report_df = pd.DataFrame(status_list)
    report_df.to_csv("regression_download_report.csv", index=False)
    print(f"\n任務結束。本次新增 {len([r for r in status_list if r['Status']=='Success'])} 個，"
          f"詳細紀錄請見 regression_download_report.csv")


if __name__ == "__main__":
    download_regression_datasets(output_dir="openml_regression_data", target_count=40)
