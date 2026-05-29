import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from visualizer import AutoMLVisualizer

def test_hr_attrition():
    print("\n--- 測試 A：IBM HR 員工離職 (分類任務) ---")
    # 1. 讀取本地 CSV
    df = pd.read_csv('dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
    
    # 簡單預處理：把目標轉為 0/1，並只挑選數值欄位（方便測試）
    df['Target'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    X = df.select_dtypes(include=['int64', 'float64']).drop(['Target'], axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. 訓練
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model.fit(X_train, y_train)
    
    # 3. 視覺化 (使用你的通用類別)
    viz = AutoMLVisualizer(model, X_test, output_dir="test_hr_results")
    viz.generate_all_plots(sample_index=10, target_feature='MonthlyIncome', prefix="hr")

def test_house_prices():
    print("\n--- 測試 B：House Prices 房價預測 (回歸任務) ---")
    # 1. 讀取本地 CSV
    df = pd.read_csv('dataset/House-Prices.csv')
    
    # 簡單預處理：只挑數值欄位，並去掉遺失值
    X = df.select_dtypes(include=['int64', 'float64']).dropna().drop(['SalePrice', 'Id'], axis=1)
    y = df.loc[X.index, 'SalePrice']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. 訓練 (注意：回歸要用 XGBRegressor)
    model = xgb.XGBRegressor(n_estimators=50, max_depth=3)
    model.fit(X_train, y_train)
    
    # 3. 視覺化
    viz = AutoMLVisualizer(model, X_test, output_dir="test_house_results")
    viz.generate_all_plots(sample_index=5, target_feature='GrLivArea', prefix="house")

if __name__ == "__main__":
    # 可以一次跑一個，或是兩個都跑
    try:
        test_hr_attrition()
    except Exception as e:
        print(f"HR 測試失敗，請檢查 CSV 檔案路徑: {e}")
        
    try:
        test_house_prices()
    except Exception as e:
        print(f"房價測試失敗，請檢查 CSV 檔案路徑: {e}")