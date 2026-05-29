# preprocessing/processors/feature_generator.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import QuantileTransformer

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import numpy as np
import scipy.sparse as sp

class MIFeatureSelector(BaseEstimator, TransformerMixin):
    """
    將 Mutual Information 特徵選擇包裝成 Pipeline 元件。
    確保訓練時篩選的特徵，測試時能精準對齊。
    """
    def __init__(self, threshold: float = 0.01, top_k: int = 300, is_classification: bool = True):
        self.threshold = threshold
        self.top_k = top_k
        self.is_classification = is_classification
        self.selected_mask_ = None # 用來記憶保留了哪些特徵

    # 🚀 補上這一段：讓它能跟 Pipeline 溝通特徵名稱
    def get_feature_names_out(self, input_features=None):
        """
        Scikit-Learn Pipeline 會自動把上一層 (Assembler) 的 1280 個特徵名稱傳進來 (input_features)。
        我們只要用 mask 篩選出保留的那 300 個名稱回傳即可！
        """
        if input_features is None:
            raise ValueError("無法取得輸入的特徵名稱。")
        
        # 使用我們在 fit 階段記錄下來的 selected_mask_ 來過濾名稱
        return np.array(input_features)[self.selected_mask_]

    def fit(self, X, y):
        print(f"  [MI Selector] 正在計算特徵重要性 (目標保留 top_{self.top_k})...")
        X_arr = X.toarray() if sp.issparse(X) else np.asarray(X)
        
        # 依據任務類型選擇 MI 演算法
        mi_fn = mutual_info_classif if self.is_classification else mutual_info_regression
        scores = mi_fn(X_arr, y, random_state=42)
        
        # 決定要保留的特徵
        if self.top_k is not None and self.top_k < len(scores):
            # 取分數最高的 top_k 個
            top_k_idx = np.argsort(scores)[-self.top_k:]
            self.selected_mask_ = np.zeros(len(scores), dtype=bool)
            self.selected_mask_[top_k_idx] = True
        else:
            # 依據 threshold 篩選
            self.selected_mask_ = scores >= self.threshold
            
        n_kept = self.selected_mask_.sum()
        n_dropped = len(self.selected_mask_) - n_kept
<<<<<<< HEAD
        print(f"  [MI Selector] 保留 {n_kept} 個特徵，過濾 {n_dropped} 個冗餘特徵。")
=======
        print(f"  ✅ [MI Selector] 保留 {n_kept} 個特徵，過濾 {n_dropped} 個冗餘特徵！")
>>>>>>> 9007facba9f5bf1a65643f4f95b4d6d67742c91e
        
        return self

    def transform(self, X):
        # 套用訓練時記憶的 mask，過濾掉不重要的特徵
        X_arr = X.toarray() if sp.issparse(X) else np.asarray(X)
        return X_arr[:, self.selected_mask_]

# =====================================================================
# 🛡️ 新增：Phase 0 防禦盾牌 (向模型組借鏡的極限防呆機制)
# =====================================================================
class RobustDataCleaner(BaseEstimator, TransformerMixin):
    """
    [Phase 0 防禦盾牌：強健資料清洗器]
    在進入複雜的特徵工程前，自動過濾與修正髒資料。
    
    1. 自動拆解 datetime 欄位 (防止時間字串把記憶體撐爆)
    2. 強制轉型混雜字串的數值欄位 (把 "1,000" 或 "N/A" 強制逼回 np.nan)
    3. 自動剔除毫無預測力的常數欄位 (Zero-variance)
    
    ※ 堅守底線：移除模型組「盲目 Label Encode 未知字串」的毒藥邏輯，
       保留純字串交給後續 LightGBM 的 Category 處理。
    """
    def __init__(self):
        self.datetime_cols_ = []
        self.numeric_coerce_cols_ = []
        self.drop_cols_ = []
        self._out_columns = []

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        
        # 1. 偵測常數欄位 (Zero-variance)
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                self.drop_cols_.append(col)

        # 2. 偵測 Object/String 欄位並進行智慧分類
        for col in df.columns:
            if col in self.drop_cols_:
                continue
                
            if df[col].dtype == object or str(df[col].dtype) == 'string':
                # 嘗試 datetime 解析 (成功率 >= 50% 視為時間欄位)
                parsed_dt = pd.to_datetime(df[col], errors="coerce")
                if parsed_dt.notna().mean() >= 0.5:
                    self.datetime_cols_.append(col)
                    continue
                
                # 嘗試數值強制解析 (成功率 >= 50% 視為被弄髒的數值欄位)
                parsed_num = pd.to_numeric(df[col], errors="coerce")
                if parsed_num.notna().mean() >= 0.5:
                    self.numeric_coerce_cols_.append(col)

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        # 1. 套用 datetime 拆解
        for col in self.datetime_cols_:
            if col in df.columns:
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = parsed.dt.year.astype("float32")
                df[f"{col}_month"] = parsed.dt.month.astype("float32")
                df[f"{col}_dayofweek"] = parsed.dt.dayofweek.astype("float32")
                df[f"{col}_dayofyear"] = parsed.dt.dayofyear.astype("float32")
                df = df.drop(columns=[col]) # 拆解完就銷毀原字串

        # 2. 套用數值強制轉型 (清掉字串雜訊)
        for col in self.numeric_coerce_cols_:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. 剔除常數欄位
        df = df.drop(columns=[c for c in self.drop_cols_ if c in df.columns], errors="ignore")

        # 紀錄輸出欄位名稱供 get_feature_names_out 使用
        self._out_columns = df.columns.tolist()
        
        # 維持 Scikit-learn 慣例，回傳 numpy array
        return df

    def get_feature_names_out(self, input_features=None):
        return np.array(self._out_columns)


# =====================================================================
# ⚙️ 以下為原有 AutoML 2.0 核心煉金模組 (註解與邏輯完全保留)
# =====================================================================

class GroupedAggregater(BaseEstimator, TransformerMixin):
    """
    [群組聚合統計轉換器]
    以指定的高基數類別（如 CardID, UserID）為基準，計算數值特徵的統計量。
    
    ※ 具備推論期防爆機制：若推論期出現未曾見過的全新群組，自動以全域統計量填補，防範 NaN 擴散。
    """
    def __init__(self, group_cols=None, num_cols=None, aggs=["mean", "std"], min_samples=5):
        self.group_cols = group_cols or []
        self.num_cols = num_cols or []
        self.aggs = aggs
        self.min_samples = min_samples
        self.maps_ = {}
        self.global_stats_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        
        # 💡 核心防禦：先學習「全域統計量」，留給推論期未見過的類別當安全墊
        for num_col in self.num_cols:
            self.global_stats_[num_col] = {}
            for agg in self.aggs:
                val = df[num_col].agg(agg) if agg != 'std' else df[num_col].std()
                # 處理極端情況：如果全域算出來是 NaN，強行補 0.0
                self.global_stats_[num_col][agg] = 0.0 if pd.isna(val) else val

        # 學習分組映射表
        for g_col in self.group_cols:
            self.maps_[g_col] = {}
            
            # 🛑 【新增煞車機制】：計算這個群組中，每個 ID 的出現次數
            # 假設你在 __init__ 有設定 self.min_samples = 10 (若無，預設為 10)
            min_samples = getattr(self, 'min_samples', 10)
            counts = df[g_col].value_counts()
            
            # 抓出出現次數大於等於門檻的「活躍 ID」清單
            valid_ids = counts[counts >= min_samples].index
            
            for num_col in self.num_cols:
                # 計算每個群組的統計特徵
                grouped = df.groupby(g_col)[num_col].agg(self.aggs)
                
                # 🛑 【執行煞車】：只保留「活躍 ID」的聚合結果
                # 那些罕見的免洗帳號或新卡片會在這裡被剔除，
                # 推論時就會無縫接軌使用上面算好的 global_stats_！
                filtered_grouped = grouped.loc[grouped.index.isin(valid_ids)]
                
                self.maps_[g_col][num_col] = filtered_grouped
                
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        out_dfs = []

        # 走查所有分組組合
        for g_col in self.group_cols:
            for num_col in self.num_cols:
                # 利用 Left Join 將訓練期學到的統計特徵對應回來
                stat_df = df[[g_col]].merge(self.maps_[g_col][num_col], left_on=g_col, right_index=True, how='left')
                stat_df = stat_df.drop(columns=[g_col])
                
                # 🛡️ 實戰防護：若推論資料包含全新 ID，用訓練期學到的全域平均/標準差頂替
                for agg in self.aggs:
                    fallback_val = self.global_stats_[num_col][agg]
                    stat_df[agg] = stat_df[agg].fillna(fallback_val)
                
                # 強制重新命名欄位，確保 get_feature_names_out 能精準對齊
                stat_df.columns = [f"{g_col}_{num_col}_{agg}" for agg in self.aggs]
                out_dfs.append(stat_df)
                
        return pd.concat(out_dfs, axis=1).values

    def get_feature_names_out(self, input_features=None):
        out_features = []
        for g_col in self.group_cols:
            for num_col in self.num_cols:
                for agg in self.aggs:
                    out_features.append(f"{g_col}_{num_col}_{agg}")
        return np.array(out_features)


class PolynomialInteracter(BaseEstimator, TransformerMixin):
    """
    [多項式交互特徵產生器]
    讓指定的黃金核心特徵進行兩兩交叉相乘與相除，主動擴充模型的非線性視野。
    
    ※ 內建除以零防爆裝甲（Epsilon Smoothing）。
    """
    def __init__(self, target_cols=None, allow_division=True):
        self.target_cols = target_cols or []
        self.allow_division = allow_division

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        out_dfs = []
        n = len(self.target_cols)
        
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = self.target_cols[i], self.target_cols[j]
                
                # ✖️ 相乘特徵
                mul_series = df[c1] * df[c2]
                inter_df = pd.DataFrame({f"{c1}_x_{c2}": mul_series})
                
                # ➗ 相除特徵 (加上 1e-5 預防客戶端資料除以零導致產生 inf)
                if self.allow_division:
                    inter_df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + 1e-5)
                    inter_df[f"{c2}_div_{c1}"] = df[c2] / (df[c1] + 1e-5)
                    
                out_dfs.append(inter_df)
                
        return pd.concat(out_dfs, axis=1).values

    def get_feature_names_out(self, input_features=None):
        out_features = []
        n = len(self.target_cols)
        for i in range(n):
            for j in range(i + 1, n):
                c1, c2 = self.target_cols[i], self.target_cols[j]
                out_features.append(f"{c1}_x_{c2}")
                if self.allow_division:
                    out_features.extend([f"{c1}_div_{c2}", f"{c2}_div_{c1}"])
        return np.array(out_features)


class NonLinearScaler(BaseEstimator, TransformerMixin):
    """
    [非線性縮放與煉金術師]
    專門對付極端偏態、長尾分佈（如交易金額）。
    支援偏態矯正的 Log1p 轉換，或強行將數據揉成完美鐘型曲線的 Quantile Normalization。
    """
    def __init__(self, target_cols=None, strategy='quantile'):
        self.target_cols = target_cols or []
        self.strategy = strategy
        self.transformers_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X)
        if self.strategy == 'quantile':
            for col in self.target_cols:
                # 建立分位數轉換器，強行投影至常態分佈
                tf = QuantileTransformer(n_quantiles=1000, random_state=42, output_distribution='normal')
                tf.fit(df[[col]].fillna(0))
                self.transformers_[col] = tf
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        out_df = pd.DataFrame(index=df.index)
        
        for col in self.target_cols:
            if self.strategy == 'quantile':
                vals = df[[col]].fillna(0).values
                out_df[f"{col}_quantile"] = self.transformers_[col].transform(vals).flatten()
            elif self.strategy == 'log':
                # 🚀 經典長尾處理：log(x + 1)，完美應對內含 0 的長尾數據
                out_df[f"{col}_log1p"] = np.log1p(np.maximum(0, df[col].fillna(0)))
                
        return out_df.values

    def get_feature_names_out(self, input_features=None):
        out_features = []
        for col in self.target_cols:
            if self.strategy == 'quantile':
                out_features.append(f"{col}_quantile")
            elif self.strategy == 'log':
                out_features.append(f"{col}_log1p")
        return np.array(out_features)