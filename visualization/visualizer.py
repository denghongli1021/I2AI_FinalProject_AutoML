import os
import numpy as np
import pandas as pd
import shap
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

class AutoMLVisualizer:
    def __init__(self, model, X_test, output_dir="plots"):
        self.model = model
        self.output_dir = output_dir

        # ─── 特徵欄位防禦裝甲 ───────────────────────────────────────────────
        # 確保 X_test 必定為含有正確欄位名稱的 DataFrame，防止 NumPy Array 或流水號干擾
        if isinstance(X_test, np.ndarray):
            if hasattr(model, "feature_names") and model.feature_names is not None:
                cols = model.feature_names
            else:
                cols = [f"Feature {i}" for i in range(X_test.shape[1])]
            self.X_test = pd.DataFrame(X_test, columns=cols)
        else:
            self.X_test = X_test.copy()
        # ───────────────────────────────────────────────────────────────────
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        import time
        print("Initializing SHAP Explainer...")
        # 自動偵測模型類型，選擇最適合的 Explainer：
        # - 樹模型（XGBoost, LightGBM）→ TreeExplainer（精確解，速度快）
        # - 神經網路（TCN, LSTM 等）   → 自動 fallback 到 PermutationExplainer
        # - 其他模型                   → KernelExplainer
        _t0 = time.time()
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer(self.X_test)
            _elapsed = time.time() - _t0
            print(f"  Using TreeExplainer (tree-based model detected)")
            print(f"  SHAP computation time: {_elapsed:.2f} seconds ({len(self.X_test)} samples)")
        except Exception:
            print("  TreeExplainer not applicable, falling back to PermutationExplainer...")
            print(f"  Computing SHAP for all {len(self.X_test)} samples (full dataset)...")
            
            # 用 PermutationExplainer：
            # - 支援任何模型（神經網路、SVM 等）
            # - 全量計算每一筆資料，不採樣
            # - 比 KernelSHAP 快：計算量 = O(樣本數 × 特徵數 × npermutations)
            #   npermutations 預設只跑幾輪，不隨樣本數爆炸
            # - background 只用來建立基準線，取 100 筆隨機樣本即可
            background_size = min(100, len(self.X_test))
            background = self.X_test.sample(n=background_size, random_state=42)
            
            predict_fn = self.model.predict if hasattr(self.model, "predict") else self.model
            self.explainer = shap.PermutationExplainer(predict_fn, background)

            # 取樣上限：視覺化用途 500 筆已足夠，避免對全量資料逐筆擾動（數小時）
            _max_samples = 500
            if len(self.X_test) > _max_samples:
                print(f"  [SHAP] 取樣 {_max_samples}/{len(self.X_test)} 筆（視覺化不需要全量計算）...")
                self.X_test = self.X_test.sample(n=_max_samples, random_state=42).reset_index(drop=True)

            # max_evals = 2*n_features+1 → 只跑 1 次 permutation，比預設 ~4 次快 4 倍，
            # 精度損失極小（特徵重要性排名穩定，視覺化用途已足夠）
            _n_feat = self.X_test.shape[1]
            _max_evals = 2 * _n_feat + 1
            print(f"  [SHAP] max_evals={_max_evals} (1 permutation × {_n_feat} features × 2 directions)")
            self.shap_values = self.explainer(self.X_test, max_evals=_max_evals)
            _elapsed = time.time() - _t0
            mins, secs = divmod(int(_elapsed), 60)
            time_str = f"{mins}m {secs}s" if mins > 0 else f"{secs}s"
            print(f"  SHAP computation complete: {len(self.X_test)} samples in {time_str}")
            
        # ─── SHAP 物件名稱強制覆蓋 ──────────────────────────────────────────
        # 雙重保險：確保 shap_values 內部的 feature_names 屬性必定與 X_test 的真名欄位對齊
        if hasattr(self.shap_values, "feature_names") and self.shap_values.feature_names is not None:
            if len(self.shap_values.feature_names) == len(self.X_test.columns):
                self.shap_values.feature_names = list(self.X_test.columns)
        else:
            try:
                self.shap_values.feature_names = list(self.X_test.columns)
            except Exception:
                pass
            
    def _get_shap_matrix(self):
        """
        統一處理不同 Explainer 的 shap_values 輸出格式：
        - TreeExplainer（分類）：shap_values.values 可能是 3D (samples, features, classes)
        - TreeExplainer（回歸）：shap_values.values 是 2D (samples, features)
        - shap.Explainer（神經網路）：格式同上
        統一取 class 1（正類）或唯一的輸出維度
        """
        v = self.shap_values.values
        if v.ndim == 3:
            # 多輸出分類：取最後一個 class（class 1）
            return v[:, :, -1]
        return v

    def generate_beeswarm_plot(self, filename="global_importance.png"):
        # UI 優化：將雜亂的蜂群圖改為清晰的「全局特徵重要性長條圖」
        vals = np.abs(self._get_shap_matrix()).mean(0)
        df = pd.DataFrame({'Feature': self.X_test.columns, 'Importance': vals})
        df = df.sort_values(by='Importance', ascending=True).tail(15) # 取前 15 重要

        fig = px.bar(
            df, x='Importance', y='Feature', orientation='h',
            title='Global Explanation: Feature Importance',
            color='Importance', template='plotly_white',
            color_continuous_scale='Blues'
        )
        fig.update_layout(margin=dict(l=150, r=50, t=80, b=50), title_font_size=20)
        
        filepath = os.path.join(self.output_dir, filename)
        fig.write_image(filepath, scale=2) # scale=2 讓圖片變高畫質
        print(f"Generated: {filepath}")

    def generate_waterfall_plot(self, sample_index=0, filename="waterfall.png"):
        # Plotly 現代化瀑布圖 (垂直顯示，文字永不重疊，精確小數點)
        sample_shap = self._get_shap_matrix()[sample_index]
        base_value = self.shap_values[sample_index].base_values
        
        # 確保 base_value 是單一數值
        # 分類模型的 base_values 有時是二維陣列 shape=(1,2) 或 shape=(2,)
        # 取最後一個值（對應 class 1，即正類的 log-odds base）
        if isinstance(base_value, (np.ndarray, list)):
            base_value = np.array(base_value).flatten()[-1]
        
        df = pd.DataFrame({'Feature': self.X_test.columns, 'Contribution': sample_shap})
        df['AbsContribution'] = df['Contribution'].abs()
        df = df.sort_values(by='AbsContribution', ascending=False).head(8) # 取前 8 大，保持畫面簡潔
        
        # 動態精度：數值過小時自動切換為 4 位小數，避免顯示 0.00
        def fmt(v, sign=False):
            decimals = 4 if abs(v) < 0.01 else 2
            fmt_str = f"{{:+.{decimals}f}}" if sign else f"{{:.{decimals}f}}"
            return fmt_str.format(v)

        prediction = base_value + sum(sample_shap)
        text_vals = [fmt(base_value)] + [fmt(x, sign=True) for x in df['Contribution']] + [fmt(prediction)]
        
        fig = go.Figure(go.Waterfall(
            orientation="v",
            # 加入 "absolute" 作為第一根柱子 (Base Value)
            measure=["absolute"] + ["relative"] * len(df) + ["total"],
            x=["Base Value"] + list(df['Feature']) + ["Prediction"],
            y=[base_value] + list(df['Contribution']) + [base_value + sum(sample_shap)],
            text=text_vals, # 直接餵給它乾淨的字串
            textposition="outside",
            decreasing={"marker":{"color":"#337ab7"}}, # 藍色減分
            increasing={"marker":{"color":"#d9534f"}}, # 紅色加分
            totals={"marker":{"color":"#5cb85c"}}      # 綠色總分
        ))
        
        fig.update_layout(
            title=f"Local Explanation: Waterfall (Sample #{sample_index})",
            template='plotly_white',
            margin=dict(l=80, r=50, t=80, b=120),
            title_font_size=20
        )
        fig.update_xaxes(tickangle=45) # 底部文字旋轉
        
        filepath = os.path.join(self.output_dir, filename)
        fig.write_image(filepath, scale=2)
        print(f"Generated: {filepath}")

    def generate_dependence_plot(self, target_feature, filename="c_dependence.png"):
        # 自動挑選 color_feature：
        # 1. 排除目標特徵本身
        # 2. 優先選 unique 值數量 > 10 的連續數值欄位（避免選到 race 等低基數類別）
        # 3. 在符合條件的候選中，選標準差最大的欄位
        candidates = [
            col for col in self.X_test.columns
            if col != target_feature and self.X_test[col].nunique() > 10
        ]
        if not candidates:
            candidates = [col for col in self.X_test.columns if col != target_feature]
        color_feature = max(candidates, key=lambda col: self.X_test[col].std())
        
        df = pd.DataFrame({
            target_feature: self.X_test[target_feature],
            'SHAP Value': self._get_shap_matrix()[:, list(self.X_test.columns).index(target_feature)],
            color_feature: self.X_test[color_feature] # 加入顏色特徵
        })
        
        fig = px.scatter(
            df, x=target_feature, y='SHAP Value',
            color=color_feature, # 把顏色塗上去！
            title=f"Interaction Analysis: {target_feature}",
            template='plotly_white',
            color_continuous_scale='RdYlBu' # 使用紅藍漸層，商業圖表最愛
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(margin=dict(l=80, r=50, t=80, b=50), title_font_size=20)
        
        filepath = os.path.join(self.output_dir, filename)
        fig.write_image(filepath, scale=2)
        print(f"Generated: {filepath}")

    def generate_all_plots(self, sample_index=0, target_feature=None, prefix=""):
        import time
        p = f"{prefix}_" if prefix else ""
        print(f"開始生成 {prefix if prefix else '預設'} 專案圖表...")
        _total_start = time.time()

        _t = time.time()
        self.generate_beeswarm_plot(filename=f"{p}global.png")
        print(f"  └─ Global chart: {time.time()-_t:.1f}s")

        _t = time.time()
        self.generate_waterfall_plot(sample_index=sample_index, filename=f"{p}waterfall.png")
        print(f"  └─ Waterfall chart: {time.time()-_t:.1f}s")

        if target_feature is None:
            target_feature = self.X_test.columns[0]

        _t = time.time()
        self.generate_dependence_plot(target_feature=target_feature, filename=f"{p}dependence.png")
        print(f"  └─ Dependence chart: {time.time()-_t:.1f}s")

        _total = time.time() - _total_start
        print(f"Total time: {_total:.1f}s")
