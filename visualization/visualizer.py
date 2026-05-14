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
        self.X_test = X_test
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        print("Initializing SHAP Explainer...")
        # 自動偵測模型類型，選擇最適合的 Explainer：
        # - 樹模型（XGBoost, LightGBM）→ TreeExplainer（精確解，速度快）
        # - 神經網路（TCN, LSTM 等）   → 自動 fallback 到 PermutationExplainer
        # - 其他模型                   → KernelExplainer
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer(self.X_test)
            print("  Using TreeExplainer (tree-based model detected)")
        except Exception:
            print("  TreeExplainer not applicable, falling back to shap.Explainer...")
            # 對神經網路，shap.Explainer 需要傳入 predict 函數（而非模型物件本身）
            # 取 X_test 前 100 筆作為 background（太多會很慢）
            background = self.X_test.iloc[:100]
            predict_fn = self.model.predict if hasattr(self.model, "predict") else self.model
            self.explainer = shap.Explainer(predict_fn, background)
            self.shap_values = self.explainer(self.X_test)
            print("  Using shap.Explainer (model-agnostic)")

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
        p = f"{prefix}_" if prefix else ""
        print(f"開始生成 {prefix if prefix else '預設'} 專案圖表...")
        self.generate_beeswarm_plot(filename=f"{p}global.png")
        self.generate_waterfall_plot(sample_index=sample_index, filename=f"{p}waterfall.png")
        if target_feature is None:
            target_feature = self.X_test.columns[0]
        self.generate_dependence_plot(target_feature=target_feature, filename=f"{p}dependence.png")