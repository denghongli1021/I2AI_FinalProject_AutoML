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
        
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        import time
        print("Initializing SHAP Explainer...")
        _t0 = time.time()
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer(self.X_test)
            _elapsed = time.time() - _t0
            print(f"  Using TreeExplainer (tree-based model detected)")
            print(f"  SHAP computation time: {_elapsed:.2f} seconds ({len(self.X_test)} samples)")
        except Exception:
            print("  TreeExplainer not applicable, falling back...")

            background_size = min(100, len(self.X_test))
            background = self.X_test.sample(n=background_size, random_state=42)
            try:
                background = background.astype(np.float64)
            except Exception:
                background = background.apply(pd.to_numeric, errors='coerce').fillna(0.0)

            if hasattr(self.model, "predict_proba"):
                predict_fn = self.model.predict_proba
            elif hasattr(self.model, "predict"):
                predict_fn = self.model.predict
            else:
                predict_fn = self.model

            _max_samples = 500
            if len(self.X_test) > _max_samples:
                print(f"  [SHAP] 取樣 {_max_samples}/{len(self.X_test)} 筆...")
                self.X_test = self.X_test.sample(n=_max_samples, random_state=42).reset_index(drop=True)
            try:
                self.X_test = self.X_test.astype(np.float64)
            except Exception:
                self.X_test = self.X_test.apply(pd.to_numeric, errors='coerce').fillna(0.0)

            _n_feat = self.X_test.shape[1]
            _max_evals = 2 * _n_feat + 1

            # PermutationExplainer.__init__ 內部會呼叫 predict_fn 計算 expected_value，
            # 可能在此觸發 numba JIT 錯誤 → 將整個創建 + 計算放入 try，失敗一律 fallback KernelExplainer
            try:
                print(f"  [SHAP] 嘗試 PermutationExplainer, max_evals={_max_evals}")
                self.explainer = shap.PermutationExplainer(predict_fn, background)
                self.shap_values = self.explainer(self.X_test, max_evals=_max_evals)
            except Exception as _je:
                print(f"  [SHAP] PermutationExplainer 失敗 ({type(_je).__name__})，改用 KernelExplainer...")
                _n_bg = min(50, len(background))
                _bg_np = background.values[:_n_bg].astype(np.float64)
                self.explainer = shap.KernelExplainer(predict_fn, _bg_np)
                _x_np = self.X_test.values[:min(50, len(self.X_test))].astype(np.float64)
                _sv = self.explainer.shap_values(_x_np, nsamples=100, silent=True)
                _x_sub = self.X_test.iloc[:min(50, len(self.X_test))].reset_index(drop=True)
                _sv_arr = np.stack(_sv, axis=-1) if isinstance(_sv, list) else np.array(_sv)
                _base = self.explainer.expected_value
                if isinstance(_base, (list, np.ndarray)):
                    _base = np.array(_base)
                self.shap_values = shap.Explanation(
                    values=_sv_arr,
                    base_values=_base,
                    data=_x_sub.values,
                    feature_names=list(_x_sub.columns),
                )
                self.X_test = _x_sub

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
        vals = np.abs(self._get_shap_matrix()).mean(0)
        df = pd.DataFrame({'Feature': self.X_test.columns, 'Importance': vals})
        df = df.sort_values(by='Importance', ascending=True).tail(15)

        fig = px.bar(
            df, x='Importance', y='Feature', orientation='h',
            title='Global Explanation: Feature Importance',
            color='Importance', template='plotly_white',
            color_continuous_scale='Blues'
        )
        fig.update_layout(margin=dict(l=150, r=50, t=80, b=50), title_font_size=20)

        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_image(filepath, scale=2)
            print(f"Generated: {filepath}")
        return fig

    def generate_waterfall_plot(self, sample_index=0, filename="waterfall.png"):
        sample_shap = self._get_shap_matrix()[sample_index]
        base_value = self.shap_values[sample_index].base_values

        if isinstance(base_value, (np.ndarray, list)):
            base_value = np.array(base_value).flatten()[-1]

        df = pd.DataFrame({'Feature': self.X_test.columns, 'Contribution': sample_shap})
        df['AbsContribution'] = df['Contribution'].abs()
        df = df.sort_values(by='AbsContribution', ascending=False).head(8)

        def fmt(v, sign=False):
            decimals = 4 if abs(v) < 0.01 else 2
            fmt_str = f"{{:+.{decimals}f}}" if sign else f"{{:.{decimals}f}}"
            return fmt_str.format(v)

        prediction = base_value + sum(sample_shap)
        text_vals = [fmt(base_value)] + [fmt(x, sign=True) for x in df['Contribution']] + [fmt(prediction)]

        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * len(df) + ["total"],
            x=["Base Value"] + list(df['Feature']) + ["Prediction"],
            y=[base_value] + list(df['Contribution']) + [base_value + sum(sample_shap)],
            text=text_vals,
            textposition="outside",
            decreasing={"marker": {"color": "#337ab7"}},
            increasing={"marker": {"color": "#d9534f"}},
            totals={"marker": {"color": "#5cb85c"}},
        ))
        fig.update_layout(
            title=f"Local Explanation: Waterfall (Sample #{sample_index})",
            template='plotly_white',
            margin=dict(l=80, r=50, t=80, b=120),
            title_font_size=20,
        )
        fig.update_xaxes(tickangle=45)

        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_image(filepath, scale=2)
            print(f"Generated: {filepath}")
        return fig

    def generate_dependence_plot(self, target_feature, filename="c_dependence.png"):
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
            color_feature: self.X_test[color_feature],
        })

        fig = px.scatter(
            df, x=target_feature, y='SHAP Value',
            color=color_feature,
            title=f"Interaction Analysis: {target_feature}",
            template='plotly_white',
            color_continuous_scale='RdYlBu',
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(margin=dict(l=80, r=50, t=80, b=50), title_font_size=20)

        if self.output_dir:
            filepath = os.path.join(self.output_dir, filename)
            fig.write_image(filepath, scale=2)
            print(f"Generated: {filepath}")
        return fig

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
