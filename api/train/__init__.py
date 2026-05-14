"""api.train package.

兩條對外介面 (都實作在 train.py):
  - `run(df, target, features, algorithms, options)` — 從原始 DataFrame 訓練,自己做特徵建構 / 切分。
  - `run_prepared(X_train_df, X_test_df, y_train, y_test, target, algorithms, options)`
    — 從預處理模組切好的 train/test 直接訓練。
"""

from .train import run, run_prepared

__all__ = ["run", "run_prepared"]
