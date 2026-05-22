"""全域設定：亂數種子、裝置偵測、路徑管理。"""
import os
import torch

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")
LOG_DIR = os.path.join(ROOT, "LOG")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
