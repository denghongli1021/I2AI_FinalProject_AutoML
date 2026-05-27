"""config: SEED, DEVICE, paths. torch is optional (skip-dl mode)."""
import os

SEED = 42
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    torch = None  # type: ignore
    DEVICE = "cpu"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ARTIFACTS_DIR = os.path.join(ROOT, "artifacts")
SUBMISSIONS_DIR = os.path.join(ROOT, "submissions")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
