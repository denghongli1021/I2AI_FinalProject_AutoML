"""
run_pipeline.py — 完整 Pipeline 執行入口（v2 架構）

流程：
  1. 載入 test/train.csv + test/test.csv，標籤編碼
  2. Tabular HPO（LGBM / XGB / CatBoost / RF / LogReg / SVM）
  3. MLP NAS（One-Shot Supernet + 演化搜尋最佳架構）
  4. MLP 訓練超參數 HPO
  5. CNN1D HPO（架構 + 訓練參數 + feature_set）
  6. Transformer HPO
  7. 所有 top-k config 執行 5-Fold CV → OOF + Test 預測（存 artifacts/）
  8. Ensemble A：Nelder-Mead Weighted Blending → sub_A_blend.csv
  9. Ensemble B：Meta-Learner Stacking → sub_B_stack.csv

執行方式（從專案根目錄）：
    python run_pipeline.py [--fast]

--fast 旗標大幅縮減 HPO/NAS 次數，適合快速驗證流程。
所有超參數由 HPO/NAS 自動決定，程式碼中不人為固定任何訓練數值。
"""
import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from src.config import DEVICE, SEED, ARTIFACTS_DIR, SUBMISSIONS_DIR
from src.hpo import TabularHPO, DLHPO, MLPTrainHPO
from src.nas import MLPNASSearcher
from src.train import run_cv
from src.ensemble import NelderMeadBlender, MetaLearnerStacker
from src.make_submission import generate_submission

TEST_DIR = os.path.join(HERE, "test")
TRAIN_CSV = os.path.join(TEST_DIR, "train.csv")
TEST_CSV = os.path.join(TEST_DIR, "test.csv")
TARGET_COL = "target_feature"
ID_COL = "id"


# ── 設定（--fast 模式大幅縮減試驗次數）────────────────────────────────────────

def get_cfg(fast: bool, n_samples: int = 10_000) -> dict:
    if fast:
        return {
            "tabular_trials": 5,
            "tabular_top_k": 1,
            "nas_epochs": 5,
            "nas_candidates": 5,
            "nas_rounds": 2,
            "mlp_train_trials": 5,
            "mlp_top_k": 1,
            "dl_trials": 5,
            "dl_top_k": 1,
            "meta_trials": 5,
            "blend_restarts": 1,
        }

    if n_samples < 500:
        # 小數據：縮減避免 NAS / Stacking 三重過擬合
        return {
            "tabular_trials": 30,
            "tabular_top_k": 1,
            "nas_epochs": 10,
            "nas_candidates": 10,
            "nas_rounds": 3,
            "mlp_train_trials": 10,
            "mlp_top_k": 1,
            "dl_trials": 10,
            "dl_top_k": 1,
            "meta_trials": 15,
            "blend_restarts": 2,
        }

    if n_samples < 50_000:
        # 中型數據：原始設定，設計吻合此區間
        return {
            "tabular_trials": 50,
            "tabular_top_k": 3,
            "nas_epochs": 30,
            "nas_candidates": 30,
            "nas_rounds": 5,
            "mlp_train_trials": 30,
            "mlp_top_k": 2,
            "dl_trials": 30,
            "dl_top_k": 2,
            "meta_trials": 30,
            "blend_restarts": 3,
        }

    # 大數據（>= 50k）：縮減 trials，NAS 與 DL 降頻避免 OOM / 極慢
    return {
        "tabular_trials": 20,
        "tabular_top_k": 2,
        "nas_epochs": 10,
        "nas_candidates": 10,
        "nas_rounds": 3,
        "mlp_train_trials": 15,
        "mlp_top_k": 1,
        "dl_trials": 15,
        "dl_top_k": 1,
        "meta_trials": 20,
        "blend_restarts": 2,
    }


# ── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="縮減 HPO/NAS 次數，快速驗證")
    parser.add_argument(
        "--skip-tabular", action="store_true", help="跳過傳統模型 HPO（使用 artifacts 快取）"
    )
    parser.add_argument(
        "--skip-dl", action="store_true", help="跳過深度學習模型（使用 artifacts 快取）"
    )
    parser.add_argument("--no-nas", action="store_true", help="跳過 NAS，MLP 使用預設架構")
    args = parser.parse_args()

    t_total = time.time()
    print(f"\n{'='*60}")
    print(f"  Pipeline v2  |  device={DEVICE}  |  fast={args.fast}")
    print(f"{'='*60}")

    # ── 1. 載入資料 ──────────────────────────────────────────────────────────
    print("\n[1/9] 載入資料 ...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    print(f"  train: {train_df.shape}  |  test: {test_df.shape}")

    cfg = get_cfg(args.fast, n_samples=len(train_df))
    scale = "small" if len(train_df) < 500 else ("large" if len(train_df) >= 50_000 else "medium")
    print(f"  data_scale={scale}  |  cfg={cfg}")

    if TARGET_COL not in train_df.columns:
        raise ValueError(f"train.csv 缺少目標欄 '{TARGET_COL}'")

    test_ids = test_df[ID_COL].values
    X_test = test_df.drop(columns=[ID_COL]).values

    le = LabelEncoder()
    y = le.fit_transform(train_df[TARGET_COL].values)
    X = train_df.drop(columns=[TARGET_COL]).values
    n_classes = len(le.classes_)
    print(f"  n_classes={n_classes}  |  n_train={len(y)}  |  n_test={len(X_test)}")
    print(f"  n_features={X.shape[1]}  |  label_map={dict(zip(range(n_classes), le.classes_))}")

    all_oof: list = []
    all_test: list = []
    model_tags: list = []

    # ── 2. Tabular HPO ───────────────────────────────────────────────────────
    if not args.skip_tabular:
        print(f"\n[2/9] Tabular HPO ({cfg['tabular_trials']} trials/model) ...")
        tabular_hpo = TabularHPO(
            model_names=["lgbm", "xgb", "catboost", "rf", "logreg", "svm"],
            n_trials=cfg["tabular_trials"],
            top_k=cfg["tabular_top_k"],
        )
        tabular_configs = tabular_hpo.run(X, y)
    else:
        print("\n[2/9] 跳過 Tabular HPO（--skip-tabular）")
        tabular_configs = []

    # ── 3. MLP NAS ───────────────────────────────────────────────────────────
    if not args.skip_dl:
        if not args.no_nas:
            print(f"\n[3/9] MLP NAS (epochs={cfg['nas_epochs']}, "
                  f"candidates={cfg['nas_candidates']}) ...")
            nas = MLPNASSearcher(
                n_supernet_epochs=cfg["nas_epochs"],
                n_candidates=cfg["nas_candidates"],
                n_evolution_rounds=cfg["nas_rounds"],
                device=DEVICE,
            )
            mlp_arch = nas.search(X, y, n_classes)
        else:
            print("\n[3/9] 跳過 NAS，使用預設 MLP 架構")
            mlp_arch = {
                "depth": 3,
                "hidden_dim": 256,
                "activations": ["gelu", "gelu", "gelu"],
                "use_skips": [True, True, True],
                "dropout": 0.2,
            }

        # ── 4. MLP 訓練參數 HPO ────────────────────────────────────────────
        print(f"\n[4/9] MLP 訓練參數 HPO ({cfg['mlp_train_trials']} trials) ...")
        mlp_hpo = MLPTrainHPO(
            arch_params=mlp_arch,
            n_trials=cfg["mlp_train_trials"],
            top_k=cfg["mlp_top_k"],
            device=DEVICE,
        )
        mlp_configs = mlp_hpo.run(X, y, n_classes)

        # ── 5. CNN1D HPO ────────────────────────────────────────────────────
        print(f"\n[5/9] CNN1D HPO ({cfg['dl_trials']} trials) ...")
        cnn_hpo = DLHPO(
            model_name="cnn1d",
            n_trials=cfg["dl_trials"],
            top_k=cfg["dl_top_k"],
            n_classes=n_classes,
            device=DEVICE,
        )
        cnn_configs = cnn_hpo.run(X, y)

        # ── 6. Transformer HPO ─────────────────────────────────────────────
        print(f"\n[6/9] Transformer HPO ({cfg['dl_trials']} trials) ...")
        tf_hpo = DLHPO(
            model_name="transformer",
            n_trials=cfg["dl_trials"],
            top_k=cfg["dl_top_k"],
            n_classes=n_classes,
            device=DEVICE,
        )
        tf_configs = tf_hpo.run(X, y)

        dl_configs = mlp_configs + cnn_configs + tf_configs
    else:
        print("\n[3-6/9] 跳過深度學習模型（--skip-dl）")
        dl_configs = []

    # ── 7. 5-Fold CV → OOF + Test Predictions ────────────────────────────
    print("\n[7/9] 5-Fold CV 訓練所有 top-k 模型 ...")
    all_configs = tabular_configs + dl_configs

    if len(all_configs) == 0:
        raise RuntimeError("沒有任何 config！請確認 HPO 成功完成或關閉 --skip 旗標。")

    for i, config in enumerate(all_configs):
        tag = f"{config['model_name']}_{config['feature_set']}_c{i}"
        # 避免重複 tag（同模型+feature_set 可能有多個 top-k config）
        tag = tag.replace("/", "_")

        oof_path = os.path.join(ARTIFACTS_DIR, f"{tag}_oof.npy")
        test_path = os.path.join(ARTIFACTS_DIR, f"{tag}_test.npy")

        if os.path.exists(oof_path) and os.path.exists(test_path):
            print(f"  [CV] 載入快取 {tag}")
            oof = np.load(oof_path)
            test_pred = np.load(test_path)
        else:
            oof, test_pred = run_cv(
                config, X, y, X_test, n_classes, device=DEVICE, tag=tag
            )

        all_oof.append(oof)
        all_test.append(test_pred)
        model_tags.append(tag)

    # ── 8. Ensemble A：Nelder-Mead Weighted Blending ──────────────────────
    print("\n[8/9] Ensemble A — Nelder-Mead Weighted Blending ...")
    blender = NelderMeadBlender(n_restarts=cfg["blend_restarts"])
    blender.fit(all_oof, y)
    test_blend = blender.predict(all_test)
    out_A = generate_submission(test_blend, test_ids, le, out_name="sub_A_blend.csv")

    # ── 9. Ensemble B：Meta-Learner Stacking ──────────────────────────────
    print("\n[9/9] Ensemble B — Meta-Learner Stacking ...")
    stacker = MetaLearnerStacker(n_meta_trials=cfg["meta_trials"])
    stacker.fit(all_oof, y)
    test_stack = stacker.predict(all_test)
    out_B = generate_submission(test_stack, test_ids, le, out_name="sub_B_stack.csv")

    # ── 總結 ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  完成！總耗時 {elapsed:.1f}s")
    print(f"  Ensemble A（Blend）→ {out_A}")
    print(f"  Ensemble B（Stack）→ {out_B}")
    print(f"  模型清單（共 {len(model_tags)} 個）:")
    for tag in model_tags:
        print(f"    {tag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
