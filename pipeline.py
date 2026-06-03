"""
pipeline.py — 單資料集 Pipeline 引擎（v2）

提供三個公開符號供 run_pipeline.py 使用：
  - TimeBudget : 時間預算管理器
  - get_cfg()  : 依資料量與 fast 旗標產生 HPO 設定
  - run()      : 接收 (X_train, y_train, X_test, n_classes, cfg, budget, ...)
                 回傳 PipelineResult（含雙 ensemble 預測及中間產物）

執行流程：
  [2a] Tabular Scout HPO  →  [2b] Full HPO（+黃金預設值保底）
  [3]  MLP NAS            →  [4]  MLP 訓練 HPO
  [5]  CNN1D / TCN HPO
  [6]  Transformer / PatchTST HPO
  [7]  5-Fold CV → OOF + Test 預測（支援 artifacts 快取）
  [8]  Ensemble A：Nelder-Mead Weighted Blending
  [9]  Ensemble B：Meta-Learner Stacking
"""
import os
import math
import json
import time
import numpy as np

from src.config import DEVICE, ARTIFACTS_DIR
from src.hpo import TabularHPO, DLHPO, MLPTrainHPO, TSNetTrainHPO
from src.nas import MLPNASSearcher, TSNASSearcher
from src.train import run_cv
from src.ensemble import NelderMeadBlender, MetaLearnerStacker

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PRESETS = os.path.join(_HERE, "src", "best_presets.json")

_ALL_TABULAR_MODELS = ["lgbm", "xgb", "catboost", "rf", "logreg", "extra_trees", "knn"]
_DEFAULT_MLP_ARCH = {
    "depth": 3, "hidden_dim": 256,
    "activations": ["gelu", "gelu", "gelu"],
    "use_skips": [True, True, True],
    "dropout": 0.2,
}
_DEFAULT_TSNET_ARCH = {
    "n_blocks": 3,
    "operations": [0, 2, 3],  # conv_k3, tcn_d2, tcn_d4
    "channels": 64,
    "dropout": 0.1,
}


# ── TimeBudget ────────────────────────────────────────────────────────────────

class TimeBudget:
    """全域計時器：追蹤已用時間，支援動態降級。time_limit=0 表示無限制。"""

    def __init__(self, limit_sec: float, t_start: float):
        self.limit = limit_sec
        self.t_start = t_start

    def elapsed(self) -> float:
        return time.time() - self.t_start

    def remaining(self) -> float:
        if self.limit <= 0:
            return float("inf")
        return max(0.0, self.limit - self.elapsed())

    def fraction_remaining(self) -> float:
        if self.limit <= 0:
            return 1.0
        return self.remaining() / self.limit

    def should_skip(self, cost_fraction: float = 0.12) -> bool:
        """若剩餘時間低於總預算的 cost_fraction，建議跳過該階段。"""
        return self.fraction_remaining() < cost_fraction

    def scale_trials(self, n_trials: int, min_trials: int = 2) -> int:
        """依剩餘時間比例縮減 trial 數，確保 >= min_trials。"""
        if self.limit <= 0:
            return n_trials
        return max(min_trials, int(n_trials * self.fraction_remaining()))

    def status_str(self) -> str:
        if self.limit <= 0:
            return f"elapsed={self.elapsed():.0f}s (no limit)"
        r = self.remaining()
        return (f"elapsed={self.elapsed():.0f}s  remaining={r:.0f}s/{self.limit:.0f}s"
                f" ({self.fraction_remaining()*100:.0f}%)")


# ── 超參數設定 ────────────────────────────────────────────────────────────────

def get_cfg(fast: bool, n_samples: int = 10_000) -> dict:
    """依 fast 旗標與資料量回傳 HPO 設定 dict。"""
    if fast:
        return {
            "tabular_trials": 5,  "tabular_top_k": 1,
            "scout_trials": 3,    "scout_val_size": 0.2, "scout_ratio": 2 / 3,
            "nas_epochs": 5,      "nas_candidates": 5,   "nas_rounds": 2,
            "mlp_train_trials": 3,"mlp_top_k": 1,
            "dl_trials": 3,       "dl_top_k": 1,
            "meta_trials": 3,     "blend_restarts": 1,
            "n_repeats": 1,       "n_seeds": 1,
            "use_kpca": False,    "use_kmeans": False,
            "use_resnet18": False,"use_1d_aug": False,
        }
    if n_samples < 500:
        return {
            "tabular_trials": 30, "tabular_top_k": 2,
            "scout_trials": 5,    "scout_val_size": 0.2, "scout_ratio": 2 / 3,
            "nas_epochs": 8,      "nas_candidates": 8,   "nas_rounds": 2,
            "mlp_train_trials": 8,"mlp_top_k": 1,
            "dl_trials": 8,       "dl_top_k": 1,
            "meta_trials": 10,    "blend_restarts": 2,
            "n_repeats": 2,       "n_seeds": 3,
            "use_kpca": True,     "use_kmeans": True,
            "use_resnet18": False,"use_1d_aug": True,
        }
    if n_samples < 50_000:
        return {
            "tabular_trials": 20, "tabular_top_k": 2,   # top_k=2 → 每模型保留兩組超參，增加 ensemble 多樣性
            "scout_trials": 7,    "scout_val_size": 0.2, "scout_ratio": 2 / 3,
            "nas_epochs": 10,     "nas_candidates": 8,   "nas_rounds": 2,
            "mlp_train_trials": 8,"mlp_top_k": 1,
            "dl_trials": 8,       "dl_top_k": 1,
            "meta_trials": 15,    "blend_restarts": 3,   # 更多 meta-learner 試次 + blend 重啟
            "n_repeats": 1,       "n_seeds": 1,
            "use_kpca": True,     "use_kmeans": True,
            "use_resnet18": True, "use_1d_aug": False,
        }
    return {
        "tabular_trials": 10, "tabular_top_k": 1,
        "scout_trials": 5,    "scout_val_size": 0.2, "scout_ratio": 2 / 3,
        "nas_epochs": 8,      "nas_candidates": 6,   "nas_rounds": 2,
        "mlp_train_trials": 6,"mlp_top_k": 1,
        "dl_trials": 6,       "dl_top_k": 1,
        "meta_trials": 8,     "blend_restarts": 1,
        "n_repeats": 1,       "n_seeds": 1,
        "use_kpca": False,    "use_kmeans": False,
        "use_resnet18": True, "use_1d_aug": False,
    }


# ── 結果容器 ──────────────────────────────────────────────────────────────────

class PipelineResult:
    """封裝 run() 的所有輸出，呼叫端按需取用。"""

    def __init__(self, test_blend, test_stack, all_oof, all_test, model_tags, blender, stacker):
        self.test_blend  = test_blend    # np.ndarray — Ensemble A（Nelder-Mead）測試預測
        self.test_stack  = test_stack    # np.ndarray — Ensemble B（Meta-Learner）測試預測
        self.all_oof     = all_oof       # list[np.ndarray] — 各模型 OOF 預測
        self.all_test    = all_test      # list[np.ndarray] — 各模型測試預測
        self.model_tags  = model_tags    # list[str]
        self.blender     = blender       # NelderMeadBlender（已 fit）
        self.stacker     = stacker       # MetaLearnerStacker（已 fit）


# ── 主引擎 ────────────────────────────────────────────────────────────────────

def run(
    X_train,  # 🚀 這裡的型別現在可以是 np.ndarray 也可以是 dict
    y_train: np.ndarray,
    X_test,   # 🚀 同上
    n_classes: int,
    cfg: dict,
    budget: TimeBudget,
    *,
    skip_tabular: bool = False,
    skip_dl:      bool = False,
    no_nas:       bool = False,
    is_ts:        bool = False,
    artifacts_dir: str = ARTIFACTS_DIR,
    presets_path:  str = _DEFAULT_PRESETS,
    metric:        str = "f1",
) -> PipelineResult:
    
    os.makedirs(artifacts_dir, exist_ok=True)
    all_oof:        list = []
    all_test:       list = []
    model_tags:     list = []
    tabular_configs:list = []
    dl_configs:     list = []

    # =========================================================================
    # 🚀 核心升級：雙軌制資料解包 (Dual-Track Unpacking)
    # =========================================================================
    if isinstance(X_train, dict):
        print("\n[大腦中樞] 偵測到雙軌制字典，啟動自動分流模式 (Tree / DL)...")
        X_train_tree = X_train["tree"]
        X_train_dl   = X_train["dl"]
        X_test_tree  = X_test["tree"]
        X_test_dl    = X_test["dl"]
    else:
        # 向後相容：如果傳入的是單一 Numpy Array，就讓兩軌共用同一份資料
        X_train_tree = X_train_dl = X_train
        X_test_tree  = X_test_dl  = X_test
    # =========================================================================

    # ── [2] Tabular HPO ───────────────────────────────────────────────────────
    if not skip_tabular:
        # Phase 1: Scout
        print(f"\n[2a] Tabular Scout ({cfg['scout_trials']} trials/model, 3-Fold CV) ...")
        print(f"  [Budget] {budget.status_str()}")
        scout_hpo = TabularHPO(
            model_names=_ALL_TABULAR_MODELS,
            n_trials=cfg["scout_trials"],
            top_k=1,
            metric=metric,
        )
        
        # 💡 提示：在 HPO 階段，我們統一給 Tree 軌道的資料讓 XGBoost/LGBM 競爭
        scout_scores, scout_best_params = scout_hpo.scout(
            X_train_tree, y_train,
            scout_trials=cfg["scout_trials"],
            val_size=cfg["scout_val_size"],
            global_cfg=cfg,
        )
        ranked     = sorted(scout_scores.items(), key=lambda kv: kv[1], reverse=True)
        n_keep     = math.ceil(len(ranked) * cfg["scout_ratio"])
        best_score = ranked[0][1] if ranked else 0.0
        threshold  = best_score * (1.0 - cfg.get("scout_drop_tol", 0.07))
        selected = [n for n, s in ranked[:n_keep] if s >= threshold]
        dropped  = [n for n, _ in ranked if n not in selected]
        print("  [Scout] 排名: " + "  ".join(f"{n}={s:.4f}" for n, s in ranked))
        print(f"  [Scout] 閾值: {threshold:.4f} (best={best_score:.4f} × 93%)")
        print(f"  [Scout] 保留 {len(selected)}/{len(ranked)}: {selected}  （淘汰: {dropped}）")

        _presets: dict = {}
        if presets_path and os.path.exists(presets_path):
            with open(presets_path, "r") as f:
                _presets = json.load(f)
        preset_configs = []
        for name in selected:
            if name in _presets:
                p  = dict(_presets[name])
                fs = p.pop("feature_set", "raw")
                preset_configs.append({"model_name": name, "feature_set": fs, "params": p, "score": 0.0})

        # Phase 2: Full HPO
        hpo_trials = budget.scale_trials(cfg["tabular_trials"])
        if budget.should_skip(cost_fraction=0.20):
            print("\n[2b] 時間預算緊迫，跳過 Tabular Full HPO → 僅使用黃金預設值")
            tabular_configs = []
        else:
            selected_scores = [scout_scores[n] for n in selected]
            score_sum = sum(selected_scores) or 1.0
            total_budget = hpo_trials * n_keep
            per_model_trials = {
                name: max(2, round(total_budget * (scout_scores[name] / score_sum)))
                for name in selected
            }
            actual_total = sum(per_model_trials.values())
            print(f"\n[2b] Tabular Full HPO ({actual_total} trials ÷ {n_keep} models, 5-Fold CV) ...")
            
            locked_fs = {
                name: params["feature_set"]
                for name, params in scout_best_params.items()
                if name in selected and "feature_set" in params
            }

            # 🚀 這裡要改成 X_train_tree.shape[1]
            n_features = X_train_tree.shape[1]
            per_model_timeout = {}
            if "catboost" in selected and n_features > 300:
                cat_trials = per_model_trials.get("catboost", 15)
                cat_timeout = max(60, min(300, int(actual_total * 3 / cat_trials)))
                per_model_timeout["catboost"] = cat_timeout

            tabular_hpo = TabularHPO(
                model_names=selected, n_trials=hpo_trials, top_k=cfg["tabular_top_k"],
                per_model_trials=per_model_trials, per_model_timeout=per_model_timeout, metric=metric,
            )
            # 🚀 HPO 使用 Tree 軌道資料
            tabular_configs = tabular_hpo.run(X_train_tree, y_train, cfg,
                                              warm_start=scout_best_params,
                                              locked_feature_sets=locked_fs)

        hpo_names = {c["model_name"] for c in tabular_configs}
        tabular_configs += [c for c in preset_configs if c["model_name"] not in hpo_names]
    else:
        print("\n[2] 跳過 Tabular HPO（skip_tabular=True）")

    # ── [3-6] DL 模型 ─────────────────────────────────────────────────────────
    if not skip_dl:
        skip_nas = no_nas or budget.should_skip(cost_fraction=0.30) or (not is_ts and len(X_train_dl) < 2000)
        if not skip_nas:
            nas_model_type = "TSNet" if is_ts else "MLP"
            print(f"\n[3] {nas_model_type} NAS (epochs={cfg['nas_epochs']}, candidates={cfg['nas_candidates']}) ...")
            nas = TSNASSearcher(...) if is_ts else MLPNASSearcher(
                n_supernet_epochs=cfg["nas_epochs"], n_candidates=cfg["nas_candidates"],
                n_evolution_rounds=cfg["nas_rounds"], device=DEVICE,
            )
            # 🧠 深度學習餵熟肉
            mlp_arch = nas.search(X_train_dl, y_train, n_classes)
        else:
            mlp_arch = _DEFAULT_TSNET_ARCH if is_ts else _DEFAULT_MLP_ARCH

        mlp_trials = budget.scale_trials(cfg["mlp_train_trials"])
        if not budget.should_skip(cost_fraction=0.20):
            print(f"\n[4] MLP/TSNet 訓練 HPO ({mlp_trials} trials) ...")
            mlp_hpo = TSNetTrainHPO(...) if is_ts else MLPTrainHPO(
                arch_params=mlp_arch, n_trials=mlp_trials, top_k=cfg["mlp_top_k"], device=DEVICE, metric=metric,
            )
            # 🧠 深度學習餵熟肉
            mlp_configs = mlp_hpo.run(X_train_dl, y_train, n_classes, cfg)
        else:
            mlp_configs = []

        skip_dl_small = not is_ts and len(X_train_dl) < 2000
        dl_trials = budget.scale_trials(cfg["dl_trials"])
        
        cnn_name  = "tcn" if is_ts else ("resnet1d" if cfg.get("use_resnet18") else "cnn1d")
        if not (budget.should_skip(cost_fraction=0.20) or skip_dl_small):
            cnn_hpo = DLHPO(model_name=cnn_name, n_trials=dl_trials, top_k=cfg["dl_top_k"], n_classes=n_classes, device=DEVICE, metric=metric)
            # 🧠 深度學習餵熟肉
            cnn_configs = cnn_hpo.run(X_train_dl, y_train, cfg)
        else:
            cnn_configs = []

        tf_name   = "patchtst" if is_ts else "transformer"
        if not (budget.should_skip(cost_fraction=0.20) or skip_dl_small):
            tf_hpo = DLHPO(model_name=tf_name, n_trials=dl_trials, top_k=cfg["dl_top_k"], n_classes=n_classes, device=DEVICE, metric=metric)
            # 🧠 深度學習餵熟肉
            tf_configs = tf_hpo.run(X_train_dl, y_train, cfg)
        else:
            tf_configs = []

        dl_configs = mlp_configs + cnn_configs + tf_configs
    else:
        print("\n[3-6] 跳過深度學習模型（skip_dl=True）")

    # ── [7] 5-Fold CV (核心分流器) ────────────────────────────────────────────────
    all_configs = tabular_configs + dl_configs
    if not all_configs:
        raise RuntimeError("沒有任何 model config！請確認 HPO 成功完成。")

    print(f"\n[7] 5-Fold CV — {len(all_configs)} 個模型 config ...")
    for i, config in enumerate(all_configs):
        tag      = f"{config['model_name']}_{config['feature_set']}_c{i}".replace("/", "_")
        oof_path = os.path.join(artifacts_dir, f"{tag}_oof.npy")
        tst_path = os.path.join(artifacts_dir, f"{tag}_test.npy")
        
        # ==========================================
        # 🚀 軌道分流器 (Data Router)
        # 根據模型名稱，決定要發配「生肉」還是「熟肉」
        # ==========================================
        tree_models = ["lgbm", "xgb", "catboost", "rf", "extra_trees"]
        
        if config['model_name'] in tree_models:
            current_X_tr = X_train_tree
            current_X_te = X_test_tree
            track_name = "Tree生肉"
        else:
            # logreg, knn, mlp, resnet1d, transformer 等全部吃熟肉
            current_X_tr = X_train_dl
            current_X_te = X_test_dl
            track_name = "DL熟肉"
            
        print(f"  [Router] 模型 {config['model_name']} 被分配至 ─> {track_name} 軌道")
        # ==========================================

        if os.path.exists(oof_path) and os.path.exists(tst_path):
            print(f"  [CV] 載入快取 {tag}")
            oof       = np.load(oof_path)
            test_pred = np.load(tst_path)
        else:
            # 🚀 這裡傳入路由分配好的 current_X_tr 和 current_X_te
            oof, test_pred = run_cv(
                config, current_X_tr, y_train, current_X_te, n_classes,
                device=DEVICE, tag=tag, global_cfg=cfg, metric=metric,
            )
        all_oof.append(oof)
        all_test.append(test_pred)
        model_tags.append(tag)

    # ── [8] Ensemble A: Nelder-Mead Blending ──────────────────────────────────
    print("\n[8] Ensemble A — Nelder-Mead Weighted Blending ...")
    blender = NelderMeadBlender(n_restarts=cfg["blend_restarts"], metric=metric)
    blender.fit(all_oof, y_train)
    test_blend = blender.predict(all_test)

    # ── [9] Ensemble B: Meta-Learner Stacking ─────────────────────────────────
    print("\n[9] Ensemble B — Meta-Learner Stacking (Concatenated) ...")
    stacker = MetaLearnerStacker(n_meta_trials=cfg["meta_trials"], metric=metric, n_samples=len(y_train))
    # 🚀 Stacker 是 LightGBM 元學習器，它需要吃原始特徵做參考，所以我們餵 Tree 軌道
    stacker.fit(all_oof, y_train, X_orig=X_train_tree, is_timeseries=is_ts)
    test_stack = stacker.predict(all_test, X_orig=X_test_tree)

    return PipelineResult(
        test_blend=test_blend,
        test_stack=test_stack,
        all_oof=all_oof,
        all_test=all_test,
        model_tags=model_tags,
        blender=blender,
        stacker=stacker,
    )