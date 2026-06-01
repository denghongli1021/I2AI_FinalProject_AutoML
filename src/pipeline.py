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
import shutil
import numpy as np

from src.config import DEVICE, ARTIFACTS_DIR
from src.hpo import TabularHPO, DLHPO, MLPTrainHPO, TSNetTrainHPO
from src.nas import MLPNASSearcher, TSNASSearcher
from src.train import run_cv
from src.ensemble import NelderMeadBlender, MetaLearnerStacker

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PRESETS = os.path.join(_HERE, "best_presets.json")

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
            "dl_trials": 3,       "transformer_trials": 10, "dl_top_k": 1,
            "meta_trials": 3,     "blend_restarts": 1,
            "n_repeats": 1,       "n_seeds": 1,
            "use_kpca": False,    "use_kmeans": False,
            "use_resnet18": False,"use_1d_aug": False,
            "is_fast": True,      "tabular_model_timeout": 1000,  # 每模型最多 ~17 分鐘
        }
    if n_samples < 500:
        return {
            "tabular_trials": 30, "tabular_top_k": 2,
            "scout_trials": 5,    "scout_val_size": 0.2, "scout_ratio": 2 / 3,
            "nas_epochs": 8,      "nas_candidates": 8,   "nas_rounds": 2,
            "mlp_train_trials": 8,"mlp_top_k": 1,
            "dl_trials": 8,       "transformer_trials": 12, "dl_top_k": 1,
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
            "dl_trials": 8,       "transformer_trials": 15, "dl_top_k": 1,
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
        "dl_trials": 6,       "transformer_trials": 10, "dl_top_k": 1,
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
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
    """
    完整 Pipeline 引擎：HPO → NAS → CV → Ensemble。

    Parameters
    ----------
    X_train, y_train : 訓練特徵與標籤（numpy，y 為整數編碼）
    X_test           : 測試特徵（無標籤）
    n_classes        : 類別數
    cfg              : 超參數設定（由 get_cfg() 產生）
    budget           : TimeBudget 時間預算管理器
    skip_tabular     : 跳過傳統模型 HPO
    skip_dl          : 跳過所有深度學習模型
    no_nas           : 跳過 NAS，MLP 使用預設架構
    is_ts            : 時序模式（TCN/PatchTST 取代 CNN/Transformer）
    artifacts_dir    : OOF/Test 預測快取目錄
    presets_path     : 黃金預設值 JSON 路徑

    Returns
    -------
    PipelineResult
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    all_oof:        list = []
    all_test:       list = []
    model_tags:     list = []
    tabular_configs:list = []
    dl_configs:     list = []

    # ── [2] Tabular HPO ───────────────────────────────────────────────────────
    if not skip_tabular:
        # Phase 1: Scout — 單次 holdout，快速淘汰弱模型
        print(f"\n[2a] Tabular Scout ({cfg['scout_trials']} trials/model, 3-Fold CV) ...")
        print(f"  [Budget] {budget.status_str()}")
        scout_hpo = TabularHPO(
            model_names=_ALL_TABULAR_MODELS,
            n_trials=cfg["scout_trials"],
            top_k=1,
            metric=metric,
        )
        scout_scores, scout_best_params = scout_hpo.scout(
            X_train, y_train,
            scout_trials=cfg["scout_trials"],
            val_size=cfg["scout_val_size"],
            global_cfg=cfg,
        )
        ranked     = sorted(scout_scores.items(), key=lambda kv: kv[1], reverse=True)
        n_keep     = math.ceil(len(ranked) * cfg["scout_ratio"])
        best_score = ranked[0][1] if ranked else 0.0
        threshold  = best_score * (1.0 - cfg.get("scout_drop_tol", 0.07))
        # 保留條件：同時滿足「前 2/3」與「不低於最佳 7%」
        selected = [n for n, s in ranked[:n_keep] if s >= threshold]
        dropped  = [n for n, _ in ranked if n not in selected]
        print("  [Scout] 排名: " + "  ".join(f"{n}={s:.4f}" for n, s in ranked))
        print(f"  [Scout] 閾值: {threshold:.4f} (best={best_score:.4f} × 93%)")
        print(f"  [Scout] 保留 {len(selected)}/{len(ranked)}: {selected}  （淘汰: {dropped}）")

        # 黃金預設值保底：Scout 結束後立即注入，確保時間不足時仍有可用模型
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
        if preset_configs:
            print(f"  [Preset] 注入 {len(preset_configs)} 組黃金預設值："
                  f"{[c['model_name'] for c in preset_configs]}")

        # Phase 2: Full HPO
        hpo_trials = budget.scale_trials(cfg["tabular_trials"])
        if budget.should_skip(cost_fraction=0.20):
            print("\n[2b] 時間預算緊迫，跳過 Tabular Full HPO → 僅使用黃金預設值")
            tabular_configs = []
        else:
            # 依 scout 分數按比例分配 trials：分數高的模型獲得更多搜尋次數，總量不變
            selected_scores = [scout_scores[n] for n in selected]
            score_sum = sum(selected_scores) or 1.0
            total_budget = hpo_trials * n_keep
            per_model_trials = {
                name: max(2, round(total_budget * (scout_scores[name] / score_sum)))
                for name in selected
            }
            actual_total = sum(per_model_trials.values())
            print(f"\n[2b] Tabular Full HPO ({actual_total} trials ÷ {n_keep} models, 5-Fold CV) ...")
            print("  [Alloc] " + "  ".join(f"{n}={t}" for n, t in per_model_trials.items()))

            # 從 Scout 結果鎖定每個模型的最佳 feature_set，讓 HPO 專注在超參數空間
            locked_fs = {
                name: params["feature_set"]
                for name, params in scout_best_params.items()
                if name in selected and "feature_set" in params
            }
            if locked_fs:
                print("  [FS Lock] " + "  ".join(f"{n}={fs}" for n, fs in locked_fs.items()))

            # fast 模式或高維 CatBoost：設定 per-model timeout 防止單模型卡住 pipeline
            n_features = X_train.shape[1]
            per_model_timeout = {}
            _fast_timeout = cfg.get("tabular_model_timeout")
            if _fast_timeout:
                # fast 模式：所有模型都加上時間上限
                for _m in selected:
                    per_model_timeout[_m] = _fast_timeout
                print(f"  [Timeout] fast 模式：各 tabular 模型最多 {_fast_timeout}s")
            elif "catboost" in selected and n_features > 300:
                # 高維資料對 CatBoost 設定 per-model timeout
                cat_trials = per_model_trials.get("catboost", 15)
                cat_timeout = max(60, min(300, int(actual_total * 3 / cat_trials)))
                per_model_timeout["catboost"] = cat_timeout
                print(f"  [Timeout] catboost={cat_timeout}s (n_features={n_features})")

            tabular_hpo = TabularHPO(
                model_names=selected,
                n_trials=hpo_trials,
                top_k=cfg["tabular_top_k"],
                per_model_trials=per_model_trials,
                per_model_timeout=per_model_timeout,
                metric=metric,
            )
            tabular_configs = tabular_hpo.run(X_train, y_train, cfg,
                                              warm_start=scout_best_params,
                                              locked_feature_sets=locked_fs)

        hpo_names = {c["model_name"] for c in tabular_configs}
        tabular_configs += [c for c in preset_configs if c["model_name"] not in hpo_names]
    else:
        print("\n[2] 跳過 Tabular HPO（skip_tabular=True）")

    # ── [3-6] DL 模型 ─────────────────────────────────────────────────────────
    if not skip_dl:
        # [3] NAS
        # 小型表格資料（< 2000 筆）做 NAS 容易過擬合且耗時，自動跳過
        skip_nas = no_nas or budget.should_skip(cost_fraction=0.30) or (
            not is_ts and len(X_train) < 2000
        )
        if not skip_nas:
            if is_ts:
                print(f"\n[3] TSNet NAS (epochs={cfg['nas_epochs']}, candidates={cfg['nas_candidates']}) ...")
                print(f"  [Budget] {budget.status_str()}")
                nas = TSNASSearcher(
                    n_supernet_epochs=cfg["nas_epochs"],
                    n_candidates=cfg["nas_candidates"],
                    n_evolution_rounds=cfg["nas_rounds"],
                    device=DEVICE,
                )
                mlp_arch = nas.search(X_train, y_train, n_classes)
            else:
                print(f"\n[3] MLP NAS (epochs={cfg['nas_epochs']}, candidates={cfg['nas_candidates']}) ...")
                print(f"  [Budget] {budget.status_str()}")
                nas = MLPNASSearcher(
                    n_supernet_epochs=cfg["nas_epochs"],
                    n_candidates=cfg["nas_candidates"],
                    n_evolution_rounds=cfg["nas_rounds"],
                    device=DEVICE,
                )
                mlp_arch = nas.search(X_train, y_train, n_classes)
        else:
            reason = "時間預算不足" if not no_nas else "no_nas=True"
            print(f"\n[3] 跳過 NAS（{reason}），使用預設架構")
            mlp_arch = _DEFAULT_TSNET_ARCH if is_ts else _DEFAULT_MLP_ARCH

        # [4] MLP / TSNet 訓練 HPO
        mlp_trials = budget.scale_trials(cfg["mlp_train_trials"])
        if budget.should_skip(cost_fraction=0.20):
            hpo_model_name = "TSNet" if is_ts else "MLP"
            print(f"\n[4] 時間預算緊迫，跳過 {hpo_model_name} 訓練 HPO")
            mlp_configs = []
        else:
            if is_ts:
                print(f"\n[4] TSNet 訓練 HPO ({mlp_trials} trials) ...")
                mlp_hpo = TSNetTrainHPO(
                    arch_params=mlp_arch, n_trials=mlp_trials,
                    top_k=cfg["mlp_top_k"], device=DEVICE,
                    metric=metric,
                )
            else:
                print(f"\n[4] MLP 訓練 HPO ({mlp_trials} trials) ...")
                mlp_hpo = MLPTrainHPO(
                    arch_params=mlp_arch, n_trials=mlp_trials,
                    top_k=cfg["mlp_top_k"], device=DEVICE,
                    metric=metric,
                )
            mlp_configs = mlp_hpo.run(X_train, y_train, n_classes, cfg)

        # [5] CNN1D / TCN HPO
        # 小型表格資料（< 2000 筆）DL 模型無法收斂且拖低 ensemble，自動跳過
        skip_dl_small = not is_ts and len(X_train) < 2000
        cnn_name  = "tcn" if is_ts else ("resnet1d" if cfg.get("use_resnet18") else "cnn1d")
        dl_trials = budget.scale_trials(cfg["dl_trials"])
        if budget.should_skip(cost_fraction=0.20) or skip_dl_small:
            reason = "時間預算緊迫" if budget.should_skip(0.20) else f"n_train={len(X_train)}<2000"
            print(f"\n[5] 跳過 {cnn_name.upper()} HPO（{reason}）")
            cnn_configs = []
        else:
            print(f"\n[5] {cnn_name.upper()} HPO ({dl_trials} trials) ...")
            cnn_hpo = DLHPO(
                model_name=cnn_name, n_trials=dl_trials,
                top_k=cfg["dl_top_k"], n_classes=n_classes, device=DEVICE,
                metric=metric,
            )
            cnn_configs = cnn_hpo.run(X_train, y_train, cfg)

        # [6] Transformer / PatchTST HPO
        tf_name   = "patchtst" if is_ts else "transformer"
        tf_trials = budget.scale_trials(cfg.get("transformer_trials", cfg["dl_trials"]))
        if budget.should_skip(cost_fraction=0.20) or skip_dl_small:
            reason = "時間預算緊迫" if budget.should_skip(0.20) else f"n_train={len(X_train)}<2000"
            print(f"\n[6] 跳過 {tf_name.upper()} HPO（{reason}）")
            tf_configs = []
        else:
            print(f"\n[6] {tf_name.upper()} HPO ({tf_trials} trials) ...")
            tf_hpo = DLHPO(
                model_name=tf_name, n_trials=tf_trials,
                top_k=cfg["dl_top_k"], n_classes=n_classes, device=DEVICE,
                metric=metric,
            )
            tf_configs = tf_hpo.run(X_train, y_train, cfg)

        dl_configs = mlp_configs + cnn_configs + tf_configs
    else:
        print("\n[3-6] 跳過深度學習模型（skip_dl=True）")

    # ── [7] 5-Fold CV ─────────────────────────────────────────────────────────
    all_configs = tabular_configs + dl_configs
    if not all_configs:
        raise RuntimeError("沒有任何 model config！請確認 HPO 成功完成。")

    print(f"\n[7] 5-Fold CV — {len(all_configs)} 個模型 config ...")
    for i, config in enumerate(all_configs):
        tag      = f"{config['model_name']}_{config['feature_set']}_c{i}".replace("/", "_")
        oof_path = os.path.join(artifacts_dir, f"{tag}_oof.npy")
        tst_path = os.path.join(artifacts_dir, f"{tag}_test.npy")
        if os.path.exists(oof_path) and os.path.exists(tst_path):
            print(f"  [CV] 載入快取 {tag}")
            oof      = np.load(oof_path)
            test_pred = np.load(tst_path)
        else:
            oof, test_pred = run_cv(
                config, X_train, y_train, X_test, n_classes,
                device=DEVICE, tag=tag, global_cfg=cfg, metric=metric,
            )
        all_oof.append(oof)
        all_test.append(test_pred)
        model_tags.append(tag)

    # ── [7.5] 儲存最佳 Tabular 與 DL 模型 ────────────────────────────────────
    from src.metrics import calculate_score as _calc
    _TABULAR   = {"lgbm", "xgb", "catboost", "rf", "extra_trees", "logreg", "knn"}
    _DL        = {"mlp", "cnn1d", "resnet1d", "tcn", "transformer", "patchtst", "tsnet"}
    _ALL_NAMES = sorted(_TABULAR | _DL, key=len, reverse=True)

    def _model_type(tag: str) -> str:
        """從 tag（如 extra_trees_raw_stat_c0）還原正確的 model_name。"""
        for n in _ALL_NAMES:
            if tag.startswith(n + "_"):
                return n
        return tag.split("_")[0]

    tab_candidates = [
        (_calc(y_train, oof.argmax(1), metric=metric), tag)
        for tag, oof in zip(model_tags, all_oof)
        if _model_type(tag) in _TABULAR
    ]
    dl_candidates = [
        (_calc(y_train, oof.argmax(1), metric=metric), tag)
        for tag, oof in zip(model_tags, all_oof)
        if _model_type(tag) in _DL
    ]
    for candidates, ext, dst_name in [
        (tab_candidates, ".pkl", "best_tabular_model.pkl"),
        (dl_candidates,  ".pt",  "best_dl_model.pt"),
    ]:
        if not candidates:
            continue
        _, best_tag = max(candidates)
        src = os.path.join(ARTIFACTS_DIR, f"{best_tag}_best_model{ext}")
        dst = os.path.join(artifacts_dir, dst_name)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [Best Model] {best_tag} → {dst_name}")
        else:
            print(f"  [Best Model] 模型檔未找到（已跳過快取？）：{src}")

    # Copy FeatureBuilder for the best tabular model (used by SHAP visualization)
    if tab_candidates:
        _, best_tab_tag = max(tab_candidates)
        fb_src = os.path.join(ARTIFACTS_DIR, f"{best_tab_tag}_best_model_fb.pkl")
        fb_dst = os.path.join(artifacts_dir, "best_tabular_model_fb.pkl")
        if os.path.exists(fb_src):
            shutil.copy2(fb_src, fb_dst)
            print(f"  [Best Model] FeatureBuilder({best_tab_tag}) → best_tabular_model_fb.pkl")

    # ── [7.5b] 每種模型類型各儲存一個最佳模型 ──────────────────────────────────
    model_best: dict = {}  # model_name -> (score, tag)
    for tag, oof in zip(model_tags, all_oof):
        mname = _model_type(tag)
        score = _calc(y_train, oof.argmax(1), metric=metric)
        if mname not in model_best or score > model_best[mname][0]:
            model_best[mname] = (score, tag)

    print("\n  [Per-Model Best] 儲存各模型類型最佳版本：")
    for mname, (score, best_tag) in sorted(model_best.items()):
        ext = ".pkl" if mname in _TABULAR else ".pt"
        src = os.path.join(ARTIFACTS_DIR, f"{best_tag}_best_model{ext}")
        dst = os.path.join(artifacts_dir, f"{mname}_best_model{ext}")
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    {mname}_best_model{ext}  (OOF {metric}={score:.4f}，來自 {best_tag})")
        else:
            print(f"    {mname}_best_model{ext}  ← 模型檔未找到（快取跳過？）")
        if mname in _TABULAR:
            fb_src = os.path.join(ARTIFACTS_DIR, f"{best_tag}_best_model_fb.pkl")
            fb_dst = os.path.join(artifacts_dir, f"{mname}_best_model_fb.pkl")
            if os.path.exists(fb_src):
                shutil.copy2(fb_src, fb_dst)

    # ── [7.6] 丟掉最差 DL 模型 ───────────────────────────────────────────────
    if len(dl_candidates) > 1:
        worst_score, worst_tag = min(dl_candidates)
        worst_idx = model_tags.index(worst_tag)
        all_oof.pop(worst_idx)
        all_test.pop(worst_idx)
        model_tags.pop(worst_idx)
        print(f"\n[7.6] Prune 最差 DL 模型 {worst_tag}（OOF {metric}={worst_score:.4f}）")

    # ── [8] Ensemble A: Nelder-Mead Blending ──────────────────────────────────
    print("\n[8] Ensemble A — Nelder-Mead Weighted Blending ...")
    blender = NelderMeadBlender(n_restarts=cfg["blend_restarts"], metric=metric)
    blender.fit(all_oof, y_train)
    test_blend = blender.predict(all_test)

    # ── [9] Ensemble B: Meta-Learner Stacking ─────────────────────────────────
    print("\n[9] Ensemble B — Meta-Learner Stacking (Concatenated) ...")
    stacker = MetaLearnerStacker(n_meta_trials=cfg["meta_trials"], metric=metric, n_samples=len(y_train))
    stacker.fit(all_oof, y_train, X_orig=X_train, is_timeseries=is_ts)
    test_stack = stacker.predict(all_test, X_orig=X_test)

    return PipelineResult(
        test_blend=test_blend,
        test_stack=test_stack,
        all_oof=all_oof,
        all_test=all_test,
        model_tags=model_tags,
        blender=blender,
        stacker=stacker,
    )
