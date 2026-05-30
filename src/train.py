"""
交叉驗證訓練迴圈。

run_tabular_cv   : Tabular 模型的 5-Fold CV，產出 OOF + test 預測
run_dl_cv        : DL 模型（MLP / CNN1D / Transformer）的 5-Fold CV，含：
                   - AdamW Optimizer
                   - CosineAnnealingLR 排程器
                   - CrossEntropyLoss + Label Smoothing
                   - Mixup Data Augmentation
                   - Early Stopping（by Val Macro F1）
train_dl_single_fold : 僅訓練單一 fold，供 HPO 快速評估使用

所有超參數（lr, batch_size, label_smoothing, mixup_alpha, mixup_prob, t_max,
n_epochs, patience, dropout, channels ...）完全由傳入的 config dict 決定。
"""
import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import SEED, DEVICE, ARTIFACTS_DIR
from .data import get_folds, get_ts_folds, make_loader
from .preprocess import FeatureBuilder
from .models.tabular import build_tabular_model
from .metrics import calculate_score, get_metric_name


# ── Mixup ────────────────────────────────────────────────────────────────────

def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, device: str):
    """對一個 batch 套用 Mixup；回傳 mixed_x, y_a, y_b, lam。"""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.beta(alpha, alpha))
    B = x.size(0)
    idx = torch.randperm(B, device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def _aug_1d(x: torch.Tensor, device: str) -> torch.Tensor:
    """Random reverse, Gaussian noise, Random Shift for 1D signals."""
    is_2d = x.ndim == 2
    if is_2d:
        x = x.unsqueeze(1)
    
    B, C, F = x.shape
    if np.random.rand() < 0.5:
        x = x + torch.randn_like(x) * 0.01
    if np.random.rand() < 0.5:
        x = x.flip(dims=[-1])
    if np.random.rand() < 0.5:
        shift = np.random.randint(-int(F * 0.05), int(F * 0.05))
        if shift > 0:
            x = torch.cat([torch.zeros((B, C, shift), device=device), x[:, :, :-shift]], dim=-1)
        elif shift < 0:
            x = torch.cat([x[:, :, -shift:], torch.zeros((B, C, -shift), device=device)], dim=-1)

    if is_2d:
        x = x.squeeze(1)
    return x


# ── DL 模型建構 ──────────────────────────────────────────────────────────────

def _build_dl_model(
    model_name: str, arch_params: dict, in_features: int, n_classes: int
) -> nn.Module:
    """依 model_name + arch_params 建立 DL 模型，不預設任何架構參數。"""
    if model_name == "mlp":
        from .models.mlp import MLP
        depth = arch_params["depth"]
        activations = arch_params.get("activations", ["gelu"] * depth)
        use_skips = arch_params.get("use_skips", [False] * depth)
        return MLP(
            in_features=in_features,
            hidden_dim=arch_params["hidden_dim"],
            depth=depth,
            activations=activations,
            use_skips=use_skips,
            dropout=arch_params["dropout"],
            n_classes=n_classes,
        )
    if model_name == "cnn1d":
        from .models.cnn1d import CNN1D
        return CNN1D(
            in_features=in_features,
            n_blocks=arch_params["n_blocks"],
            channels=arch_params["channels"],
            kernel_size=arch_params["kernel_size"],
            dropout=arch_params["dropout"],
            n_classes=n_classes,
        )
    if model_name == "transformer":
        from .models.transformer import SignalTransformer
        return SignalTransformer(
            in_features=in_features,
            patch_size=arch_params["patch_size"],
            d_model=arch_params["d_model"],
            n_heads=arch_params["n_heads"],
            depth=arch_params["depth"],
            ff_dim=arch_params["ff_dim"],
            dropout=arch_params["dropout"],
            n_classes=n_classes,
            norm_first=arch_params.get("norm_first", True),
        )
    if model_name == "resnet1d":
        from .models.cnn1d import ResNet1D_18
        return ResNet1D_18(
            in_features=in_features,
            channels=arch_params.get("channels", 64),
            dropout=arch_params.get("dropout", 0.2),
            n_classes=n_classes,
        )
    if model_name == "tcn":
        from .models.cnn1d import TCN
        return TCN(
            in_features=in_features,
            n_blocks=arch_params["n_blocks"],
            channels=arch_params["channels"],
            kernel_size=arch_params["kernel_size"],
            dropout=arch_params["dropout"],
            n_classes=n_classes,
        )
    if model_name == "patchtst":
        from .models.transformer import PatchTST
        return PatchTST(
            in_features=in_features,
            patch_size=arch_params["patch_size"],
            d_model=arch_params["d_model"],
            n_heads=arch_params["n_heads"],
            depth=arch_params["depth"],
            ff_dim=arch_params["ff_dim"],
            dropout=arch_params["dropout"],
            n_classes=n_classes,
        )
    if model_name == "tsnet":
        from .nas import TSNet
        return TSNet(
            in_features=in_features,
            channels=arch_params.get("channels", 64),
            operations=arch_params.get("operations", [0, 2, 3]),
            n_classes=n_classes,
            dropout=arch_params.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown DL model: {model_name}")


# ── 單一 Fold 訓練（供 HPO 快速評估）────────────────────────────────────────

def train_dl_single_fold(
    model_name: str,
    arch_params: dict,
    train_params: dict,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    device: str = None,
    global_cfg: dict = None,
    metric: str = "f1",
) -> float:
    """
    訓練單一 fold，回傳 Val Macro F1。
    所有訓練超參數由 train_params 傳入（lr / weight_decay / batch_size /
    label_smoothing / mixup_alpha / mixup_prob / t_max / n_epochs / patience）。
    """
    device = device or DEVICE
    torch.manual_seed(SEED)

    in_features = X_tr.shape[1]
    model = _build_dl_model(model_name, arch_params, in_features, n_classes).to(device)
    global_cfg = global_cfg or {}
    use_1d_aug = global_cfg.get("use_1d_aug", False)

    lr = train_params["lr"]
    wd = train_params["weight_decay"]
    bs = int(train_params["batch_size"])
    ls = train_params["label_smoothing"]
    ma = train_params.get("mixup_alpha", 0.0)
    mp = train_params.get("mixup_prob", 0.0)
    t_max = int(train_params["t_max"])
    n_epochs = int(train_params["n_epochs"])
    patience = int(train_params["patience"])

    # 時序模式下關閉 Mixup（樣本順序有意義，混合會引入未來洩漏）
    if global_cfg.get("is_timeseries", False):
        ma = 0.0
        mp = 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    train_loader = make_loader(X_tr, y_tr, batch_size=bs, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    use_amp = device.startswith("cuda")
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, patience_cnt = 0.0, 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_1d_aug:
                xb = _aug_1d(xb, device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if ma > 0 and np.random.rand() < mp:
                    xb, ya, yb2, lam = _mixup_batch(xb, yb, ma, device)
                    logits = model(xb)
                    loss = _mixup_loss(criterion, logits, ya, yb2, lam)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
            optimizer.zero_grad()
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        scheduler.step()

        # 驗證
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1).cpu().numpy()
        val_score = calculate_score(y_val, preds, metric=metric)

        if val_score > best_f1:
            best_f1 = val_score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # 清理 GPU 記憶體
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return best_f1


# ── Tabular early-stop 包裝 ──────────────────────────────────────────────────

def _fit_tabular_with_early_stop(
    model, model_name: str,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> None:
    """
    對 lgbm / xgb / catboost 使用 eval_set + early stopping 加速訓練。
    其他模型直接 fit。
    """
    import lightgbm as lgb

    if model_name == "lgbm":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=200, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    elif model_name == "xgb":
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
    elif model_name == "catboost":
        # CatBoost 已在 build_tabular_model 設定 od_type="Iter", od_wait=30
        # 額外傳入 eval_set 可啟用更精準的 logloss 早停
        try:
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                verbose=False,
            )
        except Exception:
            model.fit(X_tr, y_tr)
    else:
        model.fit(X_tr, y_tr)


# ── Tabular 5-Fold CV ────────────────────────────────────────────────────────

def run_tabular_cv(
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    device: str = None,
    tag: str = None,
    save_artifacts: bool = True,
    global_cfg: dict = None,
    metric: str = "f1",
) -> tuple:
    """
    對一個 tabular config 執行 5-Fold CV。
    回傳 (oof_probs [N, C], test_probs [N_test, C])，並儲存 .npy 到 artifacts/。

    config keys:
        model_name, feature_set, params

    當 global_cfg["is_timeseries"]=True 時切換為 TimeSeriesSplit（Walk-forward），
    確保訓練集永遠在驗證集之前。
    """
    device = device or DEVICE
    global_cfg = global_cfg or {}
    tag = tag or f"{config['model_name']}_{config['feature_set']}"
    n = len(y)
    oof = np.zeros((n, n_classes), dtype=np.float32)
    oof_counts = np.zeros((n, 1), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)

    is_ts = global_cfg.get("is_timeseries", False)
    n_seeds = 1 if is_ts else global_cfg.get("n_seeds", 1)
    n_repeats = 1 if is_ts else global_cfg.get("n_repeats", 1)

    best_fold_score = -np.inf
    best_fold_model = None
    best_fold_fb    = None

    for seed_idx in range(n_seeds):
        cur_seed = SEED + seed_idx * 100
        if is_ts:
            folds = get_ts_folds(n, n_splits=5)
        else:
            folds = get_folds(y, n_repeats=n_repeats, random_state=cur_seed)

        fold_pbar = tqdm(
            enumerate(folds), total=len(folds),
            desc=f"  CV {tag[:20]:20s} (S{seed_idx})", ncols=90, leave=False
        )
        for fold_idx, (tr_idx, val_idx) in fold_pbar:
            fb = FeatureBuilder(feature_set=config["feature_set"], global_cfg=global_cfg)
            # 🛡️ 加入安全索引防呆機制
            # 如果 X 是 DataFrame，就用 .iloc 切「列」；如果是 Numpy Array，就維持原本的切法
            X_tr_fold = X.iloc[tr_idx] if hasattr(X, 'iloc') else X[tr_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            
            X_tr = fb.fit_transform(X_tr_fold)
            X_val = fb.transform(X_val_fold)
            X_te = fb.transform(X_test)

            _cw = "balanced" if metric != "accuracy" else None
            model = build_tabular_model(config["model_name"], config["params"], device=device, class_weight=_cw)
            _fit_tabular_with_early_stop(model, config["model_name"], X_tr, y[tr_idx], X_val, y[val_idx])

            oof[val_idx] += model.predict_proba(X_val)
            oof_counts[val_idx] += 1
            test_preds += model.predict_proba(X_te) / (len(folds) * n_seeds)

            val_score = calculate_score(
                y[val_idx], model.predict_proba(X_val).argmax(axis=1), metric=metric
            )
            fold_pbar.set_postfix({f"fold_{metric}": f"{val_score:.4f}"})

            if val_score > best_fold_score:
                best_fold_score = val_score
                best_fold_model = model
                best_fold_fb    = fb

    oof /= np.maximum(oof_counts, 1.0)
    oof_score = calculate_score(y, oof.argmax(axis=1), metric=metric)
    print(f"  [CV] {tag:30s} OOF {get_metric_name(metric)} = {oof_score:.4f}")

    if save_artifacts:
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_oof.npy"), oof)
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_test.npy"), test_preds)
        if best_fold_model is not None:
            joblib.dump(best_fold_model, os.path.join(ARTIFACTS_DIR, f"{tag}_best_model.pkl"))
        if best_fold_fb is not None:
            joblib.dump(best_fold_fb, os.path.join(ARTIFACTS_DIR, f"{tag}_best_model_fb.pkl"))

    return oof, test_preds


# ── DL 5-Fold CV ─────────────────────────────────────────────────────────────

def run_dl_cv(
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    device: str = None,
    tag: str = None,
    save_artifacts: bool = True,
    global_cfg: dict = None,
    metric: str = "f1",
) -> tuple:
    """
    對一個 DL config 執行 5-Fold CV。
    回傳 (oof_probs [N, C], test_probs [N_test, C])，並儲存 .npy。

    config keys:
        model_name, feature_set, arch_params, train_params
    """
    device = device or DEVICE
    global_cfg = global_cfg or {}
    tag = tag or f"{config['model_name']}_{config['feature_set']}"
    n = len(y)
    oof = np.zeros((n, n_classes), dtype=np.float32)
    oof_counts = np.zeros((n, 1), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)

    best_fold_score = -np.inf
    best_fold_state = None
    best_fold_in_features = None

    train_params = config.get("train_params") or {}
    arch_params_cfg = config.get("arch_params") or {}
    bs = int(train_params.get("batch_size", 128))
    ls = train_params.get("label_smoothing", 0.0)
    ma = train_params.get("mixup_alpha", 0.0)
    mp = train_params.get("mixup_prob", 0.0)
    t_max = int(train_params.get("t_max", 10))
    n_epochs = int(train_params.get("n_epochs", 30))
    patience = int(train_params.get("patience", 7))
    use_1d_aug = global_cfg.get("use_1d_aug", False)

    is_ts = global_cfg.get("is_timeseries", False)
    # 時序模式下關閉 Mixup
    if is_ts:
        ma = 0.0
        mp = 0.0
    n_seeds = 1 if is_ts else global_cfg.get("n_seeds", 1)
    n_repeats = 1 if is_ts else global_cfg.get("n_repeats", 1)

    for seed_idx in range(n_seeds):
        cur_seed = SEED + seed_idx * 100
        if is_ts:
            folds = get_ts_folds(n, n_splits=5)
        else:
            folds = get_folds(y, n_repeats=n_repeats, random_state=cur_seed)

        for fold_idx, (tr_idx, val_idx) in enumerate(
            tqdm(folds, desc=f"  CV {tag[:20]:20s} (S{seed_idx})", ncols=90, leave=False)
        ):
            torch.manual_seed(cur_seed + fold_idx)

            fb = FeatureBuilder(feature_set=config["feature_set"], global_cfg=global_cfg)
            # 🛡️ 加入安全索引防呆機制
            # 如果 X 是 DataFrame，就用 .iloc 切「列」；如果是 Numpy Array，就維持原本的切法
            X_tr_fold = X.iloc[tr_idx] if hasattr(X, 'iloc') else X[tr_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
            
            X_tr = fb.fit_transform(X_tr_fold)
            X_val = fb.transform(X_val_fold)
            X_te = fb.transform(X_test)

            in_features = X_tr.shape[1]
            model = _build_dl_model(
                config["model_name"], arch_params_cfg, in_features, n_classes
            ).to(device)

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=train_params.get("lr", 1e-3),
                weight_decay=train_params.get("weight_decay", 1e-4),
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
            criterion = nn.CrossEntropyLoss(label_smoothing=ls)

            train_loader = make_loader(X_tr, y[tr_idx], batch_size=bs, shuffle=True)
            X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
            X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

            use_amp = device.startswith("cuda")
            amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            best_f1, patience_cnt = 0.0, 0
            best_state = None

            epoch_pbar = tqdm(
                range(n_epochs),
                desc=f"    Fold {fold_idx + 1} epochs",
                leave=False,
                ncols=90,
            )
            for epoch in epoch_pbar:
                model.train()
                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    if use_1d_aug:
                        xb = _aug_1d(xb, device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        if ma > 0 and np.random.rand() < mp:
                            xb, ya, yb2, lam = _mixup_batch(xb, yb, ma, device)
                            logits = model(xb)
                            loss = _mixup_loss(criterion, logits, ya, yb2, lam)
                        else:
                            logits = model(xb)
                            loss = criterion(logits, yb)
                    optimizer.zero_grad()
                    amp_scaler.scale(loss).backward()
                    amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                scheduler.step()

                with torch.no_grad():
                    preds = model(X_val_t).argmax(dim=1).cpu().numpy()
                val_score = calculate_score(y[val_idx], preds, metric=metric)
                epoch_pbar.set_postfix({f"val_{metric}": f"{val_score:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

                if val_score > best_f1:
                    best_f1 = val_score
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_cnt = 0
                else:
                    patience_cnt += 1
                    if patience_cnt >= patience:
                        break

            # 用最佳 checkpoint 產出預測
            if best_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            with torch.no_grad():
                fold_oof_probs = torch.softmax(model(X_val_t), dim=1).cpu().numpy()
                oof[val_idx] += fold_oof_probs
                oof_counts[val_idx] += 1
                test_preds += torch.softmax(model(X_te_t), dim=1).cpu().numpy() / (len(folds) * n_seeds)

            if best_f1 > best_fold_score:
                best_fold_score = best_f1
                best_fold_state = best_state
                best_fold_in_features = in_features

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    oof /= np.maximum(oof_counts, 1.0)
    oof_score = calculate_score(y, oof.argmax(axis=1), metric=metric)
    print(f"  [CV] {tag:30s} OOF {get_metric_name(metric)} = {oof_score:.4f}")

    if save_artifacts:
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_oof.npy"), oof)
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_test.npy"), test_preds)
        if best_fold_state is not None:
            torch.save(
                {
                    "state_dict": best_fold_state,
                    "config": config,
                    "in_features": best_fold_in_features,
                    "n_classes": n_classes,
                },
                os.path.join(ARTIFACTS_DIR, f"{tag}_best_model.pt"),
            )

    return oof, test_preds


# ── 統一入口（依 config 自動選擇 tabular / DL）───────────────────────────────

def run_cv(
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    n_classes: int,
    device: str = None,
    tag: str = None,
    save_artifacts: bool = True,
    global_cfg: dict = None,
    metric: str = "f1",
) -> tuple:
    """
    根據 config["model_name"] 自動選擇 run_tabular_cv 或 run_dl_cv。
    """
    dl_models = {"mlp", "cnn1d", "resnet1d", "transformer", "tcn", "patchtst", "tsnet"}
    if config["model_name"] in dl_models:
        return run_dl_cv(config, X, y, X_test, n_classes, device, tag, save_artifacts, global_cfg, metric)
    return run_tabular_cv(config, X, y, X_test, n_classes, device, tag, save_artifacts, global_cfg, metric)
