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
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import SEED, DEVICE, ARTIFACTS_DIR
from .data import get_folds, make_loader
from .preprocess import FeatureBuilder
from .models.tabular import build_tabular_model


# ── Mixup ────────────────────────────────────────────────────────────────────

def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, device: str):
    """對一個 batch 套用 Mixup；回傳 mixed_x, y_a, y_b, lam。"""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    B = x.size(0)
    idx = torch.randperm(B, device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _mixup_loss(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


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

    lr = train_params["lr"]
    wd = train_params["weight_decay"]
    bs = int(train_params["batch_size"])
    ls = train_params["label_smoothing"]
    ma = train_params["mixup_alpha"]
    mp = train_params["mixup_prob"]
    t_max = int(train_params["t_max"])
    n_epochs = int(train_params["n_epochs"])
    patience = int(train_params["patience"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)

    train_loader = make_loader(X_tr, y_tr, batch_size=bs, shuffle=True)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)

    best_f1, patience_cnt = 0.0, 0
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if ma > 0 and np.random.rand() < mp:
                xb, ya, yb2, lam = _mixup_batch(xb, yb, ma, device)
                logits = model(xb)
                loss = _mixup_loss(criterion, logits, ya, yb2, lam)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        # 驗證
        model.eval()
        with torch.no_grad():
            preds = model(X_val_t).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val, preds, average="macro", zero_division=0)

        if val_f1 > best_f1:
            best_f1 = val_f1
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
) -> tuple:
    """
    對一個 tabular config 執行 5-Fold CV。
    回傳 (oof_probs [N, C], test_probs [N_test, C])，並儲存 .npy 到 artifacts/。

    config keys:
        model_name, feature_set, params
    """
    device = device or DEVICE
    tag = tag or f"{config['model_name']}_{config['feature_set']}"
    folds = get_folds(y)
    n = len(y)
    oof = np.zeros((n, n_classes), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)

    fold_pbar = tqdm(enumerate(folds), total=len(folds), desc=f"  CV {tag:30s}", ncols=90)
    for fold_idx, (tr_idx, val_idx) in fold_pbar:
        fb = FeatureBuilder(feature_set=config["feature_set"])
        X_tr = fb.fit_transform(X[tr_idx])
        X_val = fb.transform(X[val_idx])
        X_te = fb.transform(X_test)

        model = build_tabular_model(config["model_name"], config["params"], device=device)
        model.fit(X_tr, y[tr_idx])

        oof[val_idx] = model.predict_proba(X_val)
        test_preds += model.predict_proba(X_te) / len(folds)

        val_f1 = f1_score(
            y[val_idx], oof[val_idx].argmax(axis=1), average="macro", zero_division=0
        )
        fold_pbar.set_postfix({"fold_f1": f"{val_f1:.4f}"})

    oof_f1 = f1_score(y, oof.argmax(axis=1), average="macro", zero_division=0)
    print(f"  [CV] {tag:30s} OOF Macro F1 = {oof_f1:.4f}")

    if save_artifacts:
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_oof.npy"), oof)
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_test.npy"), test_preds)

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
) -> tuple:
    """
    對一個 DL config 執行 5-Fold CV。
    回傳 (oof_probs [N, C], test_probs [N_test, C])，並儲存 .npy。

    config keys:
        model_name, feature_set, arch_params, train_params
    """
    device = device or DEVICE
    tag = tag or f"{config['model_name']}_{config['feature_set']}"
    folds = get_folds(y)
    n = len(y)
    oof = np.zeros((n, n_classes), dtype=np.float32)
    test_preds = np.zeros((len(X_test), n_classes), dtype=np.float32)

    train_params = config["train_params"]
    bs = int(train_params["batch_size"])
    ls = train_params["label_smoothing"]
    ma = train_params["mixup_alpha"]
    mp = train_params["mixup_prob"]
    t_max = int(train_params["t_max"])
    n_epochs = int(train_params["n_epochs"])
    patience = int(train_params["patience"])

    for fold_idx, (tr_idx, val_idx) in enumerate(
        tqdm(folds, desc=f"  CV {tag:30s}", ncols=90)
    ):
        torch.manual_seed(SEED + fold_idx)

        fb = FeatureBuilder(feature_set=config["feature_set"])
        X_tr = fb.fit_transform(X[tr_idx])
        X_val = fb.transform(X[val_idx])
        X_te = fb.transform(X_test)

        in_features = X_tr.shape[1]
        model = _build_dl_model(
            config["model_name"], config["arch_params"], in_features, n_classes
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_params["lr"],
            weight_decay=train_params["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        criterion = nn.CrossEntropyLoss(label_smoothing=ls)

        train_loader = make_loader(X_tr, y[tr_idx], batch_size=bs, shuffle=True)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)

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
                if ma > 0 and np.random.rand() < mp:
                    xb, ya, yb2, lam = _mixup_batch(xb, yb, ma, device)
                    logits = model(xb)
                    loss = _mixup_loss(criterion, logits, ya, yb2, lam)
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                preds = model(X_val_t).argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y[val_idx], preds, average="macro", zero_division=0)
            epoch_pbar.set_postfix({"val_f1": f"{val_f1:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

            if val_f1 > best_f1:
                best_f1 = val_f1
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
            oof[val_idx] = torch.softmax(model(X_val_t), dim=1).cpu().numpy()
            test_preds += torch.softmax(model(X_te_t), dim=1).cpu().numpy() / len(folds)

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    oof_f1 = f1_score(y, oof.argmax(axis=1), average="macro", zero_division=0)
    print(f"  [CV] {tag:30s} OOF Macro F1 = {oof_f1:.4f}")

    if save_artifacts:
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_oof.npy"), oof)
        np.save(os.path.join(ARTIFACTS_DIR, f"{tag}_test.npy"), test_preds)

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
) -> tuple:
    """
    根據 config["model_name"] 自動選擇 run_tabular_cv 或 run_dl_cv。
    """
    dl_models = {"mlp", "cnn1d", "transformer"}
    if config["model_name"] in dl_models:
        return run_dl_cv(config, X, y, X_test, n_classes, device, tag, save_artifacts)
    return run_tabular_cv(config, X, y, X_test, n_classes, device, tag, save_artifacts)
