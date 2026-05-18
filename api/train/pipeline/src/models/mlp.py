"""
可配置 MLP（供 NAS 搜尋與獨立訓練使用）。
架構完全由外部傳入，不預設層數、寬度或激活函式。
"""
import torch
import torch.nn as nn

ACT_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


class MLP(nn.Module):
    """
    Parameters
    ----------
    in_features : int
    hidden_dim : int
        每層寬度（所有隱藏層共享同一維度，方便 NAS 的 skip connection）。
    depth : int
        隱藏層數。
    activations : list[str]
        每層激活函式名稱（"relu" / "gelu" / "silu"），長度 = depth。
    use_skips : list[bool]
        每層是否啟用殘差連接，長度 = depth。
    dropout : float
    n_classes : int
    use_bn : bool
        是否使用 BatchNorm。
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        depth: int,
        activations: list,
        use_skips: list,
        dropout: float,
        n_classes: int,
        use_bn: bool = True,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_dim)
        self.in_bn = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.in_act = ACT_MAP[activations[0] if activations else "gelu"]()

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.skips = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(depth):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity())
            act_name = activations[i] if i < len(activations) else "gelu"
            self.acts.append(ACT_MAP[act_name]())
            self.skips.append(
                nn.Identity() if use_skips[i] else None
            )
            self.dropouts.append(nn.Dropout(dropout))

        self.head = nn.Linear(hidden_dim, n_classes)
        self._use_skips = use_skips
        self._depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_act(self.in_bn(self.in_proj(x)))
        for i in range(self._depth):
            out = self.dropouts[i](self.acts[i](self.bns[i](self.layers[i](h))))
            if self._use_skips[i]:
                h = h + out
            else:
                h = out
        return self.head(h)
