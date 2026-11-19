"""PropMLP — robust propagation-feature MLP baseline.

A deliberately simple but stable alternative to UPFD-prop, which collapsed to
the majority floor under our budget. PropMLP extracts a small set of
text-free cascade-summary features (size, distinct event proxies/URLs/media,
early-window popularity, hash-bucket concentration, temporal spread) and trains
a 2-layer MLP. It needs no message passing, so it does not suffer the
low-data fragility of the UPFD LayerNorm+GELU head, giving the main table a
non-collapsing learned propagation-only comparison point.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..causal.strata import _echo_chamber, _popularity
from ..data.schema import Cascade
from ..eval.stats import macro_f1
from ..train import DataConfig, _load_cascades, _set_seed, _stratified_split, _story_summary_rich


def _features(cascades: list[Cascade], rich: bool = False) -> np.ndarray:
    """Per-story summary feature matrix [n, d]. d=7 by default; d=15 when
    rich=True (adds temporal-dynamics features matching _story_summary_rich)."""
    if rich:
        return _story_summary_rich(cascades)
    rows = []
    for c in cascades:
        n_events = len(c.events)
        ts = [e.timestamp for e in c.events]
        spread = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
        rows.append([
            np.log1p(n_events),
            np.log1p(c.num_users()),
            np.log1p(c.num_urls()),
            np.log1p(c.num_media()),
            np.log1p(_popularity(c)),
            _echo_chamber(c),
            np.log1p(spread),
        ])
    return np.asarray(rows, dtype=np.float64)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 32, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def train_propmlp(
    seed: int,
    data_cfg: DataConfig,
    hidden_dim: int = 32,
    epochs: int = 25,
    learning_rate: float = 1e-2,
    device: torch.device | None = None,
    verbose: bool = False,
    rich: bool = False,
) -> dict:
    """PropMLP baseline. rich=True swaps in the 15-dim temporal-dynamics feature
    set (PropMLP+); the default 7-dim is kept for back-compat with prior tables."""
    _set_seed(seed)
    device = device or torch.device("cpu")

    cascades = _load_cascades(data_cfg, seed)
    tr, va, te = _stratified_split(cascades, data_cfg.split_ratios, seed)

    x_tr = _features(tr, rich=rich)
    x_te = _features(te, rich=rich)
    mu = x_tr.mean(0, keepdims=True)
    sd = x_tr.std(0, keepdims=True) + 1e-6
    x_tr = (x_tr - mu) / sd
    x_te = (x_te - mu) / sd

    xt = torch.tensor(x_tr, dtype=torch.float32, device=device)
    xe = torch.tensor(x_te, dtype=torch.float32, device=device)
    y_tr = torch.tensor([c.label for c in tr], dtype=torch.long, device=device)
    y_te = torch.tensor([c.label for c in te], dtype=torch.long, device=device)

    model = _MLP(in_dim=xt.shape[1], hidden=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.cross_entropy(model(xt), y_tr)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [propmlp ep {ep+1}/{epochs}] ce={loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        preds = model(xe).argmax(-1).cpu().numpy()
    labels = y_te.cpu().numpy()

    return {
        "seed": seed,
        "tau": float("inf"),
        "coverage": 1.0,
        "acc_all": float((preds == labels).mean()),
        "acc_kept": float((preds == labels).mean()),
        "f1_all": macro_f1(preds, labels),
        "f1_kept": macro_f1(preds, labels),
        "n_test": int(labels.shape[0]),
        "method": "propmlp_rich" if rich else "propmlp",
        "preds": preds.tolist(),
        "labels": labels.tolist(),
    }
