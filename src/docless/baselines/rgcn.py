"""RGCN -- relational (heterogeneous) graph convolution, no text.

A stronger graph baseline than GCNFN/BiGCN under the same text-free protocol
(a stronger recent heterogeneous graph baseline). Unlike
GCNFN (which collapses all hyperedges into one mean-aggregation operator), RGCN
(Schlichtkrull et al., 2018) keeps a *separate* learnable transform per relation
type -- here the four hyperedge types share/reply/coexpose/burst -- so it can in
principle exploit the heterogeneity DocLessFake also targets, but with a simpler,
well-established operator. It is trained on exactly the same per-split hypergraph,
features, budget, and readout as the other graph baselines.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..eval.stats import macro_f1
from ..hypergraph import build_hypergraph
from ..hypergraph.layers import _scatter_add
from ..train import DataConfig, _load_cascades, _set_seed, _stratified_split

N_REL = 4  # share / reply / coexpose / burst


class _RGCN(nn.Module):
    """Two-layer relational GCN over typed hyperedges (clique expansion)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        # per-relation transforms for two message-passing layers + self loop
        self.rel1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(N_REL)])
        self.rel2 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(N_REL)])
        self.self1 = nn.Linear(hidden_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def _rel_agg(self, h: Tensor, edge_index: Tensor, edge_type: Tensor, num_edges: int) -> Tensor:
        """Relation-wise node->edge->node mean aggregation; returns list per rel."""
        node_idx, edge_idx = edge_index[0], edge_index[1]
        outs = []
        for r in range(N_REL):
            # mask incidence pairs whose hyperedge is of relation r
            emask = (edge_type[edge_idx] == r)
            if emask.sum() == 0:
                outs.append(h.new_zeros(h.shape))
                continue
            n_i, e_i = node_idx[emask], edge_idx[emask]
            edge_repr = _scatter_add(h[n_i], e_i, num_edges)
            edge_deg = h.new_zeros(num_edges).scatter_add_(0, e_i, h.new_ones(n_i.shape[0]))
            edge_repr = edge_repr / (edge_deg.unsqueeze(-1) + 1e-6)
            node_msg = _scatter_add(edge_repr[e_i], n_i, h.shape[0])
            node_deg = h.new_zeros(h.shape[0]).scatter_add_(0, n_i, h.new_ones(n_i.shape[0]))
            node_msg = node_msg / (node_deg.unsqueeze(-1) + 1e-6)
            outs.append(node_msg)
        return outs

    def forward(self, h_data, story_to_burst_edges, story_ids) -> Tensor:
        h = F.relu(self.in_proj(h_data.x))
        E = int(h_data.edge_type.numel())
        m = self._rel_agg(h, h_data.edge_index, h_data.edge_type, E)
        h = F.relu(self.self1(h) + sum(self.rel1[r](m[r]) for r in range(N_REL)))
        m = self._rel_agg(h, h_data.edge_index, h_data.edge_type, E)
        h = F.relu(self.self2(h) + sum(self.rel2[r](m[r]) for r in range(N_REL)))

        node_idx, edge_idx = h_data.edge_index[0], h_data.edge_index[1]
        out = []
        for sid in story_ids:
            burst = story_to_burst_edges.get(sid, [])
            if not burst:
                out.append(h.new_zeros(h.shape[1])); continue
            mask = torch.zeros_like(edge_idx, dtype=torch.bool)
            for be in burst:
                mask |= (edge_idx == be)
            member_nodes = node_idx[mask]
            out.append(h[member_nodes].mean(0) if member_nodes.numel() else h.new_zeros(h.shape[1]))
        return self.cls(torch.stack(out))


def train_rgcn(
    seed: int,
    data_cfg: DataConfig,
    hidden_dim: int = 64,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict:
    _set_seed(seed)
    device = device or torch.device("cpu")
    cascades = _load_cascades(data_cfg, seed)
    tr, va, te = _stratified_split(cascades, data_cfg.split_ratios, seed)
    h_tr = build_hypergraph(tr, feature_dim=16, seed=seed)
    h_te = build_hypergraph(te, feature_dim=16, seed=seed + 2)
    from ..devices import move_hdata
    move_hdata(h_tr, device); move_hdata(h_te, device)
    y_tr, y_te = h_tr.story_labels, h_te.story_labels

    model = _RGCN(in_dim=16, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        logits = model(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward(); opt.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [rgcn ep {ep+1}/{epochs}] ce={loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        preds = model(h_te, h_te.story_to_burst_edges, h_te.story_ids).argmax(-1).cpu().numpy()
    labels = y_te.cpu().numpy()
    return {
        "seed": seed, "tau": float("inf"), "coverage": 1.0,
        "acc_all": float((preds == labels).mean()), "acc_kept": float((preds == labels).mean()),
        "f1_all": macro_f1(preds, labels), "f1_kept": macro_f1(preds, labels),
        "n_test": int(labels.shape[0]), "method": "rgcn",
    }
