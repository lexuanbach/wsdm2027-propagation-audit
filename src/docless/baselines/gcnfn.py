"""GCNFN-style comparator for the event-list diagnostic.

This local adaptation uses two residual node-to-hyperedge-to-node mean passes
and a burst-member mean readout. It does not implement the upstream ChebConv
operator or establish fidelity to a particular official GCNFN revision. No
text encoder is instantiated; this is not a verified ablation of that encoder.
See IMPLEMENTATION-FIDELITY.md for source and historical provenance limits.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.schema import Cascade
from ..eval.stats import macro_f1
from ..hypergraph import build_hypergraph
from ..hypergraph.build import EdgeType, NodeType
from ..hypergraph.layers import _scatter_add
from ..train import DataConfig, _load_cascades, _set_seed, _stratified_split


class _PropGCN(nn.Module):
    """Two-layer GCN with mean aggregation over hyperedges (treated as cliques).

    Each hyperedge becomes a soft clique: node embeddings inside an edge get
    averaged, mixed with the residual node embedding, and re-scattered back.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def _agg(self, h: Tensor, edge_index: Tensor, num_edges: int) -> Tensor:
        """Hyperedge mean-aggregation → broadcast back to nodes."""
        node_idx, edge_idx = edge_index[0], edge_index[1]
        # node -> edge mean
        edge_repr = _scatter_add(h[node_idx], edge_idx, num_edges)
        # divide by edge degree
        edge_deg = h.new_zeros(num_edges).scatter_add_(0, edge_idx, h.new_ones(node_idx.shape[0]))
        edge_repr = edge_repr / (edge_deg.unsqueeze(-1) + 1e-6)
        # edge -> node mean
        node_msg = _scatter_add(edge_repr[edge_idx], node_idx, h.shape[0])
        node_deg = h.new_zeros(h.shape[0]).scatter_add_(0, node_idx, h.new_ones(node_idx.shape[0]))
        node_msg = node_msg / (node_deg.unsqueeze(-1) + 1e-6)
        return node_msg

    def forward(self, h_data, story_to_burst_edges, story_ids) -> Tensor:
        h = F.relu(self.in_proj(h_data.x))
        E = int(h_data.edge_type.numel())
        h = h + F.relu(self.l1(self._agg(h, h_data.edge_index, E)))
        h = h + F.relu(self.l2(self._agg(h, h_data.edge_index, E)))

        edge_index = h_data.edge_index
        node_idx, edge_idx = edge_index[0], edge_index[1]
        out = []
        for sid in story_ids:
            burst = story_to_burst_edges.get(sid, [])
            if not burst:
                out.append(h.new_zeros(h.shape[1]))
                continue
            mask = torch.zeros_like(edge_idx, dtype=torch.bool)
            for be in burst:
                mask |= (edge_idx == be)
            member_nodes = node_idx[mask]
            if member_nodes.numel() == 0:
                out.append(h.new_zeros(h.shape[1]))
            else:
                out.append(h[member_nodes].mean(0))
        story_repr = torch.stack(out)
        return self.cls(story_repr)


def train_gcnfn(
    seed: int,
    data_cfg: DataConfig,
    hidden_dim: int = 64,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    device: torch.device | None = None,
    verbose: bool = False,
) -> dict:
    _set_seed(seed)
    if device is None:
        device = torch.device("cpu")

    cascades = _load_cascades(data_cfg, seed)
    tr, va, te = _stratified_split(cascades, data_cfg.split_ratios, seed)
    h_tr = build_hypergraph(tr, feature_dim=16, seed=seed)
    h_te = build_hypergraph(te, feature_dim=16, seed=seed + 2)

    from ..devices import move_hdata
    move_hdata(h_tr, device); move_hdata(h_te, device)

    y_tr = h_tr.story_labels
    y_te = h_te.story_labels

    model = _PropGCN(in_dim=16, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [gcnfn ep {ep+1}/{epochs}] ce={loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        logits_te = model(h_te, h_te.story_to_burst_edges, h_te.story_ids)
    preds = logits_te.argmax(-1).cpu().numpy()
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
        "method": "gcnfn",
    }
