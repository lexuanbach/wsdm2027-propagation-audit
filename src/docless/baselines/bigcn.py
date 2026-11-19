"""BiGCN (Bian et al. 2020) — bidirectional GCN over propagation trees.

The original BiGCN models rumor propagation with two GCN branches: one
over the top-down (root→leaf) tree and one over the bottom-up (leaf→root)
reverse tree, concatenating both for classification.

We adapt it to our hypergraph infrastructure: the "top-down" branch
operates on share+burst edges (early propagation direction), and the
"bottom-up" branch operates on reply+coexpose edges (response direction).
Both branches use mean-aggregating GCN layers identical to our GCNFN
baseline.  Story-level representations are the concatenation of mean-pooled
node embeddings from both branches.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..data.schema import Cascade
from ..eval.stats import macro_f1
from ..hypergraph import build_hypergraph
from ..hypergraph.build import EdgeType
from ..hypergraph.layers import _scatter_add
from ..train import DataConfig, _load_cascades, _set_seed, _stratified_split


class _BiPropGCN(nn.Module):
    """Bidirectional GCN: one branch per propagation direction."""

    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        # top-down branch (share + burst)
        self.td_l1 = nn.Linear(hidden_dim, hidden_dim)
        self.td_l2 = nn.Linear(hidden_dim, hidden_dim)
        # bottom-up branch (reply + coexpose)
        self.bu_l1 = nn.Linear(hidden_dim, hidden_dim)
        self.bu_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim * 2, num_classes)

    @staticmethod
    def _agg(h: Tensor, edge_index: Tensor, num_edges: int) -> Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        node_idx, edge_idx = edge_index[0], edge_index[1]
        edge_repr = _scatter_add(h[node_idx], edge_idx, num_edges)
        edge_deg = h.new_zeros(num_edges).scatter_add_(
            0, edge_idx, h.new_ones(node_idx.shape[0])
        )
        edge_repr = edge_repr / (edge_deg.unsqueeze(-1) + 1e-6)
        node_msg = _scatter_add(edge_repr[edge_idx], node_idx, h.shape[0])
        node_deg = h.new_zeros(h.shape[0]).scatter_add_(
            0, node_idx, h.new_ones(node_idx.shape[0])
        )
        return node_msg / (node_deg.unsqueeze(-1) + 1e-6)

    @staticmethod
    def _split_edges(h_data, direction: str):
        edge_index = h_data.edge_index
        edge_type = h_data.edge_type
        edge_idx = edge_index[1]
        if direction == "td":
            edge_mask = (edge_type == EdgeType.SHARE) | (edge_type == EdgeType.BURST)
        else:
            edge_mask = (edge_type == EdgeType.REPLY) | (edge_type == EdgeType.COEXPOSE)
        incidence_mask = edge_mask[edge_idx]
        if not incidence_mask.any():
            return edge_index.new_zeros((2, 0)), 0
        sub_nodes = edge_index[0, incidence_mask]
        _, new_eidx = edge_idx[incidence_mask].unique(return_inverse=True)
        n_edges = int(new_eidx.max().item()) + 1
        sub_ei = torch.stack([sub_nodes, new_eidx])
        return sub_ei, n_edges

    def forward(self, h_data, story_to_burst_edges, story_ids) -> Tensor:
        h = F.relu(self.in_proj(h_data.x))

        td_ei, td_ne = self._split_edges(h_data, "td")
        h_td = h + F.relu(self.td_l1(self._agg(h, td_ei, max(td_ne, 1))))
        h_td = h_td + F.relu(self.td_l2(self._agg(h_td, td_ei, max(td_ne, 1))))

        bu_ei, bu_ne = self._split_edges(h_data, "bu")
        h_bu = h + F.relu(self.bu_l1(self._agg(h, bu_ei, max(bu_ne, 1))))
        h_bu = h_bu + F.relu(self.bu_l2(self._agg(h_bu, bu_ei, max(bu_ne, 1))))

        edge_index = h_data.edge_index
        node_idx, edge_idx = edge_index[0], edge_index[1]
        out = []
        for sid in story_ids:
            burst = story_to_burst_edges.get(sid, [])
            if not burst:
                out.append(torch.zeros(h_td.shape[1] + h_bu.shape[1], device=h.device))
                continue
            mask = torch.zeros_like(edge_idx, dtype=torch.bool)
            for be in burst:
                mask |= (edge_idx == be)
            members = node_idx[mask]
            if members.numel() == 0:
                out.append(torch.zeros(h_td.shape[1] + h_bu.shape[1], device=h.device))
            else:
                out.append(torch.cat([h_td[members].mean(0), h_bu[members].mean(0)]))
        return self.cls(torch.stack(out))


def train_bigcn(
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
    move_hdata(h_tr, device)
    move_hdata(h_te, device)

    y_tr = h_tr.story_labels
    y_te = h_te.story_labels

    model = _BiPropGCN(in_dim=16, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids)
        loss = F.cross_entropy(logits, y_tr)
        loss.backward()
        opt.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [bigcn ep {ep+1}/{epochs}] ce={loss.item():.3f}")

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
        "method": "bigcn",
    }
