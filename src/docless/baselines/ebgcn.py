"""EBGCN-style edge-enhanced bidirectional GCN (no text).

A faithful CPU reimplementation, in the spirit of EBGCN (Wei et al., 2021), of a
dedicated rumor-detection propagation model, added as a
stronger / more domain-specific comparator than the generic GCNFN/BiGCN/RGCN/GATv2.
Like BiGCN it propagates in two directions -- top-down (share/burst edges) and
bottom-up (reply/coexpose edges) -- and, like EBGCN, it *infers* a per-incidence
edge weight (an edge-enhancement scorer) that reweights each aggregation, with a
consistency gate between the two directions. It uses the SAME per-split
hypergraph, node features, burst readout, and validation-only tuning as the other
graph baselines, so the comparison against PropMLP is controlled.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..eval.stats import macro_f1
from ..hypergraph import build_hypergraph
from ..hypergraph.layers import _scatter_add, _scatter_softmax
from ..train import DataConfig, _load_cascades, _set_seed, _stratified_split

TOPDOWN = {0, 3}    # share, burst
BOTTOMUP = {1, 2}   # reply, coexpose


class _DirConv(nn.Module):
    """One edge-enhanced message-passing pass restricted to a set of edge types."""

    def __init__(self, dim: int, etypes):
        super().__init__()
        self.etypes = etypes
        self.lin = nn.Linear(dim, dim)
        self.escore = nn.Linear(dim, 1)   # edge-enhancement: infer incidence weight

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        node_idx, edge_idx = edge_index[0], edge_index[1]
        E = int(edge_type.numel())
        keep = torch.zeros(E, dtype=torch.bool, device=h.device)
        for t in self.etypes:
            keep |= (edge_type == t)
        mask = keep[edge_idx]
        if mask.sum() == 0:
            return h.new_zeros(h.shape)
        ni, ei = node_idx[mask], edge_idx[mask]
        v = self.lin(h)
        w = self.escore(v[ni]).squeeze(-1)                 # learned per-incidence weight
        alpha = _scatter_softmax(w, ei, E)                 # normalise within each edge
        edge_repr = _scatter_add((alpha.unsqueeze(-1) * v[ni]), ei, E)
        w2 = self.escore(edge_repr[ei]).squeeze(-1)
        beta = _scatter_softmax(w2, ni, h.shape[0])
        node_msg = _scatter_add((beta.unsqueeze(-1) * edge_repr[ei]), ni, h.shape[0])
        return node_msg


class _EBGCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.td1, self.bu1 = _DirConv(hidden_dim, TOPDOWN), _DirConv(hidden_dim, BOTTOMUP)
        self.td2, self.bu2 = _DirConv(hidden_dim, TOPDOWN), _DirConv(hidden_dim, BOTTOMUP)
        self.gate = nn.Parameter(torch.tensor(0.5))        # top-down/bottom-up consistency gate
        self.cls = nn.Linear(2 * hidden_dim, num_classes)

    def forward(self, h_data, story_to_burst_edges, story_ids) -> Tensor:
        h = F.relu(self.in_proj(h_data.x)); ei, et = h_data.edge_index, h_data.edge_type
        td = h + F.relu(self.td1(h, ei, et)); bu = h + F.relu(self.bu1(h, ei, et))
        td = td + F.relu(self.td2(td, ei, et)); bu = bu + F.relu(self.bu2(bu, ei, et))
        g = torch.sigmoid(self.gate)
        node = torch.cat([g * td + (1 - g) * bu, bu], dim=-1)
        node_idx, edge_idx = ei[0], ei[1]; out = []
        for sid in story_ids:
            burst = story_to_burst_edges.get(sid, [])
            if not burst:
                out.append(node.new_zeros(node.shape[1])); continue
            m = torch.zeros_like(edge_idx, dtype=torch.bool)
            for be in burst:
                m |= (edge_idx == be)
            mn = node_idx[m]
            out.append(node[mn].mean(0) if mn.numel() else node.new_zeros(node.shape[1]))
        return self.cls(torch.stack(out))


def train_ebgcn(seed, data_cfg, hidden_dim=64, epochs=40, learning_rate=5e-3, device=None, verbose=False):
    _set_seed(seed); device = device or torch.device("cpu")
    casc = _load_cascades(data_cfg, seed); tr, va, te = _stratified_split(casc, data_cfg.split_ratios, seed)
    h_tr = build_hypergraph(tr, feature_dim=16, seed=seed); h_te = build_hypergraph(te, feature_dim=16, seed=seed + 2)
    from ..devices import move_hdata
    move_hdata(h_tr, device); move_hdata(h_te, device)
    y_tr, y_te = h_tr.story_labels, h_te.story_labels
    model = _EBGCN(16, hidden_dim).to(device); opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids), y_tr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(h_te, h_te.story_to_burst_edges, h_te.story_ids).argmax(-1).cpu().numpy()
    labels = y_te.cpu().numpy()
    return {"seed": seed, "f1_all": macro_f1(preds, labels), "preds": preds.tolist(), "labels": labels.tolist(),
            "method": "ebgcn", "n_test": int(labels.shape[0])}
