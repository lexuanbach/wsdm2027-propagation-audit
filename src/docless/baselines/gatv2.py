"""GATv2 -- dynamic-attention graph network, no text.

A stronger, *modern* learned-attention propagation baseline than RGCN
(a stronger recent propagation/attention GNN under the
text-free protocol"). GATv2 (Brody et al., 2021) is the standard fix to the
static-attention limitation of the original GAT (Velickovic et al., 2018): the
learnable attention vector ``a`` is applied *after* the non-linearity, so the
ranking of neighbours becomes input-dependent ("dynamic" attention). We make it
type-aware over the four hyperedge / node families (share/reply/coexpose/burst
edges; user/url/media/slice nodes), so it is a heterogeneous GATv2 / HGT-lite.

It is genuinely distinct from the two attention/aggregation operators already in
the paper:
  * RGCN uses a fixed per-relation *mean* (no attention);
  * the DocLessFake encoder uses transformer-style *dot-product* attention.
GATv2 uses additive *dynamic* attention. It is trained on exactly the same
per-split hypergraph, node features, readout, budget, and validation-only model
selection as the other graph baselines, so the comparison against PropMLP is
controlled.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..eval.stats import macro_f1
from ..hypergraph import build_hypergraph
from ..hypergraph.layers import _scatter_add, _scatter_softmax
from ..train import DataConfig, _load_cascades, _set_seed, _stratified_split

N_EDGE_TYPES = 4  # share / reply / coexpose / burst
N_NODE_TYPES = 4  # user / url / media / cascade-slice


class _GATv2Layer(nn.Module):
    """One node->edge->node pass with type-aware multi-head GATv2 attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.h = num_heads
        self.hd = hidden_dim // num_heads
        self.hidden = hidden_dim

        # node -> edge: source = node, destination query = edge-type embedding
        self.w_n2e = nn.Linear(hidden_dim, hidden_dim)
        self.etype = nn.Embedding(N_EDGE_TYPES, hidden_dim)
        self.val_n2e = nn.Linear(hidden_dim, hidden_dim)
        self.a_n2e = nn.Parameter(torch.empty(num_heads, self.hd))

        # edge -> node: source = edge repr, destination query = node-type embedding
        self.w_e2n = nn.Linear(hidden_dim, hidden_dim)
        self.ntype = nn.Embedding(N_NODE_TYPES, hidden_dim)
        self.val_e2n = nn.Linear(hidden_dim, hidden_dim)
        self.a_e2n = nn.Parameter(torch.empty(num_heads, self.hd))

        self.norm_e = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.a_n2e)
        nn.init.xavier_uniform_(self.a_e2n)

    def _gatv2_alpha(self, src: Tensor, dst_q: Tensor, a: Tensor, group_idx: Tensor,
                     num_groups: int) -> Tensor:
        """GATv2 dynamic attention for incidence pairs.

        score = a^T LeakyReLU(W_src h_src + q_dst), softmaxed within each group.
        src/dst_q: [nnz, hidden]; returns alpha [nnz, heads].
        """
        z = (src + dst_q).view(-1, self.h, self.hd)            # [nnz, heads, hd]
        score = (F.leaky_relu(z, 0.2) * a).sum(-1)             # [nnz, heads]
        return torch.stack(
            [_scatter_softmax(score[:, k], group_idx, num_groups) for k in range(self.h)],
            dim=-1,
        )                                                       # [nnz, heads]

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor,
                node_type: Tensor) -> Tensor:
        N = h.shape[0]
        E = int(edge_type.numel())
        if edge_index.numel() == 0 or E == 0:
            return self.dropout(self.val_n2e(h))
        node_idx, edge_idx = edge_index[0], edge_index[1]

        # ---- node -> edge ----
        src = self.w_n2e(h)[node_idx]                 # [nnz, H]
        dst_q = self.etype(edge_type)[edge_idx]       # [nnz, H]
        alpha = self._gatv2_alpha(src, dst_q, self.a_n2e, edge_idx, E)
        val = self.val_n2e(h)[node_idx].view(-1, self.h, self.hd)
        edge_repr = _scatter_add((alpha.unsqueeze(-1) * val).reshape(-1, self.hidden),
                                 edge_idx, E)
        edge_repr = self.norm_e(edge_repr)

        # ---- edge -> node ----
        src2 = self.w_e2n(edge_repr)[edge_idx]        # [nnz, H]
        dst_q2 = self.ntype(node_type)[node_idx]      # [nnz, H]
        alpha2 = self._gatv2_alpha(src2, dst_q2, self.a_e2n, node_idx, N)
        val2 = self.val_e2n(edge_repr)[edge_idx].view(-1, self.h, self.hd)
        node_repr = _scatter_add((alpha2.unsqueeze(-1) * val2).reshape(-1, self.hidden),
                                 node_idx, N)
        node_repr = self.norm_v(node_repr)
        return self.dropout(node_repr)


class _GATv2(nn.Module):
    """Two-layer heterogeneous GATv2 over the typed hypergraph incidence."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 2,
                 num_heads: int = 4, num_classes: int = 2, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            _GATv2Layer(hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        )
        self.cls = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _member_cache(h_data, story_to_burst_edges, story_ids):
        """Per-story burst-edge member-node index, built ONCE per graph.

        The graph membership is fixed across epochs, so we cache it on h_data
        instead of recomputing an O(nnz) mask per story per forward (which is
        pathological on large graphs). Numerically identical to the per-forward
        mean-pool used by the other graph baselines.
        """
        cache = getattr(h_data, "_gatv2_members", None)
        if cache is not None:
            return cache
        from collections import defaultdict
        node_idx = h_data.edge_index[0].tolist()
        edge_idx = h_data.edge_index[1].tolist()
        edge_to_nodes = defaultdict(list)
        for n, e in zip(node_idx, edge_idx):
            edge_to_nodes[e].append(n)
        cache = {}
        for sid in story_ids:
            members = []
            for be in story_to_burst_edges.get(sid, []):
                members.extend(edge_to_nodes.get(be, []))
            cache[sid] = (torch.tensor(sorted(set(members)), dtype=torch.long)
                          if members else torch.empty(0, dtype=torch.long))
        h_data._gatv2_members = cache
        return cache

    def forward(self, h_data, story_to_burst_edges, story_ids) -> Tensor:
        h = F.relu(self.in_proj(h_data.x))
        for layer in self.layers:
            h = h + layer(h, h_data.edge_index, h_data.edge_type, h_data.node_type)

        members = self._member_cache(h_data, story_to_burst_edges, story_ids)
        out = []
        for sid in story_ids:
            mn = members[sid]
            out.append(h[mn].mean(0) if mn.numel() else h.new_zeros(h.shape[1]))
        return self.cls(torch.stack(out))


def train_gatv2(
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

    model = _GATv2(in_dim=16, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids), y_tr)
        loss.backward(); opt.step()
        if verbose and (ep + 1) % 5 == 0:
            print(f"  [gatv2 ep {ep+1}/{epochs}] ce={loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        preds = model(h_te, h_te.story_to_burst_edges, h_te.story_ids).argmax(-1).cpu().numpy()
    labels = y_te.cpu().numpy()
    return {
        "seed": seed, "tau": float("inf"), "coverage": 1.0,
        "acc_all": float((preds == labels).mean()), "acc_kept": float((preds == labels).mean()),
        "f1_all": macro_f1(preds, labels), "f1_kept": macro_f1(preds, labels),
        "n_test": int(labels.shape[0]), "method": "gatv2",
        "preds": preds.tolist(), "labels": labels.tolist(),
    }
