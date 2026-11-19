"""Hypergraph attention layers (paper §IV-A, Eq. 4).

Two-stage incidence operator:
    h_e^l = φ_e({h_v^{l-1}}_{v ∈ e})        node → hyperedge
    h_v^l = φ_v({h_e^l}_{e ∋ v})            hyperedge → node

Both φ are type-aware multi-head attention. We avoid a torch_geometric
dependency for the core path so the module is testable in plain PyTorch.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _scatter_softmax(logits: Tensor, index: Tensor, num_groups: int) -> Tensor:
    """Numerically stable softmax over groups defined by `index`.

    scatter_reduce(reduce='amax') has poor MPS coverage on 2.x; we route
    the max op through CPU when running on MPS, which is still fast for the
    small hyperedges we build (≤ k_max per edge).
    """
    if logits.device.type == "mps":
        # Round-trip through CPU just for the amax reduction.
        lg_cpu = logits.detach().cpu()
        ix_cpu = index.detach().cpu()
        gmax_cpu = lg_cpu.new_full((num_groups,), -1e30)
        gmax_cpu = gmax_cpu.scatter_reduce(0, ix_cpu, lg_cpu, reduce="amax", include_self=True)
        g_max = gmax_cpu.to(logits.device)
    else:
        g_max = logits.new_full((num_groups,), -1e30)
        g_max = g_max.scatter_reduce(0, index, logits, reduce="amax", include_self=True)
    centered = logits - g_max[index]
    exp = centered.exp()
    g_sum = logits.new_zeros(num_groups).scatter_add_(0, index, exp)
    return exp / (g_sum[index] + 1e-12)


def _scatter_add(values: Tensor, index: Tensor, num_groups: int) -> Tensor:
    """Sum `values` (shape [nnz, F]) into groups along dim 0 indexed by `index`."""
    out = values.new_zeros((num_groups, values.shape[-1]))
    return out.scatter_add_(0, index.unsqueeze(-1).expand_as(values), values)


class HypergraphAttn(nn.Module):
    """One HGNN layer with type-aware multi-head attention on both passes."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_node_types: int = 4,
        num_edge_types: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden = hidden_dim

        # node → edge
        self.q_e = nn.Embedding(num_edge_types, hidden_dim)
        self.k_v = nn.Linear(in_dim, hidden_dim)
        self.v_v = nn.Linear(in_dim, hidden_dim)

        # edge → node
        self.q_v = nn.Embedding(num_node_types, hidden_dim)
        self.k_e = nn.Linear(hidden_dim, hidden_dim)
        self.v_e = nn.Linear(hidden_dim, hidden_dim)

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_e = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,                  # [N, F]
        edge_index: Tensor,         # [2, nnz]: row 0 = node, row 1 = edge
        edge_type: Tensor,          # [E]
        node_type: Tensor,          # [N]
    ) -> Tensor:
        N = x.shape[0]
        E = int(edge_type.numel())
        if edge_index.numel() == 0 or E == 0:
            return self.dropout(self.proj(self.k_v(x)))

        node_idx, edge_idx = edge_index[0], edge_index[1]

        # ---- node → edge ----
        q = self.q_e(edge_type)[edge_idx]                # [nnz, H]
        k = self.k_v(x)[node_idx]                        # [nnz, H]
        v = self.v_v(x)[node_idx]                        # [nnz, H]

        q = q.view(-1, self.heads, self.head_dim)
        k = k.view(-1, self.heads, self.head_dim)
        v = v.view(-1, self.heads, self.head_dim)

        score = (q * k).sum(-1) / math.sqrt(self.head_dim)   # [nnz, heads]
        alpha = torch.stack(
            [_scatter_softmax(score[:, h], edge_idx, E) for h in range(self.heads)],
            dim=-1,
        )  # [nnz, heads]
        edge_repr = _scatter_add(
            (alpha.unsqueeze(-1) * v).reshape(-1, self.hidden),
            edge_idx,
            E,
        )                                                  # [E, H]
        edge_repr = self.norm_e(edge_repr)

        # ---- edge → node ----
        q2 = self.q_v(node_type)[node_idx]                # [nnz, H]
        k2 = self.k_e(edge_repr)[edge_idx]                # [nnz, H]
        v2 = self.v_e(edge_repr)[edge_idx]                # [nnz, H]

        q2 = q2.view(-1, self.heads, self.head_dim)
        k2 = k2.view(-1, self.heads, self.head_dim)
        v2 = v2.view(-1, self.heads, self.head_dim)

        score2 = (q2 * k2).sum(-1) / math.sqrt(self.head_dim)  # [nnz, heads]
        alpha2 = torch.stack(
            [_scatter_softmax(score2[:, h], node_idx, N) for h in range(self.heads)],
            dim=-1,
        )
        node_repr = _scatter_add(
            (alpha2.unsqueeze(-1) * v2).reshape(-1, self.hidden),
            node_idx,
            N,
        )
        node_repr = self.norm_v(node_repr)
        return self.dropout(self.proj(node_repr))


class HGNNEncoder(nn.Module):
    """Stack of L hypergraph attention layers + residual."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            HypergraphAttn(
                in_dim=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_type: Tensor,
    ) -> Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = h + layer(h, edge_index, edge_type, node_type)
        return h
