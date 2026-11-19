"""Composed DocLessFake module.

forward returns a dict with everything Algorithm 1 needs:
    {
        node_repr,                  # [N, H]
        story_repr,                 # [B, H] (mean-pool of burst-slice members)
        logits_obs, logits_do,      # [B, C]
        delta,                      # [B] TV disagreement
    }
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .abstain.selector import tv_disagreement
from .causal.head import DocLessHead, observational_and_interventional
from .hypergraph import HData, HGNNEncoder


class DocLessFake(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        kp: int = 4,
        ke: int = 4,
        num_classes: int = 2,
        use_summary: bool = True,
        summary_dim: int = 7,
        use_structure: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = HGNNEncoder(in_dim, hidden_dim, num_layers, num_heads, dropout)
        self.head = DocLessHead(hidden_dim, kp, ke, num_classes)
        self.kp, self.ke = kp, ke
        # Cascade-summary fusion: a learned residual branch that injects the
        # global text-free summary vector into the structural story readout, so
        # the head sees both the hypergraph-propagated representation and the
        # aggregate cascade statistics. use_structure=False isolates the summary
        # branch (a learned analogue of the PropMLP baseline) for ablation.
        self.use_summary = use_summary
        self.use_structure = use_structure
        self.summary_proj = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def story_pool(self, node_repr: Tensor, h: HData) -> Tensor:
        """Mean-pool node representations over each story's burst edges.

        Story i is represented by the average of its cascade-slice node
        embeddings; this is the structural readout from the hypergraph,
        independent of any text.
        """
        story_ids = h.story_ids
        if not story_ids:
            return node_repr.new_zeros((0, node_repr.shape[1]))

        # We pool over the cascade-slice nodes that belong to each story's
        # burst edges. Recover the slice membership from edge_index.
        edge_index = h.edge_index
        if edge_index.numel() == 0:
            return node_repr.new_zeros((len(story_ids), node_repr.shape[1]))

        node_idx, edge_idx = edge_index[0], edge_index[1]

        out = []
        for sid in story_ids:
            burst_edges = h.story_to_burst_edges.get(sid, [])
            if not burst_edges:
                out.append(node_repr.new_zeros(node_repr.shape[1]))
                continue
            mask = torch.zeros_like(edge_idx, dtype=torch.bool)
            for be in burst_edges:
                mask |= (edge_idx == be)
            member_nodes = node_idx[mask]
            if member_nodes.numel() == 0:
                out.append(node_repr.new_zeros(node_repr.shape[1]))
            else:
                out.append(node_repr[member_nodes].mean(0))
        return torch.stack(out)

    def forward(
        self,
        h: HData,
        p_bin: Tensor,
        e_bin: Tensor,
        q_hat: Tensor,
    ) -> dict[str, Tensor]:
        node_repr = self.encoder(h.x, h.edge_index, h.edge_type, h.node_type)
        story_repr = self.story_pool(node_repr, h)
        if not self.use_structure:
            story_repr = torch.zeros_like(story_repr)
        if self.use_summary and getattr(h, "story_summary", None) is not None:
            story_repr = story_repr + self.summary_proj(h.story_summary)
        f_obs, f_do = observational_and_interventional(
            self.head, story_repr, p_bin, e_bin, q_hat
        )
        return {
            "node_repr": node_repr,
            "story_repr": story_repr,
            "logits_obs": f_obs,
            "logits_do": f_do,
            "delta": tv_disagreement(f_obs, f_do),
        }
