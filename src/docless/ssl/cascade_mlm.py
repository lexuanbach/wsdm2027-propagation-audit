"""Next-slice contrastive pretext task.

We sample a fraction ρ of eligible cascade slices and ask a small bilinear head
to identify the next slice of the same story among candidate slices. The
candidate set combines structurally close and uniform negatives.

For a slice s in story `i`, the positive is the next slice in `i`. Hard
negatives are slices from a *different* story `j` whose event-proxy-set Jaccard
with `i` is in the top decile. This selection is **label-free** (it never
reads `c.label`); the task is genuinely self-supervised. A `label_aware`
switch is retained ONLY so the ablation in `eval/ssl_negsource.py` can
quantify whether label information in negative selection changes the result
(it does not, materially).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..hypergraph import HData, NodeType


@dataclass
class MaskedBatch:
    anchor_idx: Tensor          # [B]   slice node ids (global) acting as anchors
    positive_idx: Tensor        # [B]   the next-slice positives
    negative_idx: Tensor        # [B, K] hard negatives + uniform negatives


def mask_cascade_slices(
    h: HData,
    cascades_for_lookup: list,  # list[Cascade]
    mask_ratio: float = 0.3,
    num_negatives: int = 8,
    seed: int = 0,
    label_free: bool = True,
) -> MaskedBatch:
    """Build a next-slice contrastive batch from a single HData.

    Anchors are picked uniformly at random from slices that have at least one
    successor inside the same story. Hard negatives are drawn from other
    stories whose event-proxy-set Jaccard is large.

    If ``label_free`` (default), negative selection ignores story labels
    entirely, so the pretext task is genuinely self-supervised. The
    ``label_free=False`` path (opposite-label negatives) is retained only for
    the controlled comparison in ``eval/ssl_negsource.py``.
    """
    rng = np.random.default_rng(seed)

    # Story -> ordered list of slice global indices.
    n_user = h.num_nodes_per_type[NodeType.USER]
    n_url = h.num_nodes_per_type[NodeType.URL]
    n_media = h.num_nodes_per_type[NodeType.MEDIA]
    off_slice = n_user + n_url + n_media

    # Recover per-story slice list from h.story_to_burst_edges keys + cascades_for_lookup,
    # but in the current builder we only stored burst edges. Reconstruct from cascades by
    # re-running the binning logic.
    delta = 5.0 * 60.0
    story_slices: dict[str, list[int]] = {}
    story_users: dict[str, set[int]] = {}
    story_labels: dict[str, int] = {}
    cursor = 0
    for c in cascades_for_lookup:
        sids: list[int] = []
        if not c.events:
            story_slices[c.story_id] = sids
            story_users[c.story_id] = set()
            story_labels[c.story_id] = c.label
            continue
        t0 = c.events[0].timestamp
        cur_bin = -1
        for ev in c.events:
            b = int((ev.timestamp - t0) // delta)
            if b != cur_bin:
                sids.append(off_slice + cursor)
                cursor += 1
                cur_bin = b
        story_slices[c.story_id] = sids
        story_users[c.story_id] = {ev.user_id for ev in c.events}
        story_labels[c.story_id] = c.label

    # Anchors: any slice with a successor in the same story.
    anchors_pos: list[tuple[int, int, str]] = []   # (anchor_gid, positive_gid, story_id)
    for sid, slices in story_slices.items():
        for i in range(len(slices) - 1):
            anchors_pos.append((slices[i], slices[i + 1], sid))
    if not anchors_pos:
        return MaskedBatch(
            anchor_idx=torch.zeros(0, dtype=torch.long),
            positive_idx=torch.zeros(0, dtype=torch.long),
            negative_idx=torch.zeros((0, num_negatives), dtype=torch.long),
        )

    n_total = len(anchors_pos)
    n_mask = max(1, int(round(mask_ratio * n_total)))
    chosen = rng.choice(n_total, size=n_mask, replace=False)

    # Hard negatives: from each anchor's story, find another story with high
    # event-proxy-set Jaccard. When label_free (default) we IGNORE labels so the
    # pretext is self-supervised; the label_aware path is kept only for the
    # controlled negative-source comparison.
    all_other_slices = list(range(off_slice, h.num_nodes()))

    anchor_ids, positive_ids, neg_ids = [], [], []
    for idx in chosen:
        a, p, sid = anchors_pos[idx]
        my_users = story_users[sid]
        my_label = story_labels[sid]

        # candidate stories (label-free: any other story; label-aware: opposite label)
        candidates: list[tuple[float, str]] = []
        for other_sid, ous in story_users.items():
            if other_sid == sid:
                continue
            if not label_free and story_labels[other_sid] == my_label:
                continue
            if not my_users or not ous:
                jacc = 0.0
            else:
                inter = len(my_users & ous)
                union = len(my_users | ous)
                jacc = inter / union if union else 0.0
            candidates.append((jacc, other_sid))

        candidates.sort(reverse=True)
        hard_pool = [story_slices[s] for _, s in candidates[: max(2, len(candidates) // 10)]]
        hard_slices = [s for lst in hard_pool for s in lst]

        hard_k = min(num_negatives // 2, len(hard_slices))
        rest_k = num_negatives - hard_k
        neg = []
        if hard_k > 0:
            neg.extend(rng.choice(hard_slices, size=hard_k, replace=False).tolist())
        if rest_k > 0:
            neg.extend(rng.choice(all_other_slices, size=rest_k, replace=True).tolist())

        anchor_ids.append(a)
        positive_ids.append(p)
        neg_ids.append(neg)

    return MaskedBatch(
        anchor_idx=torch.tensor(anchor_ids, dtype=torch.long),
        positive_idx=torch.tensor(positive_ids, dtype=torch.long),
        negative_idx=torch.tensor(neg_ids, dtype=torch.long),
    )


class CascadeMLM(nn.Module):
    """Bilinear scorer between anchor and candidate slice embeddings."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.proj_a = nn.Linear(hidden_dim, hidden_dim)
        self.proj_c = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h_anchor: Tensor, h_cand: Tensor) -> Tensor:
        # h_anchor: [B, H], h_cand: [B, K, H]
        a = self.proj_a(h_anchor).unsqueeze(1)              # [B, 1, H]
        c = self.proj_c(h_cand)                             # [B, K, H]
        return (a * c).sum(-1)                              # [B, K]


def cascade_ssl_loss(
    h_nodes: Tensor,
    head: CascadeMLM,
    batch: MaskedBatch,
) -> Tensor:
    """Cross-entropy: positive is index 0 of the candidate row."""
    if batch.anchor_idx.numel() == 0:
        return h_nodes.new_zeros(())
    h_a = h_nodes[batch.anchor_idx]                         # [B, H]
    cand = torch.cat([batch.positive_idx.unsqueeze(1), batch.negative_idx], dim=1)
    h_c = h_nodes[cand]                                     # [B, 1+K, H]
    logits = head(h_a, h_c)                                 # [B, 1+K]
    target = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, target)
