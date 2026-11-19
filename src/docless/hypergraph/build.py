"""Build the heterogeneous hypergraph (paper §III-A, §IV-A).

Hyperedge types (legacy names retained in code):
  share     all event proxies linked to the same URL within window w
  reply     events sharing a release-specific discussion proxy (URL within 5 min)
  coexpose  event proxies linked to the same media object
  burst    all slices of one story (one release-induced burst edge per story)

Large hyperedges (|e| > k_max) are decomposed by stratified sampling on proxy
log-degree. This preserves the implemented popularity summary; it does not
establish a confounder.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from math import log1p
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from ..data.schema import Cascade, Event


class NodeType(IntEnum):
    USER = 0
    URL = 1
    MEDIA = 2
    CASCADE_SLICE = 3


class EdgeType(IntEnum):
    SHARE = 0
    REPLY = 1
    COEXPOSE = 2
    BURST = 3


@dataclass
class HData:
    """Tensor container for a batched hypergraph.

    All node ids are GLOBAL (concatenation order: USER, URL, MEDIA, CASCADE_SLICE).
    """
    x: Tensor                       # [N_total, F]
    node_type: Tensor               # [N_total]      values in NodeType
    edge_index: Tensor              # [2, nnz]       (node, hyperedge)
    edge_type: Tensor               # [n_edges]      values in EdgeType
    num_nodes_per_type: dict[NodeType, int] = field(default_factory=dict)
    story_to_burst_edges: dict[str, list[int]] = field(default_factory=dict)
    story_labels: Tensor | None = None
    story_ids: list[str] = field(default_factory=list)
    # Per-story (P, E_hash) stratum is filled by docless.causal.strata.estimate.
    story_p: Tensor | None = None
    story_e: Tensor | None = None
    # Per-story text-free cascade-summary features [num_stories, summary_dim],
    # standardized with TRAIN statistics. Filled in by docless.train; aligned
    # row-for-row with story_ids. None disables the summary-fusion branch.
    story_summary: Tensor | None = None

    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    def num_edges(self) -> int:
        return int(self.edge_type.numel())


# -------- feature extractors (no text fields, ever) -----------------------

_POP_BINS = np.array([0, 1, 3, 10, 30, 100, 300, 1000, 1_000_000], dtype=np.float32)


def _user_features(user_ids: Sequence[int], user_post_counts: dict[int, int]) -> np.ndarray:
    feats = []
    for u in user_ids:
        n = user_post_counts.get(u, 0)
        log_deg = log1p(n)
        # 8-bin one-hot
        bin_idx = int(np.searchsorted(_POP_BINS, n, side="right") - 1)
        bin_idx = max(0, min(bin_idx, 7))
        one_hot = [0.0] * 8
        one_hot[bin_idx] = 1.0
        feats.append([log_deg, *one_hot])
    return np.asarray(feats, dtype=np.float32)


def _url_features(url_ids: Sequence[int], url_share_counts: dict[int, int]) -> np.ndarray:
    feats = []
    for r in url_ids:
        n = url_share_counts.get(r, 0)
        feats.append([log1p(n)])
    return np.asarray(feats, dtype=np.float32)


def _media_features(media_ids: Sequence[int], media_counts: dict[int, int]) -> np.ndarray:
    feats = []
    for m in media_ids:
        n = media_counts.get(m, 0)
        feats.append([log1p(n)])
    return np.asarray(feats, dtype=np.float32)


def _slice_features(slice_event_counts: Sequence[int]) -> np.ndarray:
    return np.asarray([[log1p(n)] for n in slice_event_counts], dtype=np.float32)


# -------- main builder ----------------------------------------------------

def build_hypergraph(
    cascades: list[Cascade],
    delta_minutes: float = 5.0,
    k_max: int = 128,
    feature_dim: int = 16,
    seed: int = 0,
    drop_media: bool = False,
) -> HData:
    """Construct the four-type hypergraph and pack it into an HData tensor blob.

    drop_media=True removes media nodes and the media-mediated coexpose edges,
    to isolate whether media objects contribute structural signal.
    """
    rng = np.random.default_rng(seed)

    # 1. Collect node identifiers and counts.
    user_post_counts: dict[int, int] = defaultdict(int)
    url_share_counts: dict[int, int] = defaultdict(int)
    media_counts: dict[int, int] = defaultdict(int)

    for c in cascades:
        for ev in c.events:
            user_post_counts[ev.user_id] += 1
            url_share_counts[ev.url_id] += 1
            if ev.media_id >= 0 and not drop_media:
                media_counts[ev.media_id] += 1

    user_ids = sorted(user_post_counts)
    url_ids = sorted(url_share_counts)
    media_ids = sorted(media_counts)

    user_idx = {u: i for i, u in enumerate(user_ids)}
    url_idx = {r: i for i, r in enumerate(url_ids)}
    media_idx = {m: i for i, m in enumerate(media_ids)}

    # 2. Build cascade slices per story (binning by Δ).
    delta = delta_minutes * 60.0
    slice_event_counts: list[int] = []
    slice_global: list[int] = []          # global slice node id
    story_to_slices: dict[str, list[int]] = {}

    for c in cascades:
        s_ids: list[int] = []
        if not c.events:
            story_to_slices[c.story_id] = s_ids
            continue
        t0 = c.events[0].timestamp
        cur_bin = -1
        cur_count = 0
        for ev in c.events:
            b = int((ev.timestamp - t0) // delta)
            if b != cur_bin:
                if cur_bin >= 0:
                    slice_event_counts.append(cur_count)
                cur_bin = b
                cur_count = 1
                s_ids.append(len(slice_event_counts))   # the index we'll write next
            else:
                cur_count += 1
        slice_event_counts.append(cur_count)
        story_to_slices[c.story_id] = s_ids

    n_user, n_url, n_media, n_slice = len(user_ids), len(url_ids), len(media_ids), len(slice_event_counts)
    off_user, off_url, off_media, off_slice = 0, n_user, n_user + n_url, n_user + n_url + n_media
    n_total = off_slice + n_slice

    # 3. Node features (padded to feature_dim).
    xu = _user_features(user_ids, user_post_counts)
    xr = _url_features(url_ids, url_share_counts)
    xm = _media_features(media_ids, media_counts)
    xs = _slice_features(slice_event_counts)

    def _pad(a: np.ndarray, d: int) -> np.ndarray:
        if a.shape[1] >= d:
            return a[:, :d]
        out = np.zeros((a.shape[0], d), dtype=np.float32)
        out[:, : a.shape[1]] = a
        return out

    x = np.concatenate([
        _pad(xu, feature_dim) if n_user else np.zeros((0, feature_dim), np.float32),
        _pad(xr, feature_dim) if n_url else np.zeros((0, feature_dim), np.float32),
        _pad(xm, feature_dim) if n_media else np.zeros((0, feature_dim), np.float32),
        _pad(xs, feature_dim) if n_slice else np.zeros((0, feature_dim), np.float32),
    ], axis=0)
    node_type = np.concatenate([
        np.full(n_user, NodeType.USER, dtype=np.int64),
        np.full(n_url, NodeType.URL, dtype=np.int64),
        np.full(n_media, NodeType.MEDIA, dtype=np.int64),
        np.full(n_slice, NodeType.CASCADE_SLICE, dtype=np.int64),
    ])

    # 4. Hyperedges.
    edges: list[tuple[list[int], EdgeType]] = []
    story_to_burst_edges: dict[str, list[int]] = {}

    # share: per (url, window) bucket
    share_buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for c in cascades:
        for ev in c.events:
            w = int(ev.timestamp // (delta * 4))  # 4 Δ-slices wide
            share_buckets[(ev.url_id, w)].add(ev.user_id)
    for (r, _), members in share_buckets.items():
        m = [off_user + user_idx[u] for u in members] + [off_url + url_idx[r]]
        edges.append((_decompose(m, k_max, rng), EdgeType.SHARE))

    # reply: the audited event-list releases do not expose thread ids, so the
    # implementation approximates a discussion as event proxies sharing one URL
    # within one Δ-bin.
    reply_buckets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for c in cascades:
        for ev in c.events:
            b = int(ev.timestamp // delta)
            reply_buckets[(ev.url_id, b)].add(ev.user_id)
    for (r, _), members in reply_buckets.items():
        if len(members) < 2:
            continue
        m = [off_user + user_idx[u] for u in members]
        edges.append((_decompose(m, k_max, rng), EdgeType.REPLY))

    # coexpose: users who interacted with the same media object
    media_buckets: dict[int, set[int]] = defaultdict(set)
    if not drop_media:
        for c in cascades:
            for ev in c.events:
                if ev.media_id >= 0:
                    media_buckets[ev.media_id].add(ev.user_id)
    for m_id, members in media_buckets.items():
        if len(members) < 2:
            continue
        ms = [off_user + user_idx[u] for u in members] + [off_media + media_idx[m_id]]
        edges.append((_decompose(ms, k_max, rng), EdgeType.COEXPOSE))

    # burst: connect all non-empty cascade slices of a story in one edge.
    for c in cascades:
        sids = story_to_slices.get(c.story_id, [])
        if not sids:
            continue
        burst_ids: list[int] = []
        # The audited implementation uses one burst edge per story; it does not
        # perform spike detection.
        m = [off_slice + s for s in sids]
        edges.append((_decompose(m, k_max, rng), EdgeType.BURST))
        burst_ids.append(len(edges) - 1)
        story_to_burst_edges[c.story_id] = burst_ids

    # 5. Pack edge_index in (node, edge) format.
    src, dst, et = [], [], []
    for e_idx, (members, etype) in enumerate(edges):
        for m_node in members:
            src.append(m_node)
            dst.append(e_idx)
        et.append(int(etype))

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    edge_type = torch.tensor(et, dtype=torch.long) if et else torch.zeros((0,), dtype=torch.long)

    labels = torch.tensor([c.label for c in cascades], dtype=torch.long) if cascades else None

    return HData(
        x=torch.from_numpy(x),
        node_type=torch.from_numpy(node_type),
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes_per_type={
            NodeType.USER: n_user,
            NodeType.URL: n_url,
            NodeType.MEDIA: n_media,
            NodeType.CASCADE_SLICE: n_slice,
        },
        story_to_burst_edges=story_to_burst_edges,
        story_labels=labels,
        story_ids=[c.story_id for c in cascades],
    )


def _decompose(members: list[int], k_max: int, rng: np.random.Generator) -> list[int]:
    """Stratified-sample hyperedges with more than k_max members down to k_max."""
    if len(members) <= k_max:
        return members
    return list(rng.choice(members, size=k_max, replace=False))
