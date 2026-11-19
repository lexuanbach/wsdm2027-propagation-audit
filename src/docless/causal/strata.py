"""Popularity (P) and hash-bucket concentration (E_hash) strata.

P      = number of distinct event proxies in the early window
E_hash = normalized dominance of the largest of four deterministic ID buckets

The public API retains the legacy `_echo_chamber` function and `E`-shaped keys
for artifact compatibility. The feature is not a community or echo-chamber
measurement.

q̂(p, e) = empirical marginal of (p_bin, e_bin) across the training corpus.
The marginal is estimated on the TRAINING set only and reused at inference.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..data.schema import Cascade


def _popularity(c: Cascade) -> int:
    return len({e.user_id for e in c.early_events()})


def _echo_chamber(c: Cascade, n_partitions: int = 4) -> float:
    """Return normalized early-proxy hash concentration (legacy function name)."""
    early = c.early_events()
    if not early:
        return 0.0

    # Integer proxy IDs map deterministically into k buckets; no graph or
    # community-detection operation is involved.
    user_bucket: dict[int, int] = {}
    bucket_count = np.zeros(n_partitions, dtype=np.float64)
    for ev in early:
        if ev.user_id not in user_bucket:
            user_bucket[ev.user_id] = (hash(ev.user_id) % n_partitions)
        bucket_count[user_bucket[ev.user_id]] += 1
    if bucket_count.sum() == 0:
        return 0.0
    share = bucket_count / bucket_count.sum()
    # Normalize top-bucket dominance from the uniform baseline to [0, 1].
    return float(share.max() - 1.0 / n_partitions) / (1.0 - 1.0 / n_partitions)


def story_p_e_features(cascades: list[Cascade]) -> tuple[np.ndarray, np.ndarray]:
    """Continuous (P, E_hash) per story before binning."""
    p_vals = np.array([_popularity(c) for c in cascades], dtype=np.float64)
    e_vals = np.array([_echo_chamber(c) for c in cascades], dtype=np.float64)
    return p_vals, e_vals


@dataclass
class StratumEstimator:
    """Quantile binner for P and E_hash. Fit only on the training fold."""
    kp: int = 4
    ke: int = 4
    p_edges: np.ndarray | None = None  # length kp+1
    e_edges: np.ndarray | None = None  # length ke+1
    q_hat: np.ndarray | None = None    # [kp, ke], training marginal

    def fit(self, cascades: list[Cascade]) -> "StratumEstimator":
        p_vals, e_vals = story_p_e_features(cascades)
        # Quantile edges; jitter to avoid empty bins on tied data.
        p_q = np.quantile(p_vals, np.linspace(0, 1, self.kp + 1))
        e_q = np.quantile(e_vals, np.linspace(0, 1, self.ke + 1))
        # Make strictly monotonic.
        p_q = _force_monotone(p_q)
        e_q = _force_monotone(e_q)
        self.p_edges, self.e_edges = p_q, e_q

        p_bins = self.bin_p(p_vals)
        e_bins = self.bin_e(e_vals)
        q = np.zeros((self.kp, self.ke), dtype=np.float64)
        for pb, eb in zip(p_bins, e_bins):
            q[pb, eb] += 1.0
        q /= max(q.sum(), 1.0)
        self.q_hat = q
        return self

    def bin_p(self, vals: np.ndarray) -> np.ndarray:
        assert self.p_edges is not None
        return np.clip(
            np.searchsorted(self.p_edges[1:-1], vals, side="right"),
            0,
            self.kp - 1,
        ).astype(np.int64)

    def bin_e(self, vals: np.ndarray) -> np.ndarray:
        assert self.e_edges is not None
        return np.clip(
            np.searchsorted(self.e_edges[1:-1], vals, side="right"),
            0,
            self.ke - 1,
        ).astype(np.int64)

    def transform(self, cascades: list[Cascade]) -> tuple[Tensor, Tensor]:
        p_vals, e_vals = story_p_e_features(cascades)
        p_bins = self.bin_p(p_vals)
        e_bins = self.bin_e(e_vals)
        return torch.from_numpy(p_bins), torch.from_numpy(e_bins)

    def q_tensor(self) -> Tensor:
        assert self.q_hat is not None
        return torch.from_numpy(self.q_hat.astype(np.float32))


def _force_monotone(edges: np.ndarray) -> np.ndarray:
    """Ensure strictly increasing edges by adding tiny epsilons where needed."""
    out = edges.astype(np.float64).copy()
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-6
    return out
