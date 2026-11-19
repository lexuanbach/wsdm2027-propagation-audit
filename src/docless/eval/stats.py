"""Statistical machinery used across every eval runner.

paired_bootstrap matches paper Table I/II footnotes (B=10000, paired).
macro_f1 and fpr_fnr_eo_gap match paper Table IV.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedBootstrapResult:
    mean_diff: float
    p_value: float
    ci_lo: float
    ci_hi: float


def paired_bootstrap(
    losses_a: np.ndarray,
    losses_b: np.ndarray,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedBootstrapResult:
    """Paired-bootstrap test on per-claim losses (lower = better).

    Returns mean(loss_b - loss_a) (positive = A wins), a two-sided p-value
    against the null mean_diff=0, and the (1-α) CI.
    """
    a = np.asarray(losses_a, dtype=np.float64)
    b = np.asarray(losses_b, dtype=np.float64)
    assert a.shape == b.shape, "paired bootstrap requires aligned per-claim losses"
    n = a.shape[0]
    if n == 0:
        return PairedBootstrapResult(0.0, 1.0, 0.0, 0.0)

    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    sampled = diff[idx].mean(axis=1)

    mean_diff = float(diff.mean())
    # Two-sided p: fraction of bootstrap means whose sign disagrees with mean_diff,
    # doubled. Centered on zero per the standard bootstrap test.
    centered = sampled - mean_diff
    p = float(2.0 * min(
        (centered >= abs(mean_diff)).mean(),
        (centered <= -abs(mean_diff)).mean(),
    ))
    p = float(np.clip(p, 0.0, 1.0))
    lo, hi = np.quantile(sampled, [alpha / 2, 1 - alpha / 2])
    return PairedBootstrapResult(mean_diff, p, float(lo), float(hi))


def bootstrap_ci(values: np.ndarray, n_resamples: int = 10_000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> float:
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    f1s = []
    for c in range(num_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def fpr_fnr_eo_gap(
    preds: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Per-group FPR/FNR and the equalized-odds gap from paper Eq. 11."""
    out: dict[str, dict[str, float]] = {}
    fpr_values, fnr_values = [], []
    for g in np.unique(groups):
        mask = groups == g
        p, y = preds[mask], labels[mask]
        neg = (y == 0); pos = (y == 1)
        fpr = float((p[neg] == 1).mean()) if neg.any() else 0.0
        fnr = float((p[pos] == 0).mean()) if pos.any() else 0.0
        out[str(g)] = {"fpr": fpr, "fnr": fnr}
        fpr_values.append(fpr)
        fnr_values.append(fnr)
    eo_gap = float(max(max(fpr_values), max(fnr_values)) - min(min(fpr_values), min(fnr_values)))
    return {"per_group": out, "eo_gap": eo_gap}
