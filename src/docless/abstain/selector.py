"""Disagreement-triggered abstention (paper §IV-D, Eq. 8).

Δ(s) = TV(f_do(s), f_obs(s))      total-variation distance
σ(s) = 1{Δ(s) ≤ τ}                with τ calibrated on val to satisfy
                                  coverage budget c (Eq. 3).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def tv_disagreement(logits_a: Tensor, logits_b: Tensor) -> Tensor:
    """Total-variation distance between two softmax distributions."""
    pa = logits_a.softmax(-1)
    pb = logits_b.softmax(-1)
    return 0.5 * (pa - pb).abs().sum(-1)                # [B]


def calibrate_threshold(deltas: Tensor, coverage: float) -> float:
    """Pick τ such that the fraction of points with Δ ≤ τ is ≥ coverage.

    A bisection over the empirical quantile; with `coverage = c`, τ is the
    `c`-quantile of `Δ` on the validation set.
    """
    arr = deltas.detach().cpu().numpy()
    if arr.size == 0:
        return float("inf")
    coverage = float(np.clip(coverage, 0.0, 1.0))
    return float(np.quantile(arr, coverage))


def selective_predict(
    logits_obs: Tensor,
    logits_do: Tensor,
    tau: float,
) -> tuple[Tensor, Tensor]:
    """Return (predictions, kept_mask).

    Predictions are taken from the interventional classifier f_do (the
    debiased one); abstention is decided by Δ vs. τ.
    """
    delta = tv_disagreement(logits_obs, logits_do)
    keep = (delta <= tau)
    preds = logits_do.argmax(-1)
    return preds, keep
