"""Majority-class baseline: predicts the most common label in train, regardless of input.

This is the strict floor every learned model should beat.
"""
from __future__ import annotations

from collections import Counter

import torch

from ..eval.stats import macro_f1
from ..train import DataConfig, _load_cascades, _set_seed, _stratified_split


def train_majority(seed: int, data_cfg: DataConfig, **_: object) -> dict:
    _set_seed(seed)
    cascades = _load_cascades(data_cfg, seed)
    tr, va, te = _stratified_split(cascades, data_cfg.split_ratios, seed)

    counts = Counter(c.label for c in tr)
    majority = counts.most_common(1)[0][0]
    preds = torch.full((len(te),), int(majority), dtype=torch.long)
    labels = torch.tensor([c.label for c in te], dtype=torch.long)

    return {
        "seed": seed,
        "tau": float("inf"),
        "coverage": 1.0,
        "acc_all": float((preds == labels).float().mean().item()),
        "acc_kept": float((preds == labels).float().mean().item()),
        "f1_all": macro_f1(preds.numpy(), labels.numpy()),
        "f1_kept": macro_f1(preds.numpy(), labels.numpy()),
        "n_test": int(labels.numel()),
        "method": "majority",
    }
