"""Skeleton eval runner. Reproduces Table I rows for DocLessFake on a
single dataset across multiple seeds, emits results as JSON, and computes
paired-bootstrap CIs.

Real baselines + 6 datasets will be wired in via M5/M6; this scaffold is the
shape every downstream runner (xlingual/ablation/temporal/etc) will follow.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .stats import bootstrap_ci


@dataclass
class RunResult:
    method: str
    dataset: str
    regime: str                 # "with_text" | "without_text"
    seeds: list[int]
    f1_per_seed: list[float]
    accuracy_per_seed: list[float]
    coverage_per_seed: list[float]
    notes: str = ""

    def f1_mean(self) -> float:
        return float(np.mean(self.f1_per_seed))

    def f1_std(self) -> float:
        return float(np.std(self.f1_per_seed, ddof=1)) if len(self.f1_per_seed) > 1 else 0.0

    def f1_ci(self) -> tuple[float, float]:
        return bootstrap_ci(np.asarray(self.f1_per_seed))


def run_main_table(
    method: str,
    dataset: str,
    regime: str,
    seeds: list[int],
    train_fn: Callable[[int], dict],
    out_dir: Path,
) -> RunResult:
    """Train + evaluate `method` on `dataset` for each seed. `train_fn(seed)`
    must return a dict with keys {f1_all, acc_all, coverage}.
    """
    f1s, accs, covs = [], [], []
    for s in seeds:
        m = train_fn(s)
        f1s.append(float(m["f1_all"]))
        accs.append(float(m["acc_all"]))
        covs.append(float(m["coverage"]))
    res = RunResult(
        method=method,
        dataset=dataset,
        regime=regime,
        seeds=seeds,
        f1_per_seed=f1s,
        accuracy_per_seed=accs,
        coverage_per_seed=covs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{method}__{dataset}__{regime}.json").write_text(
        json.dumps(asdict(res), indent=2)
    )
    return res
