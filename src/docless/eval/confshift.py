"""Diagnostic stratum-shift runner — paper Table V shape.

Two sweeps, both at fixed model/training settings:
  1. (K_P, K_E)  — bin counts for popularity/hash-concentration strata.
                   The paper's sensitivity bound predicts diminishing returns
                   above K=4.
  2. p(P, E)     — reweight the test set so the empirical stratum marginal
                   shifts to a target distribution. We instantiate one shift:
                   `pop_low` (down-weight high-popularity bins by 2×). The
                   This is an exploratory sensitivity check, not a causal
                   identification test.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..baselines import train_gcnfn
from ..devices import select_device
from ..train import DataConfig, ModelConfig, TrainConfig, train_one_seed


def run_bin_sweep(datasets: dict[str, DataConfig], seeds: list[int], grid: list[tuple[int, int]],
                  out_dir: Path, pretrain_epochs: int = 4, finetune_epochs: int = 20,
                  hidden_dim: int = 64) -> list[dict]:
    device = select_device()
    rows = []
    for ds_name, dcfg in datasets.items():
        for kp, ke in grid:
            for seed in seeds:
                t = time.time()
                mcfg = ModelConfig(hidden_dim=hidden_dim, kp=kp, ke=ke, num_layers=2)
                tcfg = TrainConfig(pretrain_epochs=pretrain_epochs,
                                    finetune_epochs=finetune_epochs)
                m = train_one_seed(seed, dcfg, mcfg, tcfg, verbose=False, device=device)
                m.update(dataset=ds_name, kp=kp, ke=ke, wall_s=round(time.time() - t, 2))
                rows.append(m)
                print(f"  [{ds_name:>10s} | kp={kp} ke={ke} | seed {seed}] "
                      f"f1={m['f1_all']:.3f} ({m['wall_s']:.1f}s)")
    return rows


def run(datasets, seeds, out_dir: Path,
        grid: list[tuple[int, int]] | None = None,
        pretrain_epochs: int = 2, finetune_epochs: int = 10, hidden_dim: int = 32) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # Single (kp, ke) sensitivity sweep — Table V columns
    if grid is None:
        grid = [(2, 2), (4, 4), (6, 6), (8, 8)]
    rows = run_bin_sweep(datasets, seeds, grid, out_dir,
                          pretrain_epochs=pretrain_epochs,
                          finetune_epochs=finetune_epochs,
                          hidden_dim=hidden_dim)
    (out_dir / "confshift.json").write_text(
        json.dumps({"results": rows, "wall_s_total": time.time() - t0,
                     "grid": grid}, indent=2))

    agg = {}
    for r in rows:
        key = (r["dataset"], r["kp"], r["ke"])
        agg.setdefault(key, []).append(r["f1_all"])
    cells = [
        {"dataset": d, "kp": kp, "ke": ke,
         "f1_mean": round(sum(f) / len(f), 3),
         "f1_std": round(_std(f), 3)}
        for (d, kp, ke), f in agg.items()
    ]
    (out_dir / "confshift_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def _std(xs):
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--pretrain-epochs", type=int, default=2)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--grid", type=int, nargs="+", default=[2, 4, 8],
                    help="K_P=K_E values to sweep")
    ap.add_argument("--out", type=Path, default=Path("outputs/confshift"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    grid = [(k, k) for k in args.grid]
    out = run(ds, args.seeds, args.out, grid=grid,
              pretrain_epochs=args.pretrain_epochs,
              finetune_epochs=args.finetune_epochs,
              hidden_dim=args.hidden_dim)
    print("\nConfounder-shift cells:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>10s} | kp={c['kp']} ke={c['ke']} | "
              f"F1 {c['f1_mean']:.3f} ± {c['f1_std']:.3f}")


if __name__ == "__main__":
    main()
