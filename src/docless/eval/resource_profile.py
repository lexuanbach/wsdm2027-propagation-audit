"""Resource profiling (runtime and memory).

Measures wall-clock time and peak RSS for each method under matched protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch

from ..baselines import train_gcnfn, train_majority, train_upfd_prop
from ..baselines.bigcn import train_bigcn
from ..devices import select_device
from ..train import DataConfig, ModelConfig, TrainConfig, train_one_seed


def _get_peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / (1024 * 1024)


def _docless_runner(seed, data_cfg, **kw):
    m = train_one_seed(
        seed=seed,
        data_cfg=data_cfg,
        model_cfg=ModelConfig(
            hidden_dim=kw.get("hidden_dim", 64),
            num_layers=2,
        ),
        train_cfg=TrainConfig(
            pretrain_epochs=kw.get("pretrain_epochs", 4),
            finetune_epochs=kw.get("finetune_epochs", 25),
        ),
        verbose=False,
        device=kw.get("device"),
    )
    m["method"] = "docless"
    return m


METHODS = {
    "majority": train_majority,
    "gcnfn": train_gcnfn,
    "upfd_prop": train_upfd_prop,
    "bigcn": train_bigcn,
    "docless": _docless_runner,
}


def run(
    datasets: dict[str, DataConfig],
    methods: list[str],
    seeds: list[int],
    out_dir: Path,
    hidden_dim: int = 64,
    epochs_short: int = 25,
    epochs_docless_pre: int = 4,
    epochs_docless_ft: int = 25,
) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds_name, dcfg in datasets.items():
        for method in methods:
            for seed in seeds:
                if method == "docless":
                    kw = dict(device=device, hidden_dim=hidden_dim,
                              pretrain_epochs=epochs_docless_pre,
                              finetune_epochs=epochs_docless_ft)
                elif method == "majority":
                    kw = {}
                else:
                    kw = dict(device=device, hidden_dim=hidden_dim, epochs=epochs_short)

                t0 = time.time()
                m = METHODS[method](seed, dcfg, **kw)
                wall_s = round(time.time() - t0, 2)
                peak_mb = round(_get_peak_rss_mb(), 1)

                rows.append({
                    "dataset": ds_name,
                    "method": method,
                    "seed": seed,
                    "wall_s": wall_s,
                    "peak_rss_mb": peak_mb,
                    "f1_all": m["f1_all"],
                })
                print(f"  [{ds_name:>10s} | {method:>10s} | seed {seed}] "
                      f"wall={wall_s:.1f}s peak_rss={peak_mb:.0f}MB f1={m['f1_all']:.3f}")

    (out_dir / "resource_profile.json").write_text(
        json.dumps({"results": rows}, indent=2)
    )

    # Per-method aggregate
    agg = {}
    for r in rows:
        key = (r["dataset"], r["method"])
        agg.setdefault(key, {"wall": [], "rss": []})
        agg[key]["wall"].append(r["wall_s"])
        agg[key]["rss"].append(r["peak_rss_mb"])

    import numpy as np
    cells = [
        {"dataset": d, "method": m,
         "wall_mean": round(float(np.mean(v["wall"])), 1),
         "wall_std": round(float(np.std(v["wall"], ddof=1)), 1) if len(v["wall"]) > 1 else 0.0,
         "peak_rss_mb": round(float(np.max(v["rss"])), 1),
         "n_seeds": len(v["wall"])}
        for (d, m), v in agg.items()
    ]
    (out_dir / "resource_profile_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--methods", nargs="+", default=["majority", "gcnfn", "bigcn", "upfd_prop", "docless"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--epochs-short", type=int, default=25)
    ap.add_argument("--epochs-docless-pre", type=int, default=4)
    ap.add_argument("--epochs-docless-ft", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("outputs/r7/resource_profile"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.methods, args.seeds, args.out,
              hidden_dim=args.hidden_dim,
              epochs_short=args.epochs_short,
              epochs_docless_pre=args.epochs_docless_pre,
              epochs_docless_ft=args.epochs_docless_ft)
    print("\nCells:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>10s} | {c['method']:>10s} | "
              f"wall {c['wall_mean']:.1f}s  rss {c['peak_rss_mb']:.0f}MB")


if __name__ == "__main__":
    main()
