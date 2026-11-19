"""Main table runner — reproduces the *shape* of paper Table I.

Iterates over (dataset, method, regime, seed) and emits a results parquet
ready for paired-bootstrap aggregation. Each cell is `methods[m](seed, data_cfg)`.

We don't have a text path, so the with-text/without-text distinction is a
no-op for the propagation-only methods (DocLessFake, GCNFN, UPFD-prop) and
the `text_only` regime is implemented by majority-class — i.e. it's the
floor that a hypothetical text-only model couldn't undershoot.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from ..baselines import train_gcnfn, train_majority, train_upfd_prop, train_bigcn, train_propmlp, train_rgcn
from ..devices import select_device
from ..train import DataConfig, ModelConfig, TrainConfig, train_one_seed
from .stats import paired_bootstrap


def _docless_runner(seed: int, data_cfg: DataConfig, **kw):
    m = train_one_seed(
        seed=seed,
        data_cfg=data_cfg,
        model_cfg=ModelConfig(**{k: v for k, v in kw.items()
                                  if k in {"hidden_dim", "num_layers", "num_heads",
                                            "dropout", "kp", "ke", "delta_minutes",
                                            "k_max"}}),
        train_cfg=TrainConfig(**{k: v for k, v in kw.items()
                                   if k in {"pretrain_epochs", "finetune_epochs",
                                             "batch_size", "learning_rate",
                                             "lambda_ssl", "lambda_kl_max",
                                             "kl_anneal_frac", "coverage_budget"}}),
        verbose=False,
        device=kw.get("device"),
    )
    m["method"] = "docless"
    return m


METHODS = {
    "majority": train_majority,
    "gcnfn": train_gcnfn,
    "bigcn": train_bigcn,
    "rgcn": train_rgcn,
    "propmlp": train_propmlp,
    "upfd_prop": train_upfd_prop,
    "docless": _docless_runner,
}


def run(
    datasets: dict[str, DataConfig],
    methods: list[str],
    seeds: list[int],
    out_dir: Path,
    epochs_short: int = 12,
    epochs_docless_pre: int = 4,
    epochs_docless_ft: int = 20,
    hidden_dim: int = 64,
) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    t_start = time.time()

    for ds_name, dcfg in datasets.items():
        for method in methods:
            for seed in seeds:
                t = time.time()
                if method == "docless":
                    kw = dict(
                        device=device,
                        hidden_dim=hidden_dim,
                        pretrain_epochs=epochs_docless_pre,
                        finetune_epochs=epochs_docless_ft,
                    )
                elif method == "majority":
                    kw = {}
                else:
                    kw = dict(device=device, hidden_dim=hidden_dim, epochs=epochs_short)

                m = METHODS[method](seed, dcfg, **kw)
                m["dataset"] = ds_name
                m["regime"] = "without_text"  # all our local methods are propagation-only
                m["wall_s"] = round(time.time() - t, 2)
                results.append(m)
                print(f"  [{ds_name:>10s} | {method:>10s} | seed {seed}] "
                      f"f1={m['f1_all']:.3f} cov={m['coverage']:.3f}  "
                      f"({m['wall_s']:.1f}s)")

    (out_dir / "main_table.json").write_text(
        json.dumps({"results": results, "wall_s_total": time.time() - t_start}, indent=2)
    )

    # Per-cell aggregation
    agg = {}
    for r in results:
        key = (r["dataset"], r["method"], r["regime"])
        agg.setdefault(key, []).append(r["f1_all"])
    cells = [
        {"dataset": d, "method": m, "regime": rg,
         "f1_mean": round(sum(v) / len(v), 3),
         "f1_std": round(_std(v), 3),
         "n_seeds": len(v)}
        for (d, m, rg), v in agg.items()
    ]
    (out_dir / "main_table_agg.json").write_text(json.dumps(cells, indent=2))
    return {"raw": results, "cells": cells}


def _std(xs):
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--methods", nargs="+", default=["majority", "gcnfn", "bigcn", "upfd_prop", "docless"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--balanced", action="store_true", default=True)
    ap.add_argument("--hidden-dim", type=int, default=32, help="local-CPU default")
    ap.add_argument("--epochs-short", type=int, default=12, help="baseline epochs")
    ap.add_argument("--epochs-docless-pre", type=int, default=2)
    ap.add_argument("--epochs-docless-ft", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("outputs/main_table"))
    args = ap.parse_args()

    ds_cfgs = {}
    if "coaid" in args.datasets:
        ds_cfgs["coaid"] = DataConfig(source="coaid", balanced=args.balanced)
    if "recovery" in args.datasets:
        ds_cfgs["recovery"] = DataConfig(
            source="recovery", balanced=args.balanced, data_root="data/raw/recovery",
        )
    if "synthetic" in args.datasets:
        ds_cfgs["synthetic"] = DataConfig(source="synthetic", n_stories=400)

    print(f"Running {len(args.datasets)} datasets × {len(args.methods)} methods × {len(args.seeds)} seeds")
    out = run(
        ds_cfgs, args.methods, args.seeds, args.out,
        epochs_short=args.epochs_short,
        epochs_docless_pre=args.epochs_docless_pre,
        epochs_docless_ft=args.epochs_docless_ft,
        hidden_dim=args.hidden_dim,
    )
    print("\nCells:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>10s} | {c['method']:>10s} | {c['regime']:>13s} "
              f"| F1 {c['f1_mean']:.3f} ± {c['f1_std']:.3f} ({c['n_seeds']} seeds)")


if __name__ == "__main__":
    main()
