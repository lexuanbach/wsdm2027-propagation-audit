"""Ablation runner — paper Table III shape.

Toggles the SSL, causal, and abstention components of DocLessFake one at a
time. Implemented by training with modified hyper-parameters:

    no_ssl        lambda_ssl = 0 and pretrain_epochs = 0
    no_causal     KL coefficient = 0, f_do collapses to f_obs at evaluation
    no_abstain    coverage_budget = 1.0 → τ = inf so nothing is abstained
    full          baseline (paper §V-A defaults)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..devices import select_device
from ..train import DataConfig, ModelConfig, TrainConfig, train_one_seed


VARIANTS = {
    "full":       dict(),
    "no_ssl":     dict(lambda_ssl=0.0, pretrain_epochs=0),
    "no_causal":  dict(lambda_kl_max=0.0),
    "no_abstain": dict(coverage_budget=1.0),
}


def run(datasets: dict[str, DataConfig], seeds: list[int], out_dir: Path,
        pretrain_epochs: int = 4, finetune_epochs: int = 20, hidden_dim: int = 64) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()

    for ds_name, dcfg in datasets.items():
        for var, override in VARIANTS.items():
            for seed in seeds:
                tcfg = TrainConfig(pretrain_epochs=pretrain_epochs,
                                    finetune_epochs=finetune_epochs,
                                    **{k: v for k, v in override.items()
                                        if k in {"lambda_ssl", "lambda_kl_max",
                                                  "coverage_budget"}})
                if "pretrain_epochs" in override:
                    tcfg.pretrain_epochs = override["pretrain_epochs"]
                mcfg = ModelConfig(hidden_dim=hidden_dim, num_layers=2)
                t = time.time()
                m = train_one_seed(seed, dcfg, mcfg, tcfg, verbose=False, device=device)
                m.update(dict(dataset=ds_name, variant=var, wall_s=round(time.time() - t, 2)))
                rows.append(m)
                print(f"  [{ds_name:>10s} | {var:>10s} | seed {seed}] "
                      f"f1={m['f1_all']:.3f} cov={m['coverage']:.3f} ({m['wall_s']:.1f}s)")

    (out_dir / "ablation.json").write_text(json.dumps(
        {"results": rows, "wall_s_total": time.time() - t0}, indent=2))

    agg = {}
    for r in rows:
        key = (r["dataset"], r["variant"])
        agg.setdefault(key, []).append(r["f1_all"])
    cells = [
        {"dataset": d, "variant": v,
         "f1_mean": round(sum(f) / len(f), 3),
         "f1_std": round(_std(f), 3),
         "n_seeds": len(f)}
        for (d, v), f in agg.items()
    ]
    (out_dir / "ablation_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def _std(xs):
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--pretrain-epochs", type=int, default=2)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("outputs/ablation"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.seeds, args.out,
              pretrain_epochs=args.pretrain_epochs,
              finetune_epochs=args.finetune_epochs,
              hidden_dim=args.hidden_dim)
    print("\nCells:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>10s} | {c['variant']:>10s} | "
              f"F1 {c['f1_mean']:.3f} ± {c['f1_std']:.3f}")


if __name__ == "__main__":
    main()
