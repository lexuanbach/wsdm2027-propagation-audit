"""SSL negative-source comparison.

Trains DocLessFake twice -- once with genuinely label-free hard negatives
(the default) and once with the old label-aware (opposite-label) negatives --
to quantify whether label information in negative selection inflates the SSL
contribution. If the two are comparable, the SSL gain is structural, not a
label leak.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..devices import select_device
from ..train import DataConfig, ModelConfig, TrainConfig, train_one_seed


def run(datasets, seeds, out_dir: Path, hidden_dim=64, pretrain_epochs=4, finetune_epochs=25) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for ds_name, dcfg in datasets.items():
        for label_free in (True, False):
            for seed in seeds:
                tcfg = TrainConfig(pretrain_epochs=pretrain_epochs,
                                    finetune_epochs=finetune_epochs,
                                    ssl_label_free=label_free)
                mcfg = ModelConfig(hidden_dim=hidden_dim, num_layers=2)
                m = train_one_seed(seed, dcfg, mcfg, tcfg, verbose=False, device=device)
                mode = "label_free" if label_free else "label_aware"
                rows.append(dict(dataset=ds_name, mode=mode, seed=seed, f1=m["f1_all"]))
                print(f"  [{ds_name:>8s} | {mode:>11s} | seed {seed}] f1={m['f1_all']:.3f}")
    (out_dir / "ssl_negsource.json").write_text(json.dumps({"rows": rows, "wall_s": time.time() - t0}, indent=2))
    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["mode"]), []).append(r["f1"])
    cells = [dict(dataset=d, mode=mo,
                  f1_mean=round(float(np.mean(v)), 3),
                  f1_std=round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0,
                  n_seeds=len(v))
             for (d, mo), v in agg.items()]
    (out_dir / "ssl_negsource_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--pretrain-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("outputs/r8/ssl_negsource"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.seeds, args.out, hidden_dim=args.hidden_dim,
              pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs)
    print("\nCells:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>8s} | {c['mode']:>11s} | F1 {c['f1_mean']:.3f} +/- {c['f1_std']:.3f}")


if __name__ == "__main__":
    main()
