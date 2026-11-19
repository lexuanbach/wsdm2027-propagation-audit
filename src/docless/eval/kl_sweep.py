"""KL-agreement sensitivity sweep for lambda_2.

Runs DocLessFake while varying the maximum KL coefficient used to anneal
agreement between the observational and interventional heads. This directly
addresses the concern that too much KL agreement can collapse the
TV-disagreement signal used by abstention.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..devices import select_device
from ..train import DataConfig, ModelConfig, TrainConfig, train_one_seed


def run(
    datasets: dict[str, DataConfig],
    lambdas: list[float],
    seeds: list[int],
    out_dir: Path,
    hidden_dim: int = 64,
    pretrain_epochs: int = 4,
    finetune_epochs: int = 25,
) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()

    for ds_name, dcfg in datasets.items():
        for lam in lambdas:
            for seed in seeds:
                t = time.time()
                m = train_one_seed(
                    seed=seed,
                    data_cfg=dcfg,
                    model_cfg=ModelConfig(hidden_dim=hidden_dim, num_layers=2),
                    train_cfg=TrainConfig(
                        pretrain_epochs=pretrain_epochs,
                        finetune_epochs=finetune_epochs,
                        lambda_kl_max=lam,
                        coverage_budget=0.9,
                    ),
                    verbose=False,
                    device=device,
                )
                m.update(
                    dataset=ds_name,
                    lambda_kl_max=lam,
                    selective_gain=m["f1_kept"] - m["f1_all"],
                    wall_s=round(time.time() - t, 2),
                )
                rows.append(m)
                print(
                    f"  [{ds_name:>8s} lambda={lam:.2f} seed={seed}] "
                    f"f1={m['f1_all']:.3f} kept={m['f1_kept']:.3f} "
                    f"cov={m['coverage']:.3f} ({m['wall_s']:.1f}s)"
                )

    (out_dir / "kl_sweep.json").write_text(
        json.dumps({"results": rows, "wall_s_total": time.time() - t0}, indent=2)
    )

    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["lambda_kl_max"]), []).append(r)
    cells = []
    for (d, lam), vals in agg.items():
        cells.append(
            {
                "dataset": d,
                "lambda_kl_max": lam,
                "f1_mean": float(np.mean([v["f1_all"] for v in vals])),
                "f1_std": float(np.std([v["f1_all"] for v in vals], ddof=1)) if len(vals) > 1 else 0.0,
                "coverage_mean": float(np.mean([v["coverage"] for v in vals])),
                "f1_kept_mean": float(np.mean([v["f1_kept"] for v in vals])),
                "selective_gain_mean": float(np.mean([v["selective_gain"] for v in vals])),
                "n_seeds": len(vals),
            }
        )
    cells.sort(key=lambda c: (c["dataset"], c["lambda_kl_max"]))
    (out_dir / "kl_sweep_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid"])
    ap.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3, 0.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--pretrain-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("outputs/r6/kl_sweep"))
    args = ap.parse_args()

    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(
            source="recovery", balanced=True, data_root="data/raw/recovery"
        )

    out = run(
        ds,
        args.lambdas,
        args.seeds,
        args.out,
        hidden_dim=args.hidden_dim,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
    )
    print("\nCells:")
    for c in out["cells"]:
        print(
            f"  {c['dataset']:>8s} lambda={c['lambda_kl_max']:.2f} "
            f"F1 {c['f1_mean']:.3f} ± {c['f1_std']:.3f}; "
            f"kept {c['f1_kept_mean']:.3f}; cov {c['coverage_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
