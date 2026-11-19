"""Fill the one remaining main-table cell: DLF-Full+ (15d) on GossipCop.

`rev9_rich_summary_rest.py` ran the full four-arm rich-feature audit, but on
GossipCop at 1,500/class the per-split hypergraph rebuild dominates: a single
seed of all four arms takes ~1.5 h, i.e. ~8 h for five seeds.  Only the
`docless_rich` arm feeds the missing table cell.  Protocol parity with the
published DocLessFake row is already established -- the CoAID re-run reproduced
its cell exactly (0.654 +/- 0.106), and the frozen ReCOVery/PolitiFact
rich-summary outputs match their published cells -- so the other three arms are
re-derivation rather than evidence, and are dropped here.

Config is byte-identical to `rich_summary_eval.run_dataset`'s docless_rich arm.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

OUT = "outputs/r13"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DCFG = dict(source="gossipcop", balanced=True,
            data_root="data/raw/fakenewsnet", max_per_class=1500)

if __name__ == "__main__":
    dcfg = DataConfig(**DCFG)
    dev = torch.device("cpu")
    runs = []
    for s in SEEDS:
        r = train_one_seed(
            s, dcfg,
            ModelConfig(hidden_dim=64, rich_summary=True),
            TrainConfig(pretrain_epochs=4, finetune_epochs=25),
            verbose=False, device=dev,
        )
        runs.append(r)
        print(f"[gossipcop docless_rich] seed {s}  f1={r['f1_all']:.4f}", flush=True)

    f1 = [r["f1_all"] for r in runs]
    out = {
        "arm": "docless_rich",
        "label": "DLF-Full+ (15d)",
        "dataset": "gossipcop",
        "data_config": DCFG,
        "model_config": {"hidden_dim": 64, "rich_summary": True},
        "train_config": {"pretrain_epochs": 4, "finetune_epochs": 25},
        "seeds": SEEDS,
        "f1_mean": round(float(np.mean(f1)), 3),
        "f1_std": round(float(np.std(f1)), 3),
        "per_seed": [round(v, 4) for v in f1],
    }
    json.dump(out, open(f"{OUT}/gossipcop_dlf_plus.json", "w"), indent=2)
    print("DONE", json.dumps({k: out[k] for k in ("f1_mean", "f1_std", "per_seed")}), flush=True)
