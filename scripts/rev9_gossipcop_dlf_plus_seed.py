"""One seed of DLF-Full+ (15d) on GossipCop, so the five seeds can run in parallel.

The serial runner used one core of eighteen: the cost is dominated by
single-threaded Python string work in the per-split hypergraph build, which
scales with event-proxy count (GossipCop at 1,500/class has far more event
proxies than CoAID's 214 stories). Seeds are independent -- train_one_seed takes
the seed as its only varying input -- so fanning them out changes wall-clock,
not numbers: each process reproduces exactly what the serial loop would have
written for that seed.

Config is byte-identical to rich_summary_eval.run_dataset's docless_rich arm.
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

OUT = "outputs/r13/gossipcop_seeds"
DCFG = dict(source="gossipcop", balanced=True,
            data_root="data/raw/fakenewsnet", max_per_class=1500)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    r = train_one_seed(
        a.seed, DataConfig(**DCFG),
        ModelConfig(hidden_dim=64, rich_summary=True),
        TrainConfig(pretrain_epochs=4, finetune_epochs=25),
        verbose=False, device=torch.device("cpu"),
    )
    json.dump({"seed": a.seed, "f1_all": r["f1_all"]},
              open(f"{OUT}/seed{a.seed}.json", "w"), indent=2)
    print(f"[gossipcop docless_rich] seed {a.seed} f1={r['f1_all']:.4f}", flush=True)
