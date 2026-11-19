"""Fourth-dataset eval on FakeNewsNet/GossipCop (out-of-domain, entertainment).

To keep CPU runtime tractable we subsample to 500 balanced stories per class
(1000 total, ReCOVery-comparable). Runs the full 6-method main comparison and
paired bootstrap DocLessFake(full) vs PropMLP, 5 seeds. Mirrors the
Politifact protocol; we skip the structure-vs-summary ablation on GossipCop
because the redundancy pattern is already established on the two powered
datasets (ReCOVery, Politifact).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from docless.baselines import (
    train_bigcn, train_gcnfn, train_majority, train_propmlp, train_upfd_prop,
)
from docless.eval.stats import paired_bootstrap
from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

OUT = "outputs/r9/gossipcop_3k"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DCFG = dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet",
            max_per_class=1500)
dev = torch.device("cpu")


def main():
    dcfg = DataConfig(**DCFG)
    rows = {}
    base = {
        "majority": lambda s: train_majority(s, dcfg),
        "gcnfn": lambda s: train_gcnfn(s, dcfg, hidden_dim=64, epochs=25, device=dev),
        "bigcn": lambda s: train_bigcn(s, dcfg, hidden_dim=64, epochs=25, device=dev),
        "upfd_prop": lambda s: train_upfd_prop(s, dcfg, hidden_dim=64, epochs=25, device=dev),
        "propmlp": lambda s: train_propmlp(s, dcfg, hidden_dim=64, epochs=25, device=dev),
    }
    for name, fn in base.items():
        runs = [fn(s) for s in SEEDS]
        rows[name] = runs
        print(f"[main] {name} {np.mean([r['f1_all'] for r in runs]):.3f}", flush=True)

    docless_runs = [train_one_seed(s, dcfg, ModelConfig(hidden_dim=64),
                                    TrainConfig(pretrain_epochs=4, finetune_epochs=25),
                                    verbose=False, device=dev) for s in SEEDS]
    rows["docless"] = docless_runs
    print(f"[main] docless {np.mean([r['f1_all'] for r in docless_runs]):.3f}", flush=True)

    main_agg = {m: {"f1_mean": round(float(np.mean([r["f1_all"] for r in v])), 3),
                    "f1_std": round(float(np.std([r["f1_all"] for r in v])), 3),
                    "per_seed": [round(r["f1_all"], 4) for r in v]}
                for m, v in rows.items()}

    dl, pm = [], []
    for i in range(len(SEEDS)):
        md = rows["docless"][i]; mp = rows["propmlp"][i]
        dl.append((np.asarray(md["preds"]) != np.asarray(md["labels"])).astype(float))
        pm.append((np.asarray(mp["preds"]) != np.asarray(mp["labels"])).astype(float))
    dl = np.concatenate(dl); pm = np.concatenate(pm)
    r = paired_bootstrap(dl, pm)
    paired = {"docless_f1": main_agg["docless"]["f1_mean"], "propmlp_f1": main_agg["propmlp"]["f1_mean"],
              "mean_loss_diff_propmlp_minus_docless": round(r.mean_diff, 4),
              "p_value": round(r.p_value, 4), "ci_lo": round(r.ci_lo, 4), "ci_hi": round(r.ci_hi, 4),
              "n_stories": int(dl.shape[0])}

    json.dump({"main": main_agg, "paired_docless_vs_propmlp": paired, "max_per_class": 1500},
              open(f"{OUT}/gossipcop_eval.json", "w"), indent=2)
    print("== paired:", paired, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
