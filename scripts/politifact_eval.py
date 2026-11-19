"""Third-dataset evaluation on FakeNewsNet/Politifact (out-of-domain, political).

(1) Main comparison: all six methods, 5 seeds, base config (hidden-64, 4+25).
(2) Structure-vs-summary ablation (full / summary_only / structure_only), 5 seeds.
(3) Paired bootstrap DocLessFake(full) vs PropMLP, per-story, pooled.
Mirrors the CoAID/ReCOVery protocol exactly.
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

OUT = "outputs/r9/politifact"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DCFG = dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet")
dev = torch.device("cpu")


def _docless(seed, flags):
    return train_one_seed(seed, DataConfig(**DCFG), ModelConfig(hidden_dim=64, **flags),
                          TrainConfig(pretrain_epochs=4, finetune_epochs=25), verbose=False, device=dev)


def main():
    dcfg = DataConfig(**DCFG)
    # ---- (1) main comparison ----
    main_f1 = {}
    base_baselines = {
        "majority": lambda s: train_majority(s, dcfg),
        "gcnfn": lambda s: train_gcnfn(s, dcfg, hidden_dim=64, epochs=25, device=dev),
        "bigcn": lambda s: train_bigcn(s, dcfg, hidden_dim=64, epochs=25, device=dev),
        "upfd_prop": lambda s: train_upfd_prop(s, dcfg, hidden_dim=64, epochs=25, device=dev),
        "propmlp": lambda s: train_propmlp(s, dcfg, hidden_dim=64, epochs=25, device=dev),
    }
    for name, fn in base_baselines.items():
        fs = [fn(s)["f1_all"] for s in SEEDS]
        main_f1[name] = [round(x, 4) for x in fs]
        print(f"[main] {name} {np.mean(fs):.3f}", flush=True)

    # ---- (2) structure-vs-summary (docless variants) ----
    variants = {
        "full": dict(use_summary=True, use_structure=True),
        "summary_only": dict(use_summary=True, use_structure=False),
        "structure_only": dict(use_summary=False, use_structure=True),
    }
    var_runs = {}  # name -> list of metric dicts (per seed)
    for name, flags in variants.items():
        runs = [_docless(s, flags) for s in SEEDS]
        var_runs[name] = runs
        f1 = [r["f1_all"] for r in runs]
        print(f"[struct] {name} {np.mean(f1):.3f} +/- {np.std(f1):.3f}", flush=True)

    main_f1["docless"] = [round(r["f1_all"], 4) for r in var_runs["full"]]

    agg = {m: {"f1_mean": round(float(np.mean(v)), 3), "f1_std": round(float(np.std(v)), 3), "per_seed": v}
           for m, v in main_f1.items()}
    struct = {n: {"f1_mean": round(float(np.mean([r["f1_all"] for r in runs])), 3),
                  "f1_std": round(float(np.std([r["f1_all"] for r in runs])), 3),
                  "per_seed": [round(r["f1_all"], 4) for r in runs]}
              for n, runs in var_runs.items()}
    struct["delta_struct_full_minus_summary"] = round(
        float(np.mean([r["f1_all"] for r in var_runs["full"]]) -
              np.mean([r["f1_all"] for r in var_runs["summary_only"]])), 4)

    # ---- (3) paired bootstrap docless(full) vs propmlp, per-story ----
    dl, pm = [], []
    for i, s in enumerate(SEEDS):
        md = var_runs["full"][i]
        mp = train_propmlp(s, dcfg, hidden_dim=64, epochs=25, device=dev)
        dl.append((np.asarray(md["preds"]) != np.asarray(md["labels"])).astype(float))
        pm.append((np.asarray(mp["preds"]) != np.asarray(mp["labels"])).astype(float))
    dl = np.concatenate(dl); pm = np.concatenate(pm)
    r = paired_bootstrap(dl, pm)
    paired = {"docless_f1": agg["docless"]["f1_mean"], "propmlp_f1": agg["propmlp"]["f1_mean"],
              "mean_loss_diff_propmlp_minus_docless": round(r.mean_diff, 4),
              "p_value": round(r.p_value, 4), "ci_lo": round(r.ci_lo, 4), "ci_hi": round(r.ci_hi, 4),
              "n_stories": int(dl.shape[0])}

    json.dump({"main": agg, "structure_vs_summary": struct, "paired_docless_vs_propmlp": paired},
              open(f"{OUT}/politifact_eval.json", "w"), indent=2)
    print("== struct delta:", struct["delta_struct_full_minus_summary"], flush=True)
    print("== paired:", paired, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
