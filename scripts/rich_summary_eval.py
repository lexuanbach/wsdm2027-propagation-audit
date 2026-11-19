"""Rich-feature audit (PropMLP+ and DocLessFake+).

Compares four variants on the two powered datasets (ReCOVery, Politifact):
  PropMLP        -- 7-dim summary baseline
  PropMLP+       -- 15-dim temporal-dynamics enriched baseline
  DocLessFake    -- hypergraph SSL + 7-dim summary fusion
  DocLessFake+   -- hypergraph SSL + 15-dim summary fusion
Paired bootstrap (per-story losses pooled across 5 seeds) for the two key
contrasts: PropMLP+ vs PropMLP (does richer baseline help?) and DocLessFake+
vs PropMLP+ (does structural fusion still help on top of the richer baseline?).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from docless.baselines import train_propmlp
from docless.eval.stats import paired_bootstrap
from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

OUT = "outputs/r9/rich_summary"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "recovery": dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
}
dev = torch.device("cpu")


def _loss(d):
    return (np.asarray(d["preds"]) != np.asarray(d["labels"])).astype(float)


def run_dataset(name, dcfg_kw):
    dcfg = DataConfig(**dcfg_kw)
    runs = {"propmlp": [], "propmlp_rich": [], "docless": [], "docless_rich": []}
    for s in SEEDS:
        runs["propmlp"].append(train_propmlp(s, dcfg, hidden_dim=64, epochs=25, device=dev, rich=False))
        runs["propmlp_rich"].append(train_propmlp(s, dcfg, hidden_dim=64, epochs=25, device=dev, rich=True))
        runs["docless"].append(train_one_seed(s, dcfg, ModelConfig(hidden_dim=64, rich_summary=False),
                                              TrainConfig(pretrain_epochs=4, finetune_epochs=25),
                                              verbose=False, device=dev))
        runs["docless_rich"].append(train_one_seed(s, dcfg, ModelConfig(hidden_dim=64, rich_summary=True),
                                                   TrainConfig(pretrain_epochs=4, finetune_epochs=25),
                                                   verbose=False, device=dev))
        f1s = {k: round(v[-1]["f1_all"], 3) for k, v in runs.items()}
        print(f"[{name}] seed {s}  {f1s}", flush=True)

    agg = {k: {"f1_mean": round(float(np.mean([r["f1_all"] for r in v])), 3),
               "f1_std": round(float(np.std([r["f1_all"] for r in v])), 3),
               "per_seed": [round(r["f1_all"], 4) for r in v]}
           for k, v in runs.items()}

    def paired(a_runs, b_runs):
        a = np.concatenate([_loss(r) for r in a_runs])
        b = np.concatenate([_loss(r) for r in b_runs])
        r = paired_bootstrap(a, b)
        return dict(mean_diff_b_minus_a=round(r.mean_diff, 4), p_value=round(r.p_value, 4),
                    ci_lo=round(r.ci_lo, 4), ci_hi=round(r.ci_hi, 4), n=int(a.shape[0]))

    out = {
        "main": agg,
        "paired_propmlp_rich_vs_propmlp": paired(runs["propmlp"], runs["propmlp_rich"]),
        "paired_docless_rich_vs_propmlp_rich": paired(runs["propmlp_rich"], runs["docless_rich"]),
        "paired_docless_rich_vs_propmlp": paired(runs["propmlp"], runs["docless_rich"]),
    }
    json.dump(out, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"== {name} done", flush=True)
    return out


if __name__ == "__main__":
    for name, dcfg in DATASETS.items():
        run_dataset(name, dcfg)
    print("ALL DONE", flush=True)
