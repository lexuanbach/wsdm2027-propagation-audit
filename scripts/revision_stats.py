"""Paired DLF-Full statistics:

(1) Paired bootstrap: fused DocLessFake vs PropMLP per dataset.
    Per-story 0/1 losses pooled over 5 seeds on identical splits.
(2) Permuted-stratum control: DocLessFake with (P,E_hash) randomly permuted
    vs normal on CoAID, testing whether the adjustment head uses those strata.

Base config (hidden-64, 4+25) matching the main table. Writes JSON.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from docless.baselines import train_propmlp
from docless.eval.stats import paired_bootstrap
from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

OUT = "outputs/r9/revision_stats"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DSETS = {"coaid": "data/raw/coaid", "recovery": "data/raw/recovery"}


def _loss(d):
    return (np.asarray(d["preds"]) != np.asarray(d["labels"])).astype(float)


def paired():
    out = {}
    for ds, root in DSETS.items():
        dcfg = DataConfig(source=ds, balanced=True, data_root=root)
        dl, pm = [], []
        df1, pf1 = [], []
        for s in SEEDS:
            md = train_one_seed(s, dcfg, ModelConfig(hidden_dim=64),
                                TrainConfig(pretrain_epochs=4, finetune_epochs=25),
                                verbose=False, device=torch.device("cpu"))
            mp = train_propmlp(s, dcfg, hidden_dim=64, epochs=25, device=torch.device("cpu"))
            # identical split per seed -> aligned by index
            dl.append(_loss(md)); pm.append(_loss(mp))
            df1.append(md["f1_all"]); pf1.append(mp["f1_all"])
            print(f"[paired {ds}] seed {s} docless {md['f1_all']:.3f} propmlp {mp['f1_all']:.3f}", flush=True)
        dl = np.concatenate(dl); pm = np.concatenate(pm)
        r = paired_bootstrap(dl, pm)  # mean(propmlp_loss - docless_loss); positive => docless better
        out[ds] = {
            "docless_f1_mean": round(float(np.mean(df1)), 3),
            "propmlp_f1_mean": round(float(np.mean(pf1)), 3),
            "mean_loss_diff_propmlp_minus_docless": round(r.mean_diff, 4),
            "p_value": round(r.p_value, 4),
            "ci_lo": round(r.ci_lo, 4), "ci_hi": round(r.ci_hi, 4),
            "n_stories": int(dl.shape[0]),
        }
        print(f"== paired {ds}: {out[ds]}", flush=True)
        json.dump(out, open(f"{OUT}/paired_bootstrap.json", "w"), indent=2)
    return out


def permuted():
    dcfg = DataConfig(source="coaid", balanced=True, data_root="data/raw/coaid")
    res = {}
    for name, perm in (("normal", False), ("permuted_PE", True)):
        f1s = []
        for s in [0, 1, 2]:
            m = train_one_seed(s, dcfg, ModelConfig(hidden_dim=64),
                               TrainConfig(pretrain_epochs=4, finetune_epochs=25,
                                           permute_confounders=perm),
                               verbose=False, device=torch.device("cpu"))
            f1s.append(round(m["f1_all"], 4))
            print(f"[permute {name}] seed {s} f1 {m['f1_all']:.3f}", flush=True)
        res[name] = {"f1_mean": round(float(np.mean(f1s)), 3),
                     "f1_std": round(float(np.std(f1s)), 3), "per_seed": f1s}
        json.dump(res, open(f"{OUT}/permuted_confounder.json", "w"), indent=2)
    print(f"== permuted: {res}", flush=True)
    return res


if __name__ == "__main__":
    print("PAIRED BOOTSTRAP", flush=True)
    paired()
    print("PERMUTED CONFOUNDER CONTROL", flush=True)
    permuted()
    print("ALL DONE", flush=True)
