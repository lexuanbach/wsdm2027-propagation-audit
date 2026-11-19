"""Stronger-config probe: does the hypergraph add value on top of summary
features at higher capacity (hidden-128, more SSL + finetune epochs)?

Interleaves full vs summary_only PER SEED on ReCOVery (decisive, real signal),
so the first paired comparison lands after one seed instead of after a whole
variant. Writes incremental JSON. Then CoAID (all 3 variants). Per-seed lines
flushed live.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

HID = 128
PRE = 8
FT = 35
OUT = "outputs/r9/strong_config"
os.makedirs(OUT, exist_ok=True)


def _one(dataset, data_root, flags, seed):
    dcfg = DataConfig(source=dataset, balanced=True, data_root=data_root)
    tcfg = TrainConfig(pretrain_epochs=PRE, finetune_epochs=FT)
    mcfg = ModelConfig(hidden_dim=HID, **flags)
    m = train_one_seed(seed, dcfg, mcfg, tcfg, verbose=False, device=torch.device("cpu"))
    return round(m["f1_all"], 4)


def _agg(d):
    return {k: {"f1_mean": round(float(np.mean(v)), 3),
                "f1_std": round(float(np.std(v)), 3),
                "per_seed": v} for k, v in d.items()}


if __name__ == "__main__":
    print(f"config: hidden={HID} pretrain={PRE} finetune={FT}", flush=True)

    # ---- ReCOVery: interleave full vs summary_only per seed (decisive) ----
    rec = {"full": [], "summary_only": []}
    for s in [0, 1, 2]:
        for name, flags in (("full", dict(use_summary=True, use_structure=True)),
                            ("summary_only", dict(use_summary=True, use_structure=False))):
            f = _one("recovery", "data/raw/recovery", flags, s)
            rec[name].append(f)
            print(f"[recovery] {name} seed {s} f1 {f:.3f}", flush=True)
        json.dump(_agg(rec), open(f"{OUT}/recovery.json", "w"), indent=2)
        print(f"  -> recovery so far: full {np.mean(rec['full']):.3f} | "
              f"summary_only {np.mean(rec['summary_only']):.3f}", flush=True)
    print("RECOVERY DONE", flush=True)

    # ---- CoAID: all three variants, 5 seeds (cheap) ----
    co = {"full": [], "summary_only": [], "structure_only": []}
    for name, flags in (("full", dict(use_summary=True, use_structure=True)),
                        ("summary_only", dict(use_summary=True, use_structure=False)),
                        ("structure_only", dict(use_summary=False, use_structure=True))):
        for s in [0, 1, 2, 3, 4]:
            f = _one("coaid", "data/raw/coaid", flags, s)
            co[name].append(f)
            print(f"[coaid] {name} seed {s} f1 {f:.3f}", flush=True)
        json.dump(_agg(co), open(f"{OUT}/coaid.json", "w"), indent=2)
    print("COAID DONE", flush=True)
    print("ALL DONE", flush=True)
