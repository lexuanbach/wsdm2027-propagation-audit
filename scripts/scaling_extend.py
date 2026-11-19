"""Extend the hidden-128 ReCOVery structure-vs-summary probe to 5 seeds
(seeds 0-2 already in outputs/r9/strong_config/recovery.json) and run a paired
test on the per-seed full-vs-summary differences."""
from __future__ import annotations
import json, os, numpy as np, torch
from docless.train import DataConfig, ModelConfig, TrainConfig, train_one_seed

OUT = "outputs/r9/strong_config"
prev = json.load(open(f"{OUT}/recovery.json"))
full = list(prev["full"]["per_seed"]); summ = list(prev["summary_only"]["per_seed"])
dcfg = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
tcfg = TrainConfig(pretrain_epochs=8, finetune_epochs=35)
for s in [3, 4]:
    f = train_one_seed(s, dcfg, ModelConfig(hidden_dim=128, use_summary=True, use_structure=True),
                       tcfg, verbose=False, device=torch.device("cpu"))["f1_all"]
    g = train_one_seed(s, dcfg, ModelConfig(hidden_dim=128, use_summary=True, use_structure=False),
                       tcfg, verbose=False, device=torch.device("cpu"))["f1_all"]
    full.append(round(f, 4)); summ.append(round(g, 4))
    print(f"[recovery@128] seed {s} full {f:.3f} summary {g:.3f}", flush=True)
full = np.array(full); summ = np.array(summ)
diffs = full - summ
rng = np.random.default_rng(0)
bs = rng.choice(diffs, size=(10000, len(diffs)), replace=True).mean(1)
p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
res = {"full": {"f1_mean": round(float(full.mean()), 3), "f1_std": round(float(full.std()), 3), "per_seed": full.round(4).tolist()},
       "summary_only": {"f1_mean": round(float(summ.mean()), 3), "f1_std": round(float(summ.std()), 3), "per_seed": summ.round(4).tolist()},
       "delta_struct_mean": round(float(diffs.mean()), 4),
       "delta_struct_per_seed": diffs.round(4).tolist(),
       "paired_p_value": round(float(p), 4), "n_seeds": len(full)}
json.dump(res, open(f"{OUT}/recovery_5seed.json", "w"), indent=2)
print("== 5-seed:", res, flush=True)
print("DONE", flush=True)
