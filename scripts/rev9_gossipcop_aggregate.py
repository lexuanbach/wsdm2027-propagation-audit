"""Aggregate the five parallel DLF-Full+ GossipCop seeds into the table cell.

Reports mean and population std (ddof=0) at three decimals, which is the
convention the DocLessFake/DLF-Full rows of the main table already use --
verified against the frozen ReCOVery and PolitiFact cells, which this pipeline
reproduces exactly.
"""
from __future__ import annotations

import json
import os

import numpy as np

SEED_DIR = "outputs/r13/gossipcop_seeds"
SEEDS = [0, 1, 2, 3, 4]

vals, missing = [], []
for s in SEEDS:
    p = f"{SEED_DIR}/seed{s}.json"
    if os.path.exists(p):
        vals.append((s, json.load(open(p))["f1_all"]))
    else:
        missing.append(s)

if missing:
    print(f"WAITING: seeds {missing} not finished ({len(vals)}/5 done)")
    for s, v in vals:
        print(f"  seed {s}: {v:.4f}")
    raise SystemExit(1)

f1 = np.array([v for _, v in vals], dtype=float)
out = {
    "arm": "docless_rich", "label": "DLF-Full+ (15d)", "dataset": "gossipcop",
    "seeds": SEEDS,
    "f1_mean": round(float(f1.mean()), 3),
    "f1_std": round(float(f1.std()), 3),
    "per_seed": [round(float(v), 4) for _, v in vals],
}
json.dump(out, open("outputs/r13/gossipcop_dlf_plus.json", "w"), indent=2)
print(json.dumps(out, indent=2))
print(f"\nTABLE CELL:  ${out['f1_mean']:.3f}_{{{out['f1_std']:.3f}}}$")
