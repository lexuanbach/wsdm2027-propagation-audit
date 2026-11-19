"""EBGCN-style comparator on every public-release dataset:
same per-split hypergraph, node features, readout, 5 seeds and the same
validation-selected epochs x lr grid as the other GNNs (rev1_gnn.py), so the
row can enter the main table. Writes outputs/r11/ebgcn/ebgcn.json.
Usage: PYTHONPATH=src python3 scripts/rev6_ebgcn_all.py [datasets...]
"""
import json, os, sys, time
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import rev1_gnn as R  # reuses _fit_eval, CONFIGS, DATASETS, SEEDS
from docless.baselines.ebgcn import _EBGCN
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split
from docless.hypergraph import build_hypergraph
import numpy as np

OUT = "outputs/r11/ebgcn"; os.makedirs(OUT, exist_ok=True)
names = sys.argv[1:] or ["recovery", "politifact", "gossipcop"]
path = f"{OUT}/ebgcn.json"
out = json.load(open(path)) if os.path.exists(path) else {}
for name in names:
    kw = R.DATASETS[name]; dcfg = DataConfig(**kw); res = []; t0 = time.time()
    for s in R.SEEDS:
        _set_seed(s); casc = _load_cascades(dcfg, s); tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
        h_tr = build_hypergraph(tr, feature_dim=16, seed=s); h_va = build_hypergraph(va, feature_dim=16, seed=s + 1); h_te = build_hypergraph(te, feature_dim=16, seed=s + 2)
        y_tr, y_va, y_te = h_tr.story_labels, h_va.story_labels, h_te.story_labels
        best, bestv = None, -1
        for cfg in R.CONFIGS:
            vf1 = R._fit_eval(_EBGCN, h_tr, y_tr, h_va, y_va, cfg["epochs"], cfg["lr"], s)
            if vf1 > bestv: bestv, best = vf1, cfg
        tf1 = R._fit_eval(_EBGCN, h_tr, y_tr, h_te, y_te, best["epochs"], best["lr"], s)
        res.append(round(float(tf1), 4)); print(f"[{name}] seed {s} ebgcn={tf1:.4f} cfg={best}", flush=True)
    out[name] = dict(f1_mean=round(float(np.mean(res)), 3), f1_std=round(float(np.std(res)), 3), per_seed=res, wall_s=round(time.time() - t0))
    json.dump(out, open(path, "w"), indent=2); print(f"[{name}] done {out[name]}", flush=True)
