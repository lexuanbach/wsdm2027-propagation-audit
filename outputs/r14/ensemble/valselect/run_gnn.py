"""Regenerate EBGCN-style / GATv2 with TRAIN-ONLY config selection, and dump
val + test per-story predictions and scores.

MODIFIED vs the published rows: the published protocol picks (epochs, lr) by
maximising macro-F1 ON THE VALIDATION FOLD.  That makes the published val score a
selected maximum and therefore useless as an unbiased input to an outer
val-fold model-selection rule.  Here the config is chosen on an inner
85/15 stratified holdout carved out of the TRAIN fold; the winner is then refit
on the full train fold and scored once on val and once on test.  Same number of
fits as the published protocol, so the same cost.

Usage: run_gnn.py <corpus> <ebgcn|gatv2> <seed>
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import folds, check_alignment, dump
from docless.baselines.ebgcn import _EBGCN
from docless.baselines.gatv2 import _GATv2
from docless.eval.stats import macro_f1
from docless.hypergraph import build_hypergraph
from docless.train import _set_seed, _stratified_split

dev = torch.device("cpu")
SPEC = {
    "ebgcn": (_EBGCN, [dict(epochs=25, lr=5e-4), dict(epochs=40, lr=1e-3),
                       dict(epochs=40, lr=5e-3), dict(epochs=80, lr=5e-3)]),
    "gatv2": (_GATv2, [dict(epochs=40, lr=1e-3), dict(epochs=40, lr=5e-3),
                       dict(epochs=80, lr=5e-3)]),
}


def fit(Cls, h_tr, y_tr, epochs, lr, seed):
    _set_seed(seed)
    m = Cls(in_dim=16, hidden_dim=64).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids), y_tr).backward()
        opt.step()
    m.eval()
    return m


def score(m, h):
    with torch.no_grad():
        lg = m(h, h.story_to_burst_edges, h.story_ids)
    return lg.argmax(-1).cpu().numpy(), F.softmax(lg, -1)[:, 1].cpu().numpy()


corpus, mname, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
Cls, CONFIGS = SPEC[mname]
t0 = time.time()
_set_seed(seed)
dcfg, tr, va, te = folds(corpus, seed)
check_alignment(corpus, seed, va, te)

# ---- inner train-only holdout for config selection ----
i_tr, i_va, i_rest = _stratified_split(tr, (0.85, 0.15, 0.0), seed + 7777)
i_va = i_va + i_rest
h_itr = build_hypergraph(i_tr, feature_dim=16, seed=seed)
h_iva = build_hypergraph(i_va, feature_dim=16, seed=seed + 1)
best, bestv, grid = None, -1.0, []
for cfg in CONFIGS:
    m = fit(Cls, h_itr, h_itr.story_labels, cfg["epochs"], cfg["lr"], seed)
    p, _ = score(m, h_iva)
    f1 = float(macro_f1(p, h_iva.story_labels.cpu().numpy()))
    grid.append(dict(cfg=cfg, inner_val_f1=round(f1, 4)))
    if f1 > bestv:
        bestv, best = f1, cfg

# ---- refit on the full train fold, score val and test once ----
h_tr = build_hypergraph(tr, feature_dim=16, seed=seed)
h_va = build_hypergraph(va, feature_dim=16, seed=seed + 1)
h_te = build_hypergraph(te, feature_dim=16, seed=seed + 2)
m = fit(Cls, h_tr, h_tr.story_labels, best["epochs"], best["lr"], seed)
vp, vs = score(m, h_va)
tp, ts = score(m, h_te)
y_va = h_va.story_labels.cpu().numpy(); y_te = h_te.story_labels.cpu().numpy()
assert list(y_te) == [c.label for c in te]
assert list(y_va) == [c.label for c in va]
dump(mname, corpus, seed, va, te, vp, vs, tp, ts,
     extra=dict(selected_cfg=best, inner_grid=grid, wall_s=round(time.time() - t0, 1)))
print(json.dumps(dict(corpus=corpus, model=mname, seed=seed, cfg=best,
                      val_f1=round(float(macro_f1(vp, y_va)), 4),
                      test_f1=round(float(macro_f1(tp, y_te)), 4),
                      wall_s=round(time.time() - t0, 1))), flush=True)
