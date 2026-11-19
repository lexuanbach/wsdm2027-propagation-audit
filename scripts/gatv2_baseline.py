"""Run the modern GATv2 (dynamic-attention) graph baseline on all four datasets,
5 seeds, matching the other graph baselines' protocol, and paired-bootstrap it
against PropMLP on identical per-(story,seed) test splits.

Adds a stronger, more modern propagation-GNN
baseline than RGCN under the text-free, CPU-bound protocol, on a controlled
subset. GATv2 (Brody et al., 2021) is the standard dynamic-attention GNN; we make
it type-aware over the four hyperedge/node families. Config (epochs in {40,80},
lr in {1e-3,5e-3}) is selected on the validation fold only; we report test
macro-F1 and a paired bootstrap (B=10000) over pooled per-story 0/1 losses vs
PropMLP. Positive mean_diff = PropMLP wins.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from docless.baselines.gatv2 import _GATv2
from docless.baselines.propmlp import train_propmlp
from docless.eval.stats import macro_f1, paired_bootstrap
from docless.hypergraph import build_hypergraph
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r10/gatv2"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [dict(epochs=40, lr=1e-3), dict(epochs=40, lr=5e-3), dict(epochs=80, lr=5e-3)]
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}
dev = torch.device("cpu")


def _train_eval(h_tr, y_tr, h_eval, y_eval, hidden, epochs, lr):
    model = _GATv2(in_dim=16, hidden_dim=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids), y_tr)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(h_eval, h_eval.story_to_burst_edges, h_eval.story_ids).argmax(-1).cpu().numpy()
    return preds, macro_f1(preds, y_eval.cpu().numpy())


def run(name, kw):
    dcfg = DataConfig(**kw)
    f1s = []
    gat_loss_pool, prop_loss_pool = [], []
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(dcfg, s)
        tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
        h_tr = build_hypergraph(tr, feature_dim=16, seed=s)
        h_va = build_hypergraph(va, feature_dim=16, seed=s + 1)
        h_te = build_hypergraph(te, feature_dim=16, seed=s + 2)
        y_tr, y_va, y_te = h_tr.story_labels, h_va.story_labels, h_te.story_labels
        # validation-only config selection
        best_cfg, best_val = None, -1.0
        for cfg in CONFIGS:
            _set_seed(s)
            _, vf1 = _train_eval(h_tr, y_tr, h_va, y_va, 64, cfg["epochs"], cfg["lr"])
            if vf1 > best_val:
                best_val, best_cfg = vf1, cfg
        _set_seed(s)
        preds, tf1 = _train_eval(h_tr, y_tr, h_te, y_te, 64, best_cfg["epochs"], best_cfg["lr"])
        f1s.append(round(float(tf1), 4))

        # paired comparison: PropMLP on the SAME split (same seed -> same te order)
        pr = train_propmlp(s, dcfg, device=dev)
        gat_labels = y_te.cpu().numpy()
        prop_preds = np.asarray(pr["preds"]); prop_labels = np.asarray(pr["labels"])
        assert len(prop_labels) == len(gat_labels) and (prop_labels == gat_labels).all(), \
            "PropMLP/GATv2 test splits are not aligned"
        gat_loss_pool.extend((preds != gat_labels).astype(float).tolist())
        prop_loss_pool.extend((prop_preds != prop_labels).astype(float).tolist())
        print(f"[{name}] seed {s}: gatv2={tf1:.4f} propmlp={pr['f1_all']:.4f} "
              f"(val-picked {best_cfg})", flush=True)

    pb = paired_bootstrap(np.asarray(prop_loss_pool), np.asarray(gat_loss_pool))
    agg = dict(
        f1_mean=round(float(np.mean(f1s)), 3), f1_std=round(float(np.std(f1s)), 3),
        per_seed=f1s,
        paired_vs_propmlp=dict(
            mean_acc_diff_propmlp_minus_gatv2=round(pb.mean_diff, 4),
            p_value=round(pb.p_value, 4),
            ci=[round(pb.ci_lo, 4), round(pb.ci_hi, 4)],
            n_pairs=len(gat_loss_pool),
            note="positive mean_diff => PropMLP lower loss (wins)",
        ),
    )
    json.dump(agg, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"== {name}: GATv2 {agg['f1_mean']} +/- {agg['f1_std']} | "
          f"paired vs PropMLP p={agg['paired_vs_propmlp']['p_value']} "
          f"(diff={agg['paired_vs_propmlp']['mean_acc_diff_propmlp_minus_gatv2']})", flush=True)
    return agg


if __name__ == "__main__":
    t0 = time.time()
    summary = {}
    for name, kw in DATASETS.items():
        summary[name] = run(name, kw)
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)
    print(f"ALL DONE in {round(time.time()-t0)}s", flush=True)
