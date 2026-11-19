"""Run the heterogeneous RGCN baseline (validation-only tuned) on all four
datasets, 5 seeds, matching the other graph baselines' protocol. Tuning (epochs in {25,40}, lr in {1e-3,5e-4}) is selected on the
validation fold only; we report test macro-F1.
"""
from __future__ import annotations

import json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from docless.baselines.rgcn import _RGCN
from docless.eval.stats import macro_f1
from docless.hypergraph import build_hypergraph
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r9/rgcn"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [dict(epochs=25, lr=1e-3), dict(epochs=40, lr=1e-3), dict(epochs=40, lr=5e-4)]
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}
dev = torch.device("cpu")


def _train_eval(h_tr, y_tr, h_eval, y_eval, hidden, epochs, lr):
    _RGCN  # noqa
    model = _RGCN(in_dim=16, hidden_dim=hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids), y_tr)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(h_eval, h_eval.story_to_burst_edges, h_eval.story_ids).argmax(-1).cpu().numpy()
    return macro_f1(preds, y_eval.cpu().numpy())


def run(name, kw):
    dcfg = DataConfig(**kw)
    f1s = []
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(dcfg, s)
        tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
        h_tr = build_hypergraph(tr, feature_dim=16, seed=s)
        h_va = build_hypergraph(va, feature_dim=16, seed=s + 1)
        h_te = build_hypergraph(te, feature_dim=16, seed=s + 2)
        y_tr, y_va, y_te = h_tr.story_labels, h_va.story_labels, h_te.story_labels
        # validation-only selection
        best_cfg, best_val = None, -1.0
        for cfg in CONFIGS:
            _set_seed(s)
            vf1 = _train_eval(h_tr, y_tr, h_va, y_va, 64, cfg["epochs"], cfg["lr"])
            if vf1 > best_val:
                best_val, best_cfg = vf1, cfg
        _set_seed(s)
        tf1 = _train_eval(h_tr, y_tr, h_te, y_te, 64, best_cfg["epochs"], best_cfg["lr"])
        f1s.append(round(float(tf1), 4))
        print(f"[{name}] seed {s}: test={tf1:.4f} (val-picked {best_cfg})", flush=True)
    agg = dict(f1_mean=round(float(np.mean(f1s)), 3), f1_std=round(float(np.std(f1s)), 3), per_seed=f1s)
    json.dump(agg, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"== {name}: {agg['f1_mean']} +/- {agg['f1_std']}", flush=True)
    return agg


if __name__ == "__main__":
    t0 = time.time()
    for name, kw in DATASETS.items():
        run(name, kw)
    print(f"ALL DONE in {round(time.time()-t0)}s", flush=True)
