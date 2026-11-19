"""PropMLP tuning parity.

Checks that PropMLP's lead is not a tuning artefact: we give
PropMLP and PropMLP+ the SAME validation-selected epoch x learning-rate grid the
graph baselines received (epochs in {25,40,80}, lr in {5e-4,1e-3,5e-3}), plus
PropMLP's default operating point (25 ep, 1e-2), selecting on the validation fold
only. We report the val-selected test macro-F1 next to the default so the reader
can see the lead is invariant to tuning. 5 seeds, identical splits/features.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch, torch.nn.functional as F

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r10/propmlp_parity"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
GRID = [(25, 5e-4), (40, 1e-3), (40, 5e-3), (80, 5e-3), (25, 1e-2)]  # GNN grid + PropMLP default
DATASETS = {
    "coaid":      dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}


def _fit(xtr, ytr, xev, epochs, lr, seed):
    _set_seed(seed)
    xt = torch.tensor(xtr, dtype=torch.float32); xe = torch.tensor(xev, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(xe).argmax(-1).numpy()


def run():
    out = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw); out[name] = {}
        for rich, key in [(False, "propmlp"), (True, "propmlp+")]:
            default_f1, sel_f1 = [], []
            for s in SEEDS:
                _set_seed(s); casc = _load_cascades(dcfg, s)
                tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
                xtr = _features(tr, rich=rich); xva = _features(va, rich=rich); xte = _features(te, rich=rich)
                mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
                Xtr, Xva, Xte = (xtr - mu) / sd, (xva - mu) / sd, (xte - mu) / sd
                ytr = np.array([c.label for c in tr]); yva = np.array([c.label for c in va]); yte = np.array([c.label for c in te])
                # default operating point
                default_f1.append(macro_f1(_fit(Xtr, ytr, Xte, 25, 1e-2, s), yte))
                # val-selected over the grid
                best, bestv = None, -1
                for ep, lr in GRID:
                    vf1 = macro_f1(_fit(Xtr, ytr, Xva, ep, lr, s), yva)
                    if vf1 > bestv: bestv, best = vf1, (ep, lr)
                sel_f1.append(macro_f1(_fit(Xtr, ytr, Xte, best[0], best[1], s), yte))
            out[name][key] = dict(default=round(float(np.mean(default_f1)), 3),
                                  val_selected=round(float(np.mean(sel_f1)), 3))
        d = out[name]
        print(f"[{name}] PropMLP default={d['propmlp']['default']} val-sel={d['propmlp']['val_selected']} | "
              f"PropMLP+ default={d['propmlp+']['default']} val-sel={d['propmlp+']['val_selected']}", flush=True)
        json.dump(out, open(f"{OUT}/propmlp_parity.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
