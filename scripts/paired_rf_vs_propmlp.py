"""Paired bootstrap RF vs PropMLP on all four datasets.

Same 7 cascade-summary features, identical stratified splits per seed, RF/
PropMLP train on the same train, predict on the same test; per-story 0/1
losses pooled across 5 seeds; paired_bootstrap(propmlp_losses, rf_losses)
gives mean(loss_propmlp - loss_rf) positive => RF wins.
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _features
from docless.baselines import train_propmlp
from docless.eval.stats import paired_bootstrap
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split
import torch

OUT = "outputs/r9/paired_rf"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}


def run_dataset(name, dcfg_kw):
    dcfg = DataConfig(**dcfg_kw)
    rf_losses_all, pm_losses_all = [], []
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(dcfg, s)
        tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
        x_tr = _features(tr); x_te = _features(te)
        y_tr = np.asarray([c.label for c in tr])
        y_te = np.asarray([c.label for c in te])
        sc = StandardScaler().fit(x_tr)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=1)
        rf.fit(sc.transform(x_tr), y_tr)
        rf_preds = rf.predict(sc.transform(x_te))
        rf_losses_all.append((rf_preds != y_te).astype(float))
        pm = train_propmlp(s, dcfg, hidden_dim=64, epochs=25, device=torch.device("cpu"))
        pm_preds = np.asarray(pm["preds"])
        pm_labels = np.asarray(pm["labels"])
        # alignment sanity: same labels mean same test set order
        assert np.array_equal(pm_labels, y_te), f"split misalignment on {name} seed {s}"
        pm_losses_all.append((pm_preds != pm_labels).astype(float))
    rf_l = np.concatenate(rf_losses_all); pm_l = np.concatenate(pm_losses_all)
    r = paired_bootstrap(pm_l, rf_l)
    return {
        "rf_acc": round(float(1 - rf_l.mean()), 4),
        "propmlp_acc": round(float(1 - pm_l.mean()), 4),
        "mean_loss_diff_propmlp_minus_rf": round(r.mean_diff, 4),
        "p_value": round(r.p_value, 4),
        "ci_lo": round(r.ci_lo, 4), "ci_hi": round(r.ci_hi, 4),
        "n_stories": int(rf_l.shape[0]),
    }


if __name__ == "__main__":
    out = {}
    for name, dcfg in DATASETS.items():
        out[name] = run_dataset(name, dcfg)
        print(f"[{name}]  RF={out[name]['rf_acc']:.3f}  PropMLP={out[name]['propmlp_acc']:.3f}  "
              f"Δ={out[name]['mean_loss_diff_propmlp_minus_rf']:+.4f}  p={out[name]['p_value']:.3f}  "
              f"n={out[name]['n_stories']}", flush=True)
    json.dump(out, open(f"{OUT}/paired.json", "w"), indent=2)
    print("DONE", flush=True)
