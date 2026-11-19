"""Stronger tabular baselines:
(1) A stronger, tuned gradient-boosted-tree baseline (sklearn
    HistGradientBoostingClassifier, the histogram/LightGBM-style booster) on the
    7-dim and 15-dim cascade-summary features, all four datasets, 5 seeds, with a
    small val-selected grid.
(2) An "early-window-only features" baseline: recompute *all* seven summary
    features on the early window only (not just P,E), to quantify how much of the
    strong baseline is full-cascade (non-early-warning) signal.
Cheap CPU-only; mirrors tabular_baselines.py splits exactly.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r9/stanford2"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}
# small tuning grid, selected on the validation fold
GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 31) for l2 in (0.0, 1.0)]


def _early_features(cascades, rich=False):
    """Same 7/15-dim features but computed on early-window events only."""
    trimmed = []
    for c in cascades:
        ev = c.early_events()
        trimmed.append(dataclasses.replace(c, events=list(ev), early_window_end=c.early_window_end))
    return _features(trimmed, rich=rich)


def _tuned_histgbt(x_tr, y_tr, x_va, y_va):
    best, best_f1 = None, -1.0
    for hp in GRID:
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp)
        m.fit(x_tr, y_tr)
        f1 = macro_f1(m.predict(x_va), y_va)
        if f1 > best_f1:
            best_f1, best = f1, hp
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **best)
    m.fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))
    return m


def run_dataset(name, dcfg_kw):
    dcfg = DataConfig(**dcfg_kw)
    res = {"histgbt7": [], "histgbt15": [], "early7_rf": [], "early7_mlp_full_ref": []}
    from sklearn.ensemble import RandomForestClassifier
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(dcfg, s)
        tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
        y_tr = np.asarray([c.label for c in tr]); y_va = np.asarray([c.label for c in va])
        y_te = np.asarray([c.label for c in te])
        # --- tuned HistGBT on 7 and 15 dims ---
        for tag, rich in [("histgbt7", False), ("histgbt15", True)]:
            xtr = _features(tr, rich=rich); xva = _features(va, rich=rich); xte = _features(te, rich=rich)
            sc = StandardScaler().fit(xtr)
            m = _tuned_histgbt(sc.transform(xtr), y_tr, sc.transform(xva), y_va)
            res[tag].append(round(float(macro_f1(m.predict(sc.transform(xte)), y_te)), 4))
        # --- early-window-only 7-dim features, RandomForest ---
        xtr_e = _early_features(tr); xte_e = _early_features(te)
        sc = StandardScaler().fit(xtr_e)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1)
        rf.fit(sc.transform(xtr_e), y_tr)
        res["early7_rf"].append(round(float(macro_f1(rf.predict(sc.transform(xte_e)), y_te)), 4))
        # full-cascade RF reference on identical split
        xtr_f = _features(tr); xte_f = _features(te)
        scf = StandardScaler().fit(xtr_f)
        rff = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1)
        rff.fit(scf.transform(xtr_f), y_tr)
        res["early7_mlp_full_ref"].append(round(float(macro_f1(rff.predict(scf.transform(xte_f)), y_te)), 4))
    agg = {k: dict(mean=round(float(np.mean(v)), 3), std=round(float(np.std(v)), 3), per_seed=v)
           for k, v in res.items()}
    json.dump(agg, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"[{name}] " + " ".join(f"{k}={agg[k]['mean']}" for k in agg), flush=True)
    return agg


if __name__ == "__main__":
    for name, kw in DATASETS.items():
        run_dataset(name, kw)
    print("ALL DONE", flush=True)
