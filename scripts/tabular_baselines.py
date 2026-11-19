"""Non-neural tabular baselines on the 7-dim cascade-summary features
Adds LogReg / RandomForest /
GradientBoosting over the same features as PropMLP, on all four datasets
(GossipCop subsampled to 1500/class to match the main table). Also reports
feature importances from the tree models and a GossipCop subsample
sensitivity check (re-sample with seed +1000).
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r9/tabular_baselines"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}

FEATURE_NAMES = ["log_n", "log_users", "log_urls", "log_media", "log_pop", "echo", "log_spread"]


def _models():
    return {
        "logreg": LogisticRegression(max_iter=1000, C=1.0),
        "rf":     RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=1),
        "gbt":    GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0),
    }


def run_dataset(name, dcfg_kw, seed_offset=0):
    dcfg = DataConfig(**dcfg_kw)
    f1s = {m: [] for m in _models()}
    rf_importances = []
    for s in SEEDS:
        _set_seed(s + seed_offset)
        casc = _load_cascades(dcfg, s + seed_offset)
        tr, va, te = _stratified_split(casc, dcfg.split_ratios, s + seed_offset)
        x_tr = _features(tr); x_te = _features(te)
        y_tr = np.asarray([c.label for c in tr])
        y_te = np.asarray([c.label for c in te])
        sc = StandardScaler().fit(x_tr)
        x_tr_s = sc.transform(x_tr); x_te_s = sc.transform(x_te)
        for mname, m in _models().items():
            m.fit(x_tr_s, y_tr)
            preds = m.predict(x_te_s)
            f1s[mname].append(round(float(macro_f1(preds, y_te)), 4))
            if mname == "rf":
                rf_importances.append(m.feature_importances_.tolist())
    agg = {m: {"f1_mean": round(float(np.mean(v)), 3),
               "f1_std":  round(float(np.std(v)), 3),
               "per_seed": v} for m, v in f1s.items()}
    agg["rf_feature_importance_mean"] = [round(x, 3) for x in np.mean(rf_importances, axis=0).tolist()]
    agg["feature_names"] = FEATURE_NAMES
    return agg


if __name__ == "__main__":
    # main run on all 4 datasets
    out = {}
    for name, dcfg in DATASETS.items():
        out[name] = run_dataset(name, dcfg)
        print(f"[{name}]  logreg={out[name]['logreg']['f1_mean']:.3f}  "
              f"rf={out[name]['rf']['f1_mean']:.3f}  gbt={out[name]['gbt']['f1_mean']:.3f}",
              flush=True)
    json.dump(out, open(f"{OUT}/main.json", "w"), indent=2)

    # GossipCop subsample stability. NOTE: seed_offset shifts the seed passed to
    # BOTH _load_cascades (which draws the 1500/class balanced subsample via
    # Random(seed)) and _stratified_split, for ALL five seeds -- not just seed 0.
    # So the five base runs already vary the subsample as well as the split, and
    # this arm is a second independent FAMILY of five subsample+split draws,
    # not a re-subsample of one fixed split.
    print("=== gossipcop stability (alternate subsample, seed_offset=1000) ===", flush=True)
    alt = run_dataset("gossipcop_alt", DATASETS["gossipcop"], seed_offset=1000)
    print(f"[gossipcop alt]  logreg={alt['logreg']['f1_mean']:.3f}  "
          f"rf={alt['rf']['f1_mean']:.3f}  gbt={alt['gbt']['f1_mean']:.3f}", flush=True)
    json.dump({"original": out["gossipcop"], "alternate_subsample_seed+1000": alt},
              open(f"{OUT}/gossipcop_stability.json", "w"), indent=2)
    print("DONE", flush=True)
