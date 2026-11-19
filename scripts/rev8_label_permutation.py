"""LEAK: label-permutation negative control for the tabular baselines.

Question: gradient-boosted trees are strong, but can they
exploit a *label-independent* artifact of the split or the features? The paper's
nearest existing check (10% label flip on ReCOVery) is a noise-sensitivity test,
not a leakage control, and nothing under scripts/ trains on shuffled labels.

Method. Splits, seeds, features and TEST labels are untouched; only the TRAINING
labels are permuted, within the training fold, per seed. HistGBT on the 15-d
summary and RandomForest on the 7-d summary are then retrained and scored on the
untouched test fold. Four identity-free corpora, 5 seeds.

PRE-REGISTERED READING, fixed before the run:
  * EXPECTED (the outcome that supports the paper): both models collapse into
    the no-signal band -- bounded below by the majority floor 0.333 that the
    main text already reports for these corpora, and above by roughly 0.50 for a
    balanced binary fold -- losing essentially the whole gap to their real
    scores. This belongs in the supplement.
  * FAILURE MODE: any corpus materially above 0.50 under permuted training
    labels is evidence of a label-independent artifact in the split or the
    features. It goes in the MAIN text as a threat to validity, and the
    configuration is NOT re-run, re-seeded or re-shuffled until it comes back
    clean.
  * A borderline result (0.50-0.55 on one corpus, 5-seed interval touching 0.50)
    is reported as borderline with the interval shown, not resolved by adding
    seeds.

Usage:  PYTHONPATH=src .venv/bin/python scripts/rev8_label_permutation.py
Output: outputs/r13/label_permutation/label_permutation.json
"""
from __future__ import annotations

import json
import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r13/label_permutation"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}


def _fit_score(x_tr, y_tr, x_te, y_te, seed, kind):
    sc = StandardScaler().fit(x_tr)
    a, b = sc.transform(x_tr), sc.transform(x_te)
    if kind == "histgbt15":
        m = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1,
                                           random_state=seed)
    else:
        m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1)
    m.fit(a, y_tr)
    return round(float(macro_f1(m.predict(b), y_te)), 4)


def run(name, kw):
    real = {"histgbt15": [], "rf7": []}
    perm = {"histgbt15": [], "rf7": []}
    n_test = []
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(DataConfig(**kw), s)
        tr, _va, te = _stratified_split(casc, (0.8, 0.1, 0.1), s)
        y_tr = np.asarray([c.label for c in tr])
        y_te = np.asarray([c.label for c in te])
        n_test.append(int(len(y_te)))
        # permute TRAINING labels only; splits, features and test labels untouched
        rng = np.random.default_rng(10_000 + s)
        y_tr_perm = rng.permutation(y_tr)
        x_tr15, x_te15 = _features(tr, rich=True), _features(te, rich=True)
        x_tr7, x_te7 = _features(tr), _features(te)
        real["histgbt15"].append(_fit_score(x_tr15, y_tr, x_te15, y_te, s, "histgbt15"))
        perm["histgbt15"].append(_fit_score(x_tr15, y_tr_perm, x_te15, y_te, s, "histgbt15"))
        real["rf7"].append(_fit_score(x_tr7, y_tr, x_te7, y_te, s, "rf7"))
        perm["rf7"].append(_fit_score(x_tr7, y_tr_perm, x_te7, y_te, s, "rf7"))
        print(f"[{name}] seed {s}: histgbt15 real={real['histgbt15'][-1]:.3f} "
              f"perm={perm['histgbt15'][-1]:.3f} | rf7 real={real['rf7'][-1]:.3f} "
              f"perm={perm['rf7'][-1]:.3f}", flush=True)
    out = {"n_test_per_seed": n_test, "seeds": SEEDS}
    for m in real:
        r, p = np.asarray(real[m]), np.asarray(perm[m])
        # 5-seed t interval on the permuted-label score
        se = p.std(ddof=1) / np.sqrt(len(p))
        tcrit = 2.776  # t_{0.975, df=4}
        lo, hi = float(p.mean() - tcrit * se), float(p.mean() + tcrit * se)
        out[m] = {"real_mean": round(float(r.mean()), 4), "real_per_seed": real[m],
                  "perm_mean": round(float(p.mean()), 4), "perm_per_seed": perm[m],
                  "perm_std": round(float(p.std(ddof=0)), 4),
                  "perm_ci95": [round(lo, 4), round(hi, 4)],
                  "perm_ci95_contains_0.50": bool(lo <= 0.50 <= hi),
                  "perm_min": round(float(p.min()), 4), "perm_max": round(float(p.max()), 4),
                  "collapse": round(float(r.mean() - p.mean()), 4),
                  "perm_mean_above_0.50": bool(p.mean() > 0.50)}
    return out


def main():
    res = {"protocol": "training labels permuted within the training fold only; splits, "
                       "seeds, features and test labels untouched",
           "majority_floor_reported_in_paper": 0.333,
           "prereg": "expected null: permuted-label macro-F1 in [0.333, ~0.50]; any corpus "
                     "materially above 0.50 is a main-text threat and must not be re-run",
           "results": {}}
    for name, kw in DATASETS.items():
        res["results"][name] = run(name, kw)
    # the pre-registered failure test is "materially above 0.50", i.e. an interval
    # that excludes chance -- not a point estimate a hair over it
    flagged = [k for k, v in res["results"].items()
               if not v["histgbt15"]["perm_ci95_contains_0.50"]
               or not v["rf7"]["perm_ci95_contains_0.50"]]
    res["corpora_whose_permuted_ci_excludes_0.50"] = flagged
    res["corpora_whose_permuted_point_estimate_exceeds_0.50"] = [
        k for k, v in res["results"].items()
        if v["histgbt15"]["perm_mean_above_0.50"] or v["rf7"]["perm_mean_above_0.50"]]
    with open(f"{OUT}/label_permutation.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nflagged (permuted-label 95% CI excludes 0.50):", flagged or "none")
    for k, v in res["results"].items():
        print(f"  {k:11s} histgbt15 {v['histgbt15']['real_mean']:.3f} -> "
              f"{v['histgbt15']['perm_mean']:.3f}   rf7 {v['rf7']['real_mean']:.3f} -> "
              f"{v['rf7']['perm_mean']:.3f}")


if __name__ == "__main__":
    main()
