"""Cascade-size-only logistic regression: the weakest honest baseline.

A one-feature control below the seven-feature summary, to
show how much of the text-free signal is just "how big did this cascade get".
Model: logistic regression on the single standardised feature log(1+n), where
n is the story's event count -- nothing else.  Same corpora, same
deterministic splits and seeds as every other event-list result, so it can be
read directly against the PropMLP and HistGBT rows.

Usage: PYTHONPATH=src python3 scripts/rev9_size_only.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rev2_power import DATASETS, SEEDS                                  # noqa: E402
from docless.eval.stats import macro_f1                                 # noqa: E402
from docless.train import (DataConfig, _load_cascades, _set_seed,       # noqa: E402
                           _stratified_split)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r15" / "size_only"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"feature": "log1p(event count) only, standardised on train",
           "model": "LogisticRegression(max_iter=2000)",
           "splits": "same _stratified_split(seed) as every other event-list result",
           "corpora": {}}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        f1s, accs = [], []
        for s in SEEDS:
            _set_seed(s)
            casc = _load_cascades(dcfg, s)
            tr, _va, te = _stratified_split(casc, dcfg.split_ratios, s)
            xtr = np.array([[np.log1p(len(c.events))] for c in tr])
            xte = np.array([[np.log1p(len(c.events))] for c in te])
            ytr = np.array([c.label for c in tr]); yte = np.array([c.label for c in te])
            sc = StandardScaler().fit(xtr)
            m = LogisticRegression(max_iter=2000, random_state=s).fit(sc.transform(xtr), ytr)
            pred = m.predict(sc.transform(xte))
            f1s.append(macro_f1(pred, yte)); accs.append(float((pred == yte).mean()))
        res["corpora"][name] = {
            "macro_f1_mean": round(float(np.mean(f1s)), 4),
            "macro_f1_std": round(float(np.std(f1s)), 4),
            "acc_mean": round(float(np.mean(accs)), 4),
            "per_seed_macro_f1": [round(float(x), 4) for x in f1s],
            "n_test": len(yte),
        }
        c = res["corpora"][name]
        print(f"{name:11s} size-only macro-F1 = {c['macro_f1_mean']:.4f} "
              f"(sd {c['macro_f1_std']:.4f}, n_test {c['n_test']})")
    (OUT / "size_only.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
