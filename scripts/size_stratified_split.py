"""Cascade-size-stratified (extrapolation) split audit.

Motivation: the paper reports temporal, cross-dataset and
publisher-disjoint splits, but never a size-stratified one -- so it never tests
whether models trained on small cascades transfer to large ones, which is the
deployment case for early detection.

Protocol: order each powered corpus's stories by cascade size n, then
  small2large : train on the smallest 80%, validate next 10%, test largest 10%
  large2small : the mirror image (train largest 80%, test smallest 10%)
Both are deterministic given the corpus, so the 5 seeds average model
randomness only (as in the temporal split).  Random-split numbers on the same
corpora and features are the reference.

This is an EXTRAPOLATION protocol on a feature set the paper itself measures as
size/popularity-dominated (RF importances log_pop 0.318, log_spread 0.223,
log_n 0.186), so a large drop is the expected outcome.  We report whatever
comes out.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r11/size_split"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
RATIOS = (0.8, 0.1, 0.1)
DATASETS = {
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop": dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet",
                      max_per_class=1500),
}


def _size_split(cascades, reverse=False):
    """Deterministic split ordered by cascade size (number of events)."""
    order = sorted(range(len(cascades)),
                   key=lambda i: (len(cascades[i].events), i), reverse=reverse)
    n = len(order)
    n_tr = int(RATIOS[0] * n)
    n_va = int(RATIOS[1] * n)
    idx_tr, idx_va, idx_te = order[:n_tr], order[n_tr:n_tr + n_va], order[n_tr + n_va:]
    pick = lambda ix: [cascades[i] for i in ix]
    return pick(idx_tr), pick(idx_va), pick(idx_te)


def _fit_eval(x_tr, y_tr, x_te, y_te, seed, rich):
    out = {}
    # PropMLP / PropMLP+ (hidden 64, 25 full-batch epochs, Adam 1e-2)
    _set_seed(seed)
    mu = x_tr.mean(0, keepdims=True); sd = x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    xe = torch.tensor((x_te - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        out["propmlp+" if rich else "propmlp"] = float(macro_f1(m(xe).argmax(-1).numpy(), y_te))
    if not rich:
        sc = StandardScaler().fit(x_tr)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1)
        rf.fit(sc.transform(x_tr), y_tr)
        out["rf"] = float(macro_f1(rf.predict(sc.transform(x_te)), y_te))
    else:
        hg = HistGradientBoostingClassifier(random_state=seed)
        hg.fit(x_tr, y_tr)
        out["histgbt"] = float(macro_f1(hg.predict(x_te), y_te))
    return out


def run(name, dcfg_kw):
    dcfg = DataConfig(**dcfg_kw)
    modes = {}
    stats = {}
    for mode in ("random", "small2large", "large2small"):
        acc = {}
        for s in SEEDS:
            _set_seed(s)
            casc = _load_cascades(dcfg, s)
            if mode == "random":
                tr, _va, te = _stratified_split(casc, RATIOS, s)
            else:
                tr, _va, te = _size_split(casc, reverse=(mode == "large2small"))
            y_tr = np.asarray([c.label for c in tr]); y_te = np.asarray([c.label for c in te])
            for rich in (False, True):
                r = _fit_eval(_features(tr, rich=rich), y_tr,
                              _features(te, rich=rich), y_te, s, rich)
                for k, v in r.items():
                    acc.setdefault(k, []).append(round(v, 4))
            if s == 0 and mode != "random":
                ntr = [len(c.events) for c in tr]; nte = [len(c.events) for c in te]
                stats[mode] = {
                    "train_size_median": float(np.median(ntr)),
                    "train_size_max": int(max(ntr)),
                    "test_size_median": float(np.median(nte)),
                    "test_size_min": int(min(nte)),
                    "test_prior_misinfo": round(float(y_te.mean()), 3),
                    "n_test": int(len(y_te)),
                }
        modes[mode] = {k: {"f1_mean": round(float(np.mean(v)), 3),
                           "f1_std": round(float(np.std(v)), 3),
                           "per_seed": v} for k, v in acc.items()}
    out = {"modes": modes, "split_stats": stats}
    for mode in ("small2large", "large2small"):
        out.setdefault("deltas_vs_random", {})[mode] = {
            k: round(modes[mode][k]["f1_mean"] - modes["random"][k]["f1_mean"], 3)
            for k in modes["random"]
        }
    return out


if __name__ == "__main__":
    res = {}
    for name, dcfg in DATASETS.items():
        res[name] = run(name, dcfg)
        for mode in ("random", "small2large", "large2small"):
            cells = " ".join(f"{k}={v['f1_mean']:.3f}" for k, v in res[name]["modes"][mode].items())
            print(f"[{name}/{mode}] {cells}", flush=True)
        print(f"[{name}] deltas vs random: {res[name]['deltas_vs_random']}", flush=True)
    json.dump(res, open(f"{OUT}/size_split.json", "w"), indent=2)
    print("DONE", flush=True)
