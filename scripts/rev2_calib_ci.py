"""ECE with bootstrap CIs.

ECE on n~60-90 per-seed test sets is high-variance, so we report CIs. We recompute ECE under natural prevalence (same protocol as rev1_calib),
pool the per-test (prob, label) pairs across 5 seeds, and report a 95% bootstrap
CI (resampling test examples). The point estimate decides the corrected bold; the
CI shows how much of the table is separable at all.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _MLP, _features
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r10/calib_ci"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid", balanced=False, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery", balanced=False, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=False, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=False, data_root="data/raw/fakenewsnet", max_per_class=4000),
}


def _ece_vec(p, y, n_bins=10):
    """per-example contribution is not additive; recompute ECE on a subsample."""
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0; n = len(y)
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0: continue
        e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(e)


def _ece_ci(p, y, B=5000, seed=0):
    p, y = np.asarray(p), np.asarray(y); n = len(y)
    rng = np.random.default_rng(seed)
    pt = _ece_vec(p, y)
    samp = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n); samp[b] = _ece_vec(p[idx], y[idx])
    return pt, [float(np.quantile(samp, .025)), float(np.quantile(samp, .975))]


def _propmlp_proba(tr, te, rich, seed):
    xtr, xte = _features(tr, rich=rich), _features(te, rich=rich)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    f = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor([c.label for c in tr], dtype=torch.long)
    _set_seed(seed); m = _MLP(in_dim=xtr.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(f(xtr)), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return F.softmax(m(f(xte)), -1)[:, 1].numpy()


def run():
    out = {}
    for name, kw in DATASETS.items():
        pool = {m: ([], []) for m in ["propmlp", "propmlp+", "rf", "histgbt", "histgbt_iso"]}
        for s in SEEDS:
            _set_seed(s); casc = _load_cascades(DataConfig(**kw), s)
            tr, va, te = _stratified_split(casc, (0.8, 0.1, 0.1), s)
            ytr = np.array([c.label for c in tr]); yva = np.array([c.label for c in va]); yte = np.array([c.label for c in te])
            for mn, rich in [("propmlp", False), ("propmlp+", True)]:
                pool[mn][0].extend(_propmlp_proba(tr, te, rich, s)); pool[mn][1].extend(yte)
            xtr, xva, xte = _features(tr), _features(va), _features(te)
            sc = StandardScaler().fit(xtr); Xtr, Xva, Xte = sc.transform(xtr), sc.transform(xva), sc.transform(xte)
            rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(Xtr, ytr)
            pool["rf"][0].extend(rf.predict_proba(Xte)[:, 1]); pool["rf"][1].extend(yte)
            hg = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=s).fit(Xtr, ytr)
            pv, pt = hg.predict_proba(Xva)[:, 1], hg.predict_proba(Xte)[:, 1]
            pool["histgbt"][0].extend(pt); pool["histgbt"][1].extend(yte)
            if len(set(yva)) > 1:
                iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yva)
                pool["histgbt_iso"][0].extend(iso.predict(pt)); pool["histgbt_iso"][1].extend(yte)
        out[name] = {}
        for m, (p, y) in pool.items():
            if not p: continue
            pt, ci = _ece_ci(np.array(p), np.array(y), seed=0)
            out[name][m] = dict(ece=round(pt, 3), ci95=[round(c, 3) for c in ci])
        best = min(out[name], key=lambda m: out[name][m]["ece"])
        print(f"[{name}] best-ECE={best}  " + "  ".join(
            f"{m}={out[name][m]['ece']}[{out[name][m]['ci95'][0]},{out[name][m]['ci95'][1]}]" for m in out[name]), flush=True)
        json.dump(out, open(f"{OUT}/calib_ci.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
