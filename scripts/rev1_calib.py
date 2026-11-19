"""Calibration + ranking under NATURAL prevalence.

Full per-model table: ROC-AUC, PR-AUC, Brier, ECE on each corpus's natural
class prevalence (balanced=False), 5 seeds. Adds post-hoc ISOTONIC calibration
(fit on the validation fold) for the over-confident HistGBT, reporting ECE/Brier
before vs after. Resolves the prose-only calibration ranges and the HistGBT
ECE=0.21 contradiction.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _MLP, _features
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r10/calibration"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid", balanced=False, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery", balanced=False, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=False, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=False, data_root="data/raw/fakenewsnet", max_per_class=4000),
}


def _ece(p, y, n_bins=10):
    p = np.asarray(p); y = np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for i in range(n_bins):
        m = (p > bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0: continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(e)


def _propmlp_proba(tr, va, te, rich, seed):
    xtr, xva, xte = _features(tr, rich=rich), _features(va, rich=rich), _features(te, rich=rich)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    f = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32)
    ytr = torch.tensor([c.label for c in tr], dtype=torch.long)
    m = _MLP(in_dim=xtr.shape[1], hidden=64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(f(xtr)), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pv = F.softmax(m(f(xva)), -1)[:, 1].numpy(); pt = F.softmax(m(f(xte)), -1)[:, 1].numpy()
    return pv, pt


def _metrics(p, y):
    y = np.asarray(y)
    return dict(roc=float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
                pr=float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
                brier=float(brier_score_loss(y, p)), ece=_ece(p, y))


def run(name, kw):
    res = {m: {k: [] for k in ["roc", "pr", "brier", "ece"]} for m in ["propmlp", "propmlp+", "rf", "histgbt", "histgbt_iso"]}
    for s in SEEDS:
        _set_seed(s); casc = _load_cascades(DataConfig(**kw), s)
        tr, va, te = _stratified_split(casc, (0.8, 0.1, 0.1), s)
        ytr = np.array([c.label for c in tr]); yva = np.array([c.label for c in va]); yte = np.array([c.label for c in te])
        # PropMLP / PropMLP+
        for mn, rich in [("propmlp", False), ("propmlp+", True)]:
            _, pt = _propmlp_proba(tr, va, te, rich, s)
            for k, v in _metrics(pt, yte).items(): res[mn][k].append(v)
        # tabular
        xtr, xva, xte = _features(tr), _features(va), _features(te)
        sc = StandardScaler().fit(xtr)
        Xtr, Xva, Xte = sc.transform(xtr), sc.transform(xva), sc.transform(xte)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(Xtr, ytr)
        for k, v in _metrics(rf.predict_proba(Xte)[:, 1], yte).items(): res["rf"][k].append(v)
        hg = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=s).fit(Xtr, ytr)
        pv, pt = hg.predict_proba(Xva)[:, 1], hg.predict_proba(Xte)[:, 1]
        for k, v in _metrics(pt, yte).items(): res["histgbt"][k].append(v)
        # isotonic post-hoc on validation fold
        if len(set(yva)) > 1:
            iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yva)
            pt_iso = iso.predict(pt)
            for k, v in _metrics(pt_iso, yte).items(): res["histgbt_iso"][k].append(v)
    agg = {m: {k: round(float(np.nanmean(v)), 3) for k, v in d.items() if v} for m, d in res.items()}
    print(f"[{name}] " + " | ".join(f"{m}: roc{agg[m].get('roc')} ece{agg[m].get('ece')} brier{agg[m].get('brier')}" for m in agg), flush=True)
    return agg


if __name__ == "__main__":
    out = {n: run(n, kw) for n, kw in DATASETS.items()}
    json.dump(out, open(f"{OUT}/calibration.json", "w"), indent=2)
    print("DONE", flush=True)
