"""Temporal-drift analysis + a panel of label-free mitigations (superset of
temporal_drift.py). For each powered dataset we diagnose per-feature covariate
shift (KS, train-standardised mean shift), class-prior shift, and era spans,
then evaluate PropMLP (7d), PropMLP+ (15d), and RandomForest (7d) under the
future-test split (5 seeds) with six normalisation/adaptation strategies:

  baseline  -- standardise with TRAIN mean/std (paper's current protocol)
  testtime  -- standardise the future era with ITS OWN mean/std
  quantile  -- per-era quantile transform to N(0,1) (marginal alignment)
  recency   -- train-stat standardise + exponential recency weights on training
               stories (emphasise the era closest to deployment)
  impweight -- covariate-shift importance weighting: a logistic domain
               classifier estimates the train->test density ratio, used as
               training sample weights (Shimodaira/Sugiyama)
  coral     -- CORAL second-order alignment of train features to the future-era
               covariance (Sun et al., 2016)

All six are label-free with respect to the future era (they use only future-era
*feature* values, observable at deployment), so each is a legitimate test-time
correction. We report which mitigation recovers the most future-test macro-F1.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _temporal_split

OUT = "outputs/r10/temporal_drift"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
FEATURE_NAMES = ["log_n", "log_users", "log_urls", "log_media", "log_pop", "echo", "log_spread"]
DATASETS = {
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}
MITIGATIONS = ["baseline", "testtime", "quantile", "recency", "impweight", "coral"]


def _matrix_pow(C, p):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    return (V * (w ** p)) @ V.T


def _story_time(c):
    return min((e.timestamp for e in c.events), default=0.0)


def _prep(x_tr, x_te, tr, mode):
    """Return (x_tr_n, x_te_n, sample_weight_or_None)."""
    if mode == "baseline":
        sc = StandardScaler().fit(x_tr)
        return sc.transform(x_tr), sc.transform(x_te), None
    if mode == "testtime":
        return StandardScaler().fit_transform(x_tr), StandardScaler().fit_transform(x_te), None
    if mode == "quantile":
        qtr = QuantileTransformer(output_distribution="normal",
                                  n_quantiles=min(len(x_tr), 200), random_state=0)
        qte = QuantileTransformer(output_distribution="normal",
                                  n_quantiles=min(len(x_te), 200), random_state=0)
        return qtr.fit_transform(x_tr), qte.fit_transform(x_te), None
    # the remaining modes standardise with train stats, then adapt
    sc = StandardScaler().fit(x_tr)
    a, b = sc.transform(x_tr), sc.transform(x_te)
    if mode == "recency":
        t = np.asarray([_story_time(c) for c in tr], dtype=np.float64)
        if t.max() > t.min():
            half = (t.max() - t.min()) / 2.0
            w = 0.5 ** ((t.max() - t) / (half + 1e-9))
        else:
            w = np.ones(len(tr))
        w = w / w.mean()
        return a, b, w
    if mode == "impweight":
        xd = np.vstack([a, b]); yd = np.r_[np.zeros(len(a)), np.ones(len(b))]
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(xd, yd)
        p = np.clip(clf.predict_proba(a)[:, 1], 1e-3, 1 - 1e-3)
        w = np.clip(p / (1.0 - p), 0.1, 10.0)
        w = w / w.mean()
        return a, b, w
    if mode == "coral":
        Cs = np.cov(a, rowvar=False) + np.eye(a.shape[1])
        Ct = np.cov(b, rowvar=False) + np.eye(b.shape[1])
        A = _matrix_pow(Cs, -0.5) @ _matrix_pow(Ct, 0.5)
        return a @ A, b, None
    raise ValueError(mode)


def _fit_predict(model_name, x_tr_n, y_tr, x_te_n, w, seed):
    if model_name == "rf":
        m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1)
        m.fit(x_tr_n, y_tr, sample_weight=w)
        return m.predict(x_te_n)
    xt = torch.tensor(x_tr_n, dtype=torch.float32)
    xe = torch.tensor(x_te_n, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    wt = torch.tensor(w, dtype=torch.float32) if w is not None else None
    model = _MLP(in_dim=xt.shape[1], hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(25):
        model.train(); opt.zero_grad()
        if wt is None:
            loss = F.cross_entropy(model(xt), yt)
        else:
            loss = (F.cross_entropy(model(xt), yt, reduction="none") * wt).mean()
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(xe).argmax(-1).numpy()


def _eval(model_name, dcfg, mode, rich):
    vals = []
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(dcfg, s)
        tr, _va, te = _temporal_split(casc, dcfg.split_ratios, s)
        x_tr = _features(tr, rich=rich); x_te = _features(te, rich=rich)
        y_tr = np.asarray([c.label for c in tr]); y_te = np.asarray([c.label for c in te])
        x_tr_n, x_te_n, w = _prep(x_tr, x_te, tr, mode)
        preds = _fit_predict(model_name, x_tr_n, y_tr, x_te_n, w, s)
        vals.append(float(macro_f1(preds, y_te)))
    return dict(f1_mean=round(float(np.mean(vals)), 3),
                f1_std=round(float(np.std(vals)), 3),
                per_seed=[round(v, 4) for v in vals])


def _drift(dcfg):
    casc = _load_cascades(dcfg, 0)
    tr, _va, te = _temporal_split(casc, dcfg.split_ratios, 0)
    x_tr = _features(tr); x_te = _features(te)
    per_feat = []
    for j, nm in enumerate(FEATURE_NAMES):
        a, b = x_tr[:, j], x_te[:, j]
        per_feat.append(dict(feature=nm, ks=round(float(ks_2samp(a, b).statistic), 3),
                             std_mean_shift=round(float((b.mean() - a.mean()) / (a.std() + 1e-9)), 3)))
    per_feat.sort(key=lambda d: -d["ks"])
    y_tr = np.asarray([c.label for c in tr]); y_te = np.asarray([c.label for c in te])
    return dict(n_train=len(tr), n_test=len(te),
                train_misinfo_rate=round(float(y_tr.mean()), 3),
                test_misinfo_rate=round(float(y_te.mean()), 3),
                per_feature=per_feat)


if __name__ == "__main__":
    out = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        print(f"=== {name}: drift diagnosis ===", flush=True)
        drift = _drift(dcfg)
        for f in drift["per_feature"]:
            print(f"  {f['feature']:11s} KS={f['ks']:.3f}  mean_shift={f['std_mean_shift']:+.3f}", flush=True)
        print(f"  class prior train->test: {drift['train_misinfo_rate']}->{drift['test_misinfo_rate']}", flush=True)
        models = [("propmlp", False), ("propmlp_plus", True), ("rf", False)]
        results, best = {}, {}
        for mname, rich in models:
            results[mname] = {m: _eval(mname, dcfg, m, rich) for m in MITIGATIONS}
            base = results[mname]["baseline"]["f1_mean"]
            deltas = {m: round(results[mname][m]["f1_mean"] - base, 3)
                      for m in MITIGATIONS if m != "baseline"}
            bm = max(deltas, key=deltas.get)
            best[mname] = dict(baseline=base, deltas=deltas, best_mitigation=bm,
                               best_f1=results[mname][bm]["f1_mean"], best_gain=deltas[bm])
            line = "  ".join(f"{m}={results[mname][m]['f1_mean']:.3f}" for m in MITIGATIONS)
            print(f"  {mname:13s} {line}  -> best {bm} ({deltas[bm]:+.3f})", flush=True)
        out[name] = dict(drift=drift, results=results, best=best)
        json.dump(out[name], open(f"{OUT}/{name}_v2.json", "w"), indent=2)
    json.dump(out, open(f"{OUT}/summary_v2.json", "w"), indent=2)
    print("ALL DONE", flush=True)
