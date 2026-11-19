"""Mitigation-delta CIs + a validation-fold selection rule.

#2  For every label-free drift correction, report a paired story-level bootstrap
    95% CI on the DELTA itself (corrected - baseline on the same future-test
    stories), not only the CI on the baseline level. If no delta excludes zero,
    the '+0.05 recovery' must be demoted to inconclusive.

#3  A label-free SELECTION RULE that makes the correction deployable: choose
    which correction to apply using only the *validation era* (historical,
    labelled, no future-test labels), then apply it to the future-test era.
    We report baseline / oracle-best / val-selected so the reader can see how
    much of the oracle gain a deployable rule actually captures.

Temporal split: earliest 80% train, next 10% validation era, latest 10% test
era (deterministic; seeds vary model init only). Corrections touch only feature
values (or training weights) and never read future-test labels.
"""
from __future__ import annotations
import json, os
from collections import defaultdict
import numpy as np
import torch, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _temporal_split

OUT = "outputs/r10/mitig_delta"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CORR = ["base", "test_t", "quant", "impwt", "coral"]
DATASETS = {
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}


def _msqrt(C, p):
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None); return (V * w ** p) @ V.T


def _fit_predict(model, Xtr, ytr, Xte, seed, sample_w=None):
    if model == "rf":
        m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1)
        m.fit(Xtr, ytr, sample_weight=sample_w); return m.predict(Xte)
    _set_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32); xe = torch.tensor(Xte, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    w = None if sample_w is None else torch.tensor(sample_w, dtype=torch.float32)
    m = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad()
        logits = m(xt)
        loss = F.cross_entropy(logits, yt, reduction="none")
        loss = (loss * w).mean() if w is not None else loss.mean()
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(xe).argmax(-1).numpy()


def _apply(corr, model, x_tr_raw, y_tr, x_ev_raw, seed):
    """Return per-story correctness vector via correction `corr` on eval era x_ev."""
    sc = StandardScaler().fit(x_tr_raw)
    Xtr = sc.transform(x_tr_raw)
    if corr == "base":
        Xev = sc.transform(x_ev_raw); w = None
    elif corr == "test_t":  # per-era standardisation of the eval era
        Xev = StandardScaler().fit_transform(x_ev_raw); w = None
    elif corr == "quant":   # per-era quantile transform to N(0,1)
        n = len(x_ev_raw); qt = QuantileTransformer(n_quantiles=max(10, min(100, n)), output_distribution="normal", random_state=seed)
        Xev = qt.fit_transform(x_ev_raw); w = None
    elif corr == "impwt":   # covariate-shift importance weighting (logistic domain clf)
        Xev = sc.transform(x_ev_raw)
        Z = np.vstack([Xtr, Xev]); d = np.r_[np.zeros(len(Xtr)), np.ones(len(Xev))]
        clf = LogisticRegression(max_iter=500).fit(Z, d)
        pt = clf.predict_proba(Xtr)[:, 1]; w = np.clip(pt / (1 - pt + 1e-6), 0.1, 10.0)
    elif corr == "coral":   # second-order alignment of train -> eval
        Cs = np.cov(Xtr, rowvar=False) + np.eye(Xtr.shape[1])
        Xev = sc.transform(x_ev_raw); Ct = np.cov(Xev, rowvar=False) + np.eye(Xev.shape[1])
        A = _msqrt(Cs, -0.5) @ _msqrt(Ct, 0.5); Xtr = Xtr @ A; w = None
    preds = _fit_predict(model, Xtr, y_tr, Xev, seed, w)
    return preds


def _boot_delta(corr_a, corr_b, B=10000, seed=0):
    """story bootstrap of mean(corr_a - corr_b); same test stories, aligned."""
    d = np.asarray(corr_a, float) - np.asarray(corr_b, float); n = len(d)
    rng = np.random.default_rng(seed); samp = d[rng.integers(0, n, size=(B, n))].mean(1)
    return float(d.mean()), [float(np.quantile(samp, .025)), float(np.quantile(samp, .975))]


def run():
    out = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw); out[name] = {}
        for model in ["propmlp", "propmlp_rich", "rf"]:
            rich = model == "propmlp_rich"
            # accumulate per-(story,seed) correctness on TEST era for each corr,
            # and seed-level F1 on VAL era for the selection rule
            test_corr = {c: [] for c in CORR}; test_f1 = {c: [] for c in CORR}
            val_f1 = {c: [] for c in CORR}
            for s in SEEDS:
                _set_seed(s); casc = _load_cascades(dcfg, s)
                tr, va, te = _temporal_split(casc, (0.8, 0.1, 0.1), 0)
                xtr = _features(tr, rich=rich); xva = _features(va, rich=rich); xte = _features(te, rich=rich)
                ytr = np.array([c.label for c in tr]); yva = np.array([c.label for c in va]); yte = np.array([c.label for c in te])
                for c in CORR:
                    pv = _apply(c, "rf" if model == "rf" else "mlp", xtr, ytr, xva, s)
                    val_f1[c].append(macro_f1(pv, yva))
                    pt = _apply(c, "rf" if model == "rf" else "mlp", xtr, ytr, xte, s)
                    test_f1[c].append(macro_f1(pt, yte))
                    test_corr[c].extend((pt == yte).astype(float))
            n_test = len(test_corr["base"]) // len(SEEDS)
            # delta CIs vs base
            deltas = {}
            for c in CORR:
                if c == "base":
                    continue
                obs, ci = _boot_delta(test_corr[c], test_corr["base"])
                deltas[c] = dict(test_f1=round(float(np.mean(test_f1[c])), 3), delta=round(obs, 4),
                                 ci95=[round(v, 3) for v in ci], excludes0=bool(ci[0] > 0 or ci[1] < 0))
            # selection rule: pick corr maximizing mean VAL F1 (label-free at test)
            val_means = {c: float(np.mean(val_f1[c])) for c in CORR}
            sel = max(val_means, key=val_means.get)
            oracle = max(CORR, key=lambda c: float(np.mean(test_f1[c])))
            out[name][model] = dict(
                n_test=n_test, base_f1=round(float(np.mean(test_f1["base"])), 3),
                selected=sel, selected_f1=round(float(np.mean(test_f1[sel])), 3),
                oracle=oracle, oracle_f1=round(float(np.mean(test_f1[oracle])), 3),
                val_means={c: round(v, 3) for c, v in val_means.items()},
                deltas=deltas)
            d = out[name][model]
            print(f"[{name}/{model}] n={n_test} base={d['base_f1']} sel={sel}->{d['selected_f1']} "
                  f"oracle={oracle}->{d['oracle_f1']} | any-delta-excl0={any(v['excludes0'] for v in deltas.values())}", flush=True)
        json.dump(out, open(f"{OUT}/mitig_delta.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
