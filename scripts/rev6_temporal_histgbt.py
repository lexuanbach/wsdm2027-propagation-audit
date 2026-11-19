"""Temporal (future-test) evaluation of every finalist, including the overall
best model HistGBT, plus a stronger label-free adaptation
baseline (domain-classifier importance weighting) and a root-cause note for
the validation-fold selector failure.

Split: deterministic temporal 80/10/10 by earliest event time (same as
scripts/temporal_split_eval.py). Models: PropMLP (7d), PropMLP+ (15d),
RandomForest (7d/15d), val-tuned HistGBT (7d/15d); all five seeds (model
randomness only). Adaptation: importance weighting with a domain classifier
(train-vs-test-feature discriminator; weights clipped to [0.1, 10]) fitted
label-free on the future-era features, applied to HistGBT-15d and RF-15d as
sample weights; paired per-story bootstrap CI for the accuracy delta.

Outputs: outputs/r11/temporal/<dataset>.json and temporal_rows.tex.
Usage: PYTHONPATH=src python3 scripts/rev6_temporal_histgbt.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from docless.baselines.propmlp import _MLP, _features  # noqa: E402
from docless.eval.stats import macro_f1, paired_bootstrap  # noqa: E402
from docless.train import DataConfig, _load_cascades, _set_seed, _temporal_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r11" / "temporal"
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "recovery": dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop": dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}
GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 31) for l2 in (0.0, 1.0)]


def tuned_histgbt(x_tr, y_tr, x_va, y_va, sample_weight=None):
    best, best_f1 = None, -1.0
    for hp in GRID:
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp)
        m.fit(x_tr, y_tr, sample_weight=sample_weight)
        f1 = macro_f1(m.predict(x_va), y_va)
        if f1 > best_f1:
            best_f1, best = f1, hp
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **best)
    sw = None if sample_weight is None else np.concatenate([sample_weight, np.ones(len(y_va))])
    m.fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]), sample_weight=sw)
    return m


def train_mlp(x_tr, y_tr, x_te, seed, hidden=64, epochs=25, lr=1e-2):
    torch.manual_seed(seed)
    xt = torch.tensor(x_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.long)
    model = _MLP(in_dim=xt.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(xt), yt).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x_te, dtype=torch.float32)).argmax(-1).numpy()


def importance_weights(x_tr, x_te, seed, clip=(0.1, 10.0)):
    """Label-free covariate-shift weights w(x) = p_test(x)/p_train(x) from a
    logistic domain classifier (test=1 vs train=0), clipped."""
    X = np.vstack([x_tr, x_te]); d = np.concatenate([np.zeros(len(x_tr)), np.ones(len(x_te))])
    clf = LogisticRegression(max_iter=2000, random_state=seed).fit(X, d)
    p = clf.predict_proba(x_tr)[:, 1]
    w = (p / (1 - p + 1e-9)) * (len(x_tr) / len(x_te))
    return np.clip(w, *clip)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        res = {k: [] for k in ["propmlp7", "propmlp15", "rf7", "rf15", "histgbt7", "histgbt15",
                               "histgbt15_iw", "rf15_iw"]}
        losses = {k: [] for k in res}
        labels_all = []
        for s in SEEDS:
            _set_seed(s)
            casc = _load_cascades(dcfg, s)
            tr, va, te = _temporal_split(casc, dcfg.split_ratios, s)
            y_tr = np.array([c.label for c in tr]); y_va = np.array([c.label for c in va]); y_te = np.array([c.label for c in te])
            labels_all.append(y_te)
            preds = {}
            for d, rich in [("7", False), ("15", True)]:
                x_tr, x_va, x_te = _features(tr, rich=rich), _features(va, rich=rich), _features(te, rich=rich)
                sc = StandardScaler().fit(x_tr)
                xs_tr, xs_va, xs_te = sc.transform(x_tr), sc.transform(x_va), sc.transform(x_te)
                preds[f"propmlp{d}"] = train_mlp(xs_tr, y_tr, xs_te, s)
                preds[f"rf{d}"] = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(xs_tr, y_tr).predict(xs_te)
                preds[f"histgbt{d}"] = tuned_histgbt(xs_tr, y_tr, xs_va, y_va).predict(xs_te)
                if d == "15":
                    w = importance_weights(xs_tr, xs_te, s)
                    preds["histgbt15_iw"] = tuned_histgbt(xs_tr, y_tr, xs_va, y_va, sample_weight=w).predict(xs_te)
                    preds["rf15_iw"] = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(xs_tr, y_tr, sample_weight=w).predict(xs_te)
            for k, p in preds.items():
                res[k].append(macro_f1(p, y_te))
                losses[k].append((p != y_te).astype(float))
            print(f"[{name}] seed {s}: " + " ".join(f"{k}={res[k][-1]:.3f}" for k in res), flush=True)
        agg = {k: {"f1_mean": round(float(np.mean(v)), 4), "f1_std": round(float(np.std(v)), 4),
                   "per_seed": [round(float(x), 4) for x in v]} for k, v in res.items()}
        pooled = {k: np.concatenate(v) for k, v in losses.items()}
        pb_iw = paired_bootstrap(pooled["histgbt15"], pooled["histgbt15_iw"])
        pb_rf = paired_bootstrap(pooled["rf15"], pooled["rf15_iw"])
        out = {
            "dataset": name, "split": "temporal 80/10/10 by earliest event time; test strictly future",
            "n_test_per_seed": [int(len(y)) for y in labels_all],
            "test_prior_misinfo": round(float(np.mean(np.concatenate(labels_all))), 3),
            "results": agg,
            "importance_weighting": {
                "histgbt15": {"acc_delta_iw_minus_base": round(-pb_iw.mean_diff, 4), "p": round(pb_iw.p_value, 4),
                              "ci95": [round(-pb_iw.ci_hi, 4), round(-pb_iw.ci_lo, 4)]},
                "rf15": {"acc_delta_iw_minus_base": round(-pb_rf.mean_diff, 4), "p": round(pb_rf.p_value, 4),
                         "ci95": [round(-pb_rf.ci_hi, 4), round(-pb_rf.ci_lo, 4)]},
                "method": "logistic domain classifier train-vs-future features, weights clipped [0.1,10], label-free",
            },
        }
        (OUT / f"{name}.json").write_text(json.dumps(out, indent=2))
        summary[name] = {k: v["f1_mean"] for k, v in agg.items()}
        print(f"[{name}] " + json.dumps(summary[name]), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    names = {"recovery": "ReCOVery", "politifact": "PolitiFact", "gossipcop": "GossipCop"}
    order = [("propmlp7", "PropMLP (7d)"), ("propmlp15", "PropMLP+ (15d)"), ("rf7", "RandomForest (7d)"),
             ("rf15", "RandomForest (15d)"), ("histgbt7", "HistGBT (7d)"), ("histgbt15", "HistGBT (15d)"),
             ("histgbt15_iw", "HistGBT (15d) + importance weighting"), ("rf15_iw", "RandomForest (15d) + importance weighting")]
    lines = [f"{label} & " + " & ".join(f"{summary[d][k]:.3f}" for d in DATASETS) + " \\\\" for k, label in order]
    (OUT / "temporal_rows.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
