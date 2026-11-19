"""Leave-one-feature-out ablation: is PropMLP's signal just cascade size?

Motivation: the paper reports WHICH features dominate (RF
importances) but never removes them, so "the summary models are really just
counting retweets" is untested.

Two drop arms on the 7-d summary, identical splits/seeds/models:
  full            : all 7 features
  drop_size       : without log_n
  drop_size_users : without log_n AND log_users

The second arm matters because log_n (RF importance 0.186) and log_users
(0.176) are strongly correlated on GossipCop: dropping log_n alone can simply
be absorbed by log_users and prove nothing.

Per-story 0/1 losses pooled across 5 seeds; paired_bootstrap(full, reduced)
returns mean(loss_full - loss_reduced), so a POSITIVE delta means the reduced
model wins and a NEGATIVE delta means dropping the feature hurt.

Reports whatever comes out, including a null.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1, paired_bootstrap
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r11/feature_drop"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
FEATURE_NAMES = ["log_n", "log_users", "log_urls", "log_media", "log_pop", "echo", "log_spread"]
ARMS = {
    "full":            [0, 1, 2, 3, 4, 5, 6],
    "drop_size":       [1, 2, 3, 4, 5, 6],
    "drop_size_users": [2, 3, 4, 5, 6],
}
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}


def fit_propmlp(x_tr, y_tr, x_te, seed):
    """Same recipe as baselines.train_propmlp: hidden 64, 25 full-batch epochs,
    Adam 1e-2, on z-scored features -- only the input width changes."""
    _set_seed(seed)
    mu = x_tr.mean(0, keepdims=True)
    sd = x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32)
    xe = torch.tensor((x_te - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    model = _MLP(in_dim=xt.shape[1], hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(25):
        model.train()
        opt.zero_grad()
        F.cross_entropy(model(xt), yt).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return model(xe).argmax(-1).numpy()


def fit_rf(x_tr, y_tr, x_te):
    sc = StandardScaler().fit(x_tr)
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=1)
    rf.fit(sc.transform(x_tr), y_tr)
    return rf.predict(sc.transform(x_te))


def run_dataset(name, dcfg_kw):
    dcfg = DataConfig(**dcfg_kw)
    losses = {m: {a: [] for a in ARMS} for m in ("propmlp", "rf")}
    f1s = {m: {a: [] for a in ARMS} for m in ("propmlp", "rf")}
    n_test = 0
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(dcfg, s)
        tr, _va, te = _stratified_split(casc, dcfg.split_ratios, s)
        x_tr_full = _features(tr)
        x_te_full = _features(te)
        y_tr = np.asarray([c.label for c in tr])
        y_te = np.asarray([c.label for c in te])
        n_test += len(y_te)
        for arm, cols in ARMS.items():
            x_tr = x_tr_full[:, cols]
            x_te = x_te_full[:, cols]
            pm = fit_propmlp(x_tr, y_tr, x_te, s)
            rf = fit_rf(x_tr, y_tr, x_te)
            losses["propmlp"][arm].append((pm != y_te).astype(float))
            losses["rf"][arm].append((rf != y_te).astype(float))
            f1s["propmlp"][arm].append(round(float(macro_f1(pm, y_te)), 4))
            f1s["rf"][arm].append(round(float(macro_f1(rf, y_te)), 4))

    out = {"n_stories_pooled": int(n_test), "feature_names": FEATURE_NAMES}
    for m in ("propmlp", "rf"):
        entry = {}
        pooled = {a: np.concatenate(losses[m][a]) for a in ARMS}
        for a in ARMS:
            entry[a] = {
                "f1_mean": round(float(np.mean(f1s[m][a])), 3),
                "f1_std": round(float(np.std(f1s[m][a])), 3),
                "per_seed": f1s[m][a],
                "acc": round(float(1 - pooled[a].mean()), 4),
            }
        for a in ("drop_size", "drop_size_users"):
            r = paired_bootstrap(pooled["full"], pooled[a])
            # paired_bootstrap(a, b) returns mean(loss_b - loss_a) with a=full,
            # b=reduced. So mean_diff is a LOSS delta: POSITIVE means the reduced
            # model has higher loss, i.e. dropping the feature(s) HURT.
            entry[a]["paired_vs_full"] = {
                "mean_loss_delta_reduced_minus_full": round(r.mean_diff, 4),
                "acc_cost_of_dropping": round(r.mean_diff, 4),
                "direction": ("dropping hurts" if r.mean_diff > 0 else
                              "dropping helps" if r.mean_diff < 0 else "no change"),
                "p_value": round(r.p_value, 4),
                "ci_lo": round(r.ci_lo, 4),
                "ci_hi": round(r.ci_hi, 4),
            }
        out[m] = entry
    return out


TABLE2_FULL = {
    ("coaid", "propmlp"): 0.618, ("recovery", "propmlp"): 0.659,
    ("politifact", "propmlp"): 0.657, ("gossipcop", "propmlp"): 0.836,
    ("coaid", "rf"): 0.595, ("recovery", "rf"): 0.651,
    ("politifact", "rf"): 0.667, ("gossipcop", "rf"): 0.865,
}


if __name__ == "__main__":
    res = {}
    for name, dcfg in DATASETS.items():
        res[name] = run_dataset(name, dcfg)
        for m in ("propmlp", "rf"):
            e = res[name][m]
            print(f"[{name}/{m}] full={e['full']['f1_mean']:.3f} "
                  f"drop_size={e['drop_size']['f1_mean']:.3f} "
                  f"(cost={e['drop_size']['paired_vs_full']['acc_cost_of_dropping']:+.4f}, "
                  f"p={e['drop_size']['paired_vs_full']['p_value']:.3f})  "
                  f"drop_size_users={e['drop_size_users']['f1_mean']:.3f} "
                  f"(cost={e['drop_size_users']['paired_vs_full']['acc_cost_of_dropping']:+.4f}, "
                  f"p={e['drop_size_users']['paired_vs_full']['p_value']:.3f})", flush=True)
    # The full arm must reproduce the Table 2 cells, else we ablated the wrong thing.
    repro = {}
    bad = []
    for (ds, m), exp in TABLE2_FULL.items():
        got = res[ds][m]["full"]["f1_mean"]
        ok = abs(got - exp) <= 0.001
        repro[f"{ds}_{m}"] = {"computed": got, "table2": exp, "match": ok}
        if not ok:
            bad.append(f"{ds}/{m}: {got} vs {exp}")
    res["_full_arm_reproduces_table2"] = repro
    print("Table-2 reproduction of the full arm:", "all match" if not bad else bad)
    json.dump(res, open(f"{OUT}/feature_drop.json", "w"), indent=2)
    print("DONE", flush=True)
