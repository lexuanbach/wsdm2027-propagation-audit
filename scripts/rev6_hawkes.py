"""Point-process / growth-curve cascade features as a stronger non-neural frontier
(WSDM checklist: "Hawkes or point-process parameters ... report whether graph
gains remain after all classic structure features are available").

Identity-free releases expose event timestamps but no tree structure, so the
classic *shape* features (depth/width/virality) are only available in the
graph-informative regime (already in rev6_graph_informative.py). Here we add,
per cascade, a univariate exponential-kernel Hawkes fit (MLE; baseline rate mu,
branching ratio alpha, decay beta) and three growth-curve descriptors (time to
peak hourly rate, peak-hour share, log slope of log-count vs log-time over the
first 24h) to the 15-d summary, and compare HistGBT / RandomForest / PropMLP+
on 15-d vs 15-d+6 under the identical stratified splits and seeds (paired
per-story bootstrap on the tree).

Output: outputs/r11/hawkes/hawkes.json
Usage: PYTHONPATH=src python3 scripts/rev6_hawkes.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from docless.baselines.propmlp import _MLP, _features  # noqa: E402
from docless.eval.stats import macro_f1, paired_bootstrap  # noqa: E402
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r11" / "hawkes"
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid": dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "recovery": dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop": dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}
GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 31) for l2 in (0.0, 1.0)]


def hawkes_mle(t: np.ndarray) -> tuple[float, float, float]:
    """Exponential-kernel Hawkes: lambda(t) = mu + sum_{t_j<t} alpha*beta*exp(-beta (t-t_j)).
    Times in hours, shifted to start at 0; T = last event time (+1e-3). Returns (mu, alpha, beta)."""
    t = np.sort(np.asarray(t, dtype=float)); t = t - t[0]
    n = len(t); T = t[-1] + 1e-3
    if n < 3 or T <= 1e-3:
        return (n / max(T, 1e-3), 0.0, 1.0)
    dt = np.diff(t)

    def nll(theta):
        mu, alpha, beta = np.exp(theta)
        R = np.zeros(n)
        for i in range(1, n):
            R[i] = np.exp(-beta * dt[i - 1]) * (1.0 + R[i - 1])
        lam = mu + alpha * beta * R
        ll = np.sum(np.log(lam + 1e-12)) - mu * T - alpha * np.sum(1.0 - np.exp(-beta * (T - t)))
        return -ll

    best = None
    for a0 in (0.2, 0.8):
        for b0 in (0.5, 5.0):
            x0 = np.log([max(n / T * 0.5, 1e-3), a0, b0])
            r = minimize(nll, x0, method="L-BFGS-B", bounds=[(-12, 6), (-8, np.log(5.0)), (-6, 6)])
            if best is None or r.fun < best.fun:
                best = r
    mu, alpha, beta = np.exp(best.x)
    return (float(mu), float(alpha), float(beta))


def growth_features(t: np.ndarray) -> list[float]:
    t = np.sort(np.asarray(t, dtype=float)); t = (t - t[0]) / 3600.0  # hours
    n = len(t)
    if n < 2:
        return [0.0, 1.0, 0.0]
    hours = np.floor(t).astype(int)
    counts = np.bincount(hours)
    peak_h = int(np.argmax(counts)); peak_share = float(counts.max() / n)
    m = t <= 24.0
    tt = t[m]; k = np.arange(1, len(tt) + 1)
    if len(tt) >= 3:
        x = np.log1p(tt); y = np.log(k)
        slope = float(np.polyfit(x, y, 1)[0]) if np.ptp(x) > 1e-6 else 0.0
    else:
        slope = 0.0
    return [np.log1p(peak_h), peak_share, slope]


def extra_features(cascades) -> np.ndarray:
    rows = []
    for c in cascades:
        ts = np.asarray([e.timestamp for e in c.events], dtype=float)
        th = np.sort(ts) / 3600.0
        mu, alpha, beta = hawkes_mle(th)
        rows.append([np.log1p(mu), alpha, np.log(beta), *growth_features(ts)])
    return np.asarray(rows, dtype=np.float64)


def tuned_histgbt(x_tr, y_tr, x_va, y_va):
    best, best_f1 = None, -1.0
    for hp in GRID:
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp).fit(x_tr, y_tr)
        f1 = macro_f1(m.predict(x_va), y_va)
        if f1 > best_f1:
            best_f1, best = f1, hp
    return HistGradientBoostingClassifier(max_iter=300, random_state=0, **best).fit(
        np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        res = {k: [] for k in ["histgbt15", "histgbt21", "rf15", "rf21", "propmlp15", "propmlp21"]}
        losses = {k: [] for k in res}
        for s in SEEDS:
            _set_seed(s)
            casc = _load_cascades(dcfg, s)
            tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
            y_tr = np.array([c.label for c in tr]); y_va = np.array([c.label for c in va]); y_te = np.array([c.label for c in te])
            f15 = {k: _features(v, rich=True) for k, v in (("tr", tr), ("va", va), ("te", te))}
            fx = {k: extra_features(v) for k, v in (("tr", tr), ("va", va), ("te", te))}
            f21 = {k: np.hstack([f15[k], fx[k]]) for k in f15}
            preds = {}
            for tag, feats in (("15", f15), ("21", f21)):
                sc = StandardScaler().fit(feats["tr"])
                a, b, c = sc.transform(feats["tr"]), sc.transform(feats["va"]), sc.transform(feats["te"])
                preds[f"histgbt{tag}"] = tuned_histgbt(a, y_tr, b, y_va).predict(c)
                preds[f"rf{tag}"] = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(a, y_tr).predict(c)
                preds[f"propmlp{tag}"] = train_mlp(a, y_tr, c, s)
            for k, p in preds.items():
                res[k].append(macro_f1(p, y_te)); losses[k].append((p != y_te).astype(float))
            print(f"[{name}] seed {s}: " + " ".join(f"{k}={res[k][-1]:.3f}" for k in res), flush=True)
        pooled = {k: np.concatenate(v) for k, v in losses.items()}
        out = {"results": {k: {"f1_mean": round(float(np.mean(v)), 4), "f1_std": round(float(np.std(v)), 4)} for k, v in res.items()}}
        for m in ("histgbt", "rf", "propmlp"):
            pb = paired_bootstrap(pooled[f"{m}15"], pooled[f"{m}21"])  # acc gain of 21 over 15 = -(mean_diff)
            out[f"paired_{m}_21_minus_15"] = {"acc_gain": round(-pb.mean_diff, 4), "p": round(pb.p_value, 4),
                                              "ci95": [round(-pb.ci_hi, 4), round(-pb.ci_lo, 4)]}
        out["extra_features"] = ["log1p(mu)", "alpha (branching ratio)", "log(beta)", "log1p(hour of peak rate)",
                                 "peak-hour share", "log-log growth slope (first 24h)"]
        result[name] = out
        print(f"[{name}] " + json.dumps({k: v["f1_mean"] for k, v in out["results"].items()}), flush=True)
        (OUT / "hawkes.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
