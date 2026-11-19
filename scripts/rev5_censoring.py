"""Fixed observation-window control for the temporal drop.

Concern: newer (future-test) cascades may be observed over a SHORTER
wall-clock horizon at scrape time, so 'temporal spread' and other count features
shift mechanically -- conflating real drift with right-censoring. Control: cap
EVERY cascade (train and test) to the events in a FIXED window [t0, t0+W] from
its own first event, recompute all features on that window, then run the temporal
future-test split. If the ~0.2 macro-F1 drop persists under a fixed horizon (so
old and new stories are observed for the same duration), the drop is genuine
drift, not censoring. PropMLP + RandomForest (fast), 5 seeds, PolitiFact+GossipCop.
"""
from __future__ import annotations
import json, os
from dataclasses import replace
import numpy as np
import torch, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split, _temporal_split

OUT = os.environ.get("CENSORING_OUT", "outputs/r10/censoring"); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DAY = 86400.0
HOUR = 3600.0
# Default reproduces the published 1/2/7-day sweep.  CENSORING_FINE=1 adds the
# finer early-detection horizons (1 h, 6 h, 3 d) ; the
# shared windows are recomputed identically so the two runs can be checked
# against each other.
WINDOWS = {"uncapped": None, "7d": 7 * DAY, "2d": 2 * DAY, "1d": 1 * DAY}
if os.environ.get("CENSORING_FINE") == "1":
    WINDOWS = {"uncapped": None, "7d": 7 * DAY, "3d": 3 * DAY, "1d": 1 * DAY,
               "6h": 6 * HOUR, "1h": 1 * HOUR}
DATASETS = {
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}


def _window(cascades, W):
    if W is None:
        return cascades
    out = []
    for c in cascades:
        if not c.events:
            out.append(c); continue
        t0 = min(e.timestamp for e in c.events)
        ev = [e for e in c.events if e.timestamp <= t0 + W]
        out.append(replace(c, events=ev))
    return out


def _propmlp_f1(tr, te, seed):
    _set_seed(seed)
    xtr, xte = _features(tr), _features(te)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((xtr - mu) / sd, dtype=torch.float32); xe = torch.tensor((xte - mu) / sd, dtype=torch.float32)
    yt = torch.tensor([c.label for c in tr], dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return macro_f1(m(xe).argmax(-1).numpy(), np.array([c.label for c in te]))


def _rf_f1(tr, te, seed):
    xtr, xte = _features(tr), _features(te)
    sc = StandardScaler().fit(xtr)
    m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1)
    m.fit(sc.transform(xtr), np.array([c.label for c in tr]))
    return macro_f1(m.predict(sc.transform(xte)), np.array([c.label for c in te]))


def run():
    out = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw); out[name] = {}
        for wname, W in WINDOWS.items():
            cell = {"propmlp_random": [], "propmlp_temporal": [], "rf_random": [], "rf_temporal": []}
            for s in SEEDS:
                _set_seed(s); casc = _window(_load_cascades(dcfg, s), W)
                rtr, _rv, rte = _stratified_split(casc, (0.8, 0.1, 0.1), s)
                ttr, _tv, tte = _temporal_split(casc, (0.8, 0.1, 0.1), 0)
                cell["propmlp_random"].append(_propmlp_f1(rtr, rte, s))
                cell["propmlp_temporal"].append(_propmlp_f1(ttr, tte, s))
                cell["rf_random"].append(_rf_f1(rtr, rte, s))
                cell["rf_temporal"].append(_rf_f1(ttr, tte, s))
            agg = {k: round(float(np.mean(v)), 3) for k, v in cell.items()}
            agg["propmlp_drop"] = round(agg["propmlp_random"] - agg["propmlp_temporal"], 3)
            agg["rf_drop"] = round(agg["rf_random"] - agg["rf_temporal"], 3)
            agg["n_test_temporal"] = len(tte)
            out[name][wname] = agg
            print(f"[{name}/{wname}] PropMLP rand={agg['propmlp_random']} temp={agg['propmlp_temporal']} drop={agg['propmlp_drop']} | "
                  f"RF rand={agg['rf_random']} temp={agg['rf_temporal']} drop={agg['rf_drop']}", flush=True)
            json.dump(out, open(f"{OUT}/censoring.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
