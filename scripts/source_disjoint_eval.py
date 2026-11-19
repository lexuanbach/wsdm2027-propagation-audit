"""Source(publisher)-disjoint split audit. We map each FakeNewsNet story to its publisher domain (from the article
URL), then hold out a label-balanced set of *unseen* publishers for test, so the
model is evaluated on outlets it never saw in training -- a hard, deployment-
realistic generalisation test for propagation features. We run the validated
simple models (PropMLP, PropMLP+, RandomForest) under this split on Politifact
and GossipCop and compare to their random-split numbers.
"""
from __future__ import annotations

import csv, json, os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _features, _MLP
from docless.data.loaders.coaid import _stable_url_id
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed

OUT = "outputs/r9/source_disjoint"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
SOURCES = {"politifact": "data/raw/fakenewsnet", "gossipcop": "data/raw/fakenewsnet"}


def _domain(url: str) -> str:
    u = url.strip().lower()
    if "://" in u:
        u = u.split("://", 1)[1]
    u = u.split("/", 1)[0]
    if u.startswith("www."):
        u = u[4:]
    return u or "unknown"


def _storyid_to_domain(source, root, max_per_class=None):
    """Map story_id -> publisher domain by re-hashing the article URLs."""
    m = {}
    for kind in ("fake", "real"):
        path = os.path.join(root, f"{source}_{kind}.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                url = (row.get("news_url") or "").strip()
                if not url:
                    continue
                sid = f"{source}:{_stable_url_id(url):016x}"
                m[sid] = _domain(url)
    return m


def _disjoint_split(casc, dom_map, seed):
    """Hold out unseen publishers (~18% of stories), keeping both labels in test."""
    rng = np.random.default_rng(seed)
    # domains grouped by their (majority) label
    by_dom = {}
    for i, c in enumerate(casc):
        d = dom_map.get(c.story_id, "unknown")
        by_dom.setdefault(d, []).append(i)
    doms = list(by_dom)
    # label of a domain = majority label of its stories
    dom_label = {d: int(round(np.mean([casc[i].label for i in idx]))) for d, idx in by_dom.items()}
    test_idx = []
    for lab in (0, 1):
        cand = [d for d in doms if dom_label[d] == lab]
        rng.shuffle(cand)
        target = int(0.18 * sum(len(by_dom[d]) for d in cand))
        acc = 0
        for d in cand:
            if acc >= target:
                break
            test_idx.extend(by_dom[d]); acc += len(by_dom[d])
    test_set = set(test_idx)
    tr = [casc[i] for i in range(len(casc)) if i not in test_set]
    te = [casc[i] for i in test_idx]
    return tr, te


def _rf(tr, te, rich=False, seed=0):
    xtr = _features(tr, rich=rich); xte = _features(te, rich=rich)
    ytr = np.array([c.label for c in tr]); yte = np.array([c.label for c in te])
    sc = StandardScaler().fit(xtr)
    m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1)
    m.fit(sc.transform(xtr), ytr)
    return float(macro_f1(m.predict(sc.transform(xte)), yte))


def _mlp(tr, te, rich=False, seed=0):
    _set_seed(seed)
    xtr = _features(tr, rich=rich); xte = _features(te, rich=rich)
    mu = xtr.mean(0, keepdims=True); sd = xtr.std(0, keepdims=True) + 1e-6
    xtr = (xtr - mu) / sd; xte = (xte - mu) / sd
    xt = torch.tensor(xtr, dtype=torch.float32); xe = torch.tensor(xte, dtype=torch.float32)
    ytr = torch.tensor([c.label for c in tr], dtype=torch.long)
    yte = np.array([c.label for c in te])
    model = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(25):
        model.train(); opt.zero_grad(); F.cross_entropy(model(xt), ytr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(xe).argmax(-1).numpy()
    return float(macro_f1(preds, yte))


def run(name, root):
    dc = DataConfig(source=name, balanced=True, data_root=root,
                    **({"max_per_class": 1500} if name == "gossipcop" else {}))
    res = {"propmlp": [], "propmlp_plus": [], "rf": [], "n_test": [], "n_test_dom": []}
    for s in SEEDS:
        _set_seed(s)
        casc = _load_cascades(dc, s)
        dom_map = _storyid_to_domain(name, root)
        tr, te = _disjoint_split(casc, dom_map, s)
        res["propmlp"].append(_mlp(tr, te, rich=False, seed=s))
        res["propmlp_plus"].append(_mlp(tr, te, rich=True, seed=s))
        res["rf"].append(_rf(tr, te, rich=False, seed=s))
        res["n_test"].append(len(te))
        res["n_test_dom"].append(len({dom_map.get(c.story_id) for c in te}))
    agg = {k: (round(float(np.mean(v)), 3) if k in ("propmlp", "propmlp_plus", "rf") else int(np.mean(v)))
           for k, v in res.items()}
    agg["per_seed"] = {k: [round(x, 3) for x in res[k]] for k in ("propmlp", "propmlp_plus", "rf")}
    json.dump(agg, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"[{name}] {agg}", flush=True)
    return agg


if __name__ == "__main__":
    for name, root in SOURCES.items():
        run(name, root)
    print("ALL DONE", flush=True)
