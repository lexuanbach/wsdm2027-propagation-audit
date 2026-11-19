"""Temporal (deployment-realistic) split audit.

We order each powered dataset's stories by creation time and
train on the earliest 80%, validate on the next 10%, and test on the latest
10% (test strictly in the future of training). We compare the validated simple
baselines (PropMLP, PropMLP+, RandomForest) and DocLessFake under this temporal
split against their random-split numbers.

Cheap tabular/MLP models are run on both powered datasets (Politifact,
GossipCop). DocLessFake (expensive) is run on Politifact only, to confirm the
"architecture does not beat the strong baseline" conclusion survives a temporal
split. The temporal split is deterministic, so model-randomness is averaged over
5 seeds for the learned models (RF uses its own per-seed random_state).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _features, train_propmlp
from docless.eval.stats import macro_f1
from docless.train import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    _load_cascades,
    _set_seed,
    _temporal_split,
    train_one_seed,
)

OUT = "outputs/r9/temporal_split"
os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
dev = torch.device("cpu")

DATASETS = {
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop": dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet",
                      max_per_class=1500),
}


def _rf_temporal(dcfg, seed, rich=False):
    _set_seed(seed)
    casc = _load_cascades(dcfg, seed)
    tr, _va, te = _temporal_split(casc, dcfg.split_ratios, seed)
    x_tr = _features(tr, rich=rich); x_te = _features(te, rich=rich)
    y_tr = np.asarray([c.label for c in tr]); y_te = np.asarray([c.label for c in te])
    sc = StandardScaler().fit(x_tr)
    m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1)
    m.fit(sc.transform(x_tr), y_tr)
    preds = m.predict(sc.transform(x_te))
    return float(macro_f1(preds, y_te))


def _propmlp_temporal(dcfg, seed, rich=False):
    # train_propmlp splits internally; replicate with temporal split here.
    _set_seed(seed)
    casc = _load_cascades(dcfg, seed)
    tr, _va, te = _temporal_split(casc, dcfg.split_ratios, seed)
    import torch.nn.functional as F
    from docless.baselines.propmlp import _MLP
    x_tr = _features(tr, rich=rich); x_te = _features(te, rich=rich)
    mu = x_tr.mean(0, keepdims=True); sd = x_tr.std(0, keepdims=True) + 1e-6
    x_tr = (x_tr - mu) / sd; x_te = (x_te - mu) / sd
    xt = torch.tensor(x_tr, dtype=torch.float32); xe = torch.tensor(x_te, dtype=torch.float32)
    y_tr = torch.tensor([c.label for c in tr], dtype=torch.long)
    y_te = torch.tensor([c.label for c in te], dtype=torch.long)
    model = _MLP(in_dim=xt.shape[1], hidden=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(25):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(xt), y_tr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(xe).argmax(-1).numpy()
    return float(macro_f1(preds, y_te.numpy()))


def run(name, dcfg_kw, with_docless):
    dcfg = DataConfig(**dcfg_kw)
    res = {}
    for tag, fn in [("propmlp", lambda s: _propmlp_temporal(dcfg, s, rich=False)),
                    ("propmlp_plus", lambda s: _propmlp_temporal(dcfg, s, rich=True)),
                    ("rf", lambda s: _rf_temporal(dcfg, s, rich=False))]:
        vals = [fn(s) for s in SEEDS]
        res[tag] = dict(f1_mean=round(float(np.mean(vals)), 3),
                        f1_std=round(float(np.std(vals)), 3),
                        per_seed=[round(v, 4) for v in vals])
        print(f"[{name}] temporal {tag}: {res[tag]['f1_mean']}", flush=True)
    if with_docless:
        vals = []
        for s in SEEDS:
            r = train_one_seed(s, dcfg, ModelConfig(hidden_dim=64, rich_summary=False),
                               TrainConfig(pretrain_epochs=4, finetune_epochs=25),
                               verbose=False, device=dev, split_fn=_temporal_split)
            vals.append(r["f1_all"])
            print(f"[{name}] temporal docless seed {s}: {round(r['f1_all'],4)}", flush=True)
        res["docless"] = dict(f1_mean=round(float(np.mean(vals)), 3),
                              f1_std=round(float(np.std(vals)), 3),
                              per_seed=[round(v, 4) for v in vals])
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=2)
    print(f"== {name} done: {json.dumps(res)}", flush=True)
    return res


if __name__ == "__main__":
    run("politifact", DATASETS["politifact"], with_docless=True)
    run("gossipcop", DATASETS["gossipcop"], with_docless=False)
    print("ALL DONE", flush=True)
