"""Regenerate the four cheap main-table rows with TRAIN-ONLY fits and dump
val + test per-story predictions and scores.

  propmlp     PropMLP (7d)        - identical recipe to rich_summary_eval.py
  propmlp15   PropMLP+ (15d)      - identical recipe (rich=True)
  rf          RandomForest (7d)   - identical recipe to tabular_baselines.py
  histgbt15   HistGBT (15d)       - MODIFIED: hyperparameters selected by 5-fold CV
                                    *inside train* instead of on val, then fit on
                                    train only. The published protocol selects on
                                    val and refits on train+val, which would make
                                    its val predictions in-sample and unusable for
                                    honest selection.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASETS, SEEDS, folds, check_alignment, dump, ROOT
from docless.baselines.propmlp import _features, _MLP
from docless.eval.stats import macro_f1
from docless.train import _set_seed

GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 31) for l2 in (0.0, 1.0)]


def _propmlp(tr, va, te, rich, seed):
    """Mirrors train_propmlp(seed, dcfg, hidden_dim=64, epochs=25, rich=rich)
    exactly (same RNG consumption order), plus a val forward pass."""
    x_tr = _features(tr, rich=rich)
    x_te = _features(te, rich=rich)
    x_va = _features(va, rich=rich)
    mu = x_tr.mean(0, keepdims=True); sd = x_tr.std(0, keepdims=True) + 1e-6
    f = lambda x: torch.tensor((x - mu) / sd, dtype=torch.float32)
    xt, xe, xv = f(x_tr), f(x_te), f(x_va)
    y_tr = torch.tensor([c.label for c in tr], dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(xt), y_tr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        lv, lt = m(xv), m(xe)
    return (lv.argmax(-1).numpy(), F.softmax(lv, -1)[:, 1].numpy(),
            lt.argmax(-1).numpy(), F.softmax(lt, -1)[:, 1].numpy())


def _cv_select_histgbt(x, y, seed):
    """Hyperparameter selection by 5-fold CV strictly inside the TRAIN fold."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(x, y))
    best, best_f1 = None, -1.0
    scores = []
    for hp in GRID:
        fs = []
        for itr, iva in splits:
            m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp)
            m.fit(x[itr], y[itr])
            fs.append(macro_f1(m.predict(x[iva]), y[iva]))
        f1 = float(np.mean(fs)); scores.append(f1)
        if f1 > best_f1:
            best_f1, best = f1, hp
    return best, best_f1, scores


def run(name):
    out = {}
    for s in SEEDS:
        t0 = time.time()
        dcfg, tr, va, te = folds(name, s)
        check_alignment(name, s, va, te)
        y_tr = np.asarray([c.label for c in tr])
        y_va = np.asarray([c.label for c in va])
        y_te = np.asarray([c.label for c in te])

        # ---- PropMLP (7d) and PropMLP+ (15d) ----
        for tag, rich in (("propmlp", False), ("propmlp15", True)):
            _set_seed(s)
            _ = __import__("docless.train", fromlist=["_load_cascades"])  # noop, keep order
            vp, vs, tp, ts = _propmlp(tr, va, te, rich, s)
            dump(tag, name, s, va, te, vp, vs, tp, ts)
            out.setdefault(tag, []).append((macro_f1(vp, y_va), macro_f1(tp, y_te)))

        # ---- RandomForest (7d) ----
        xtr = _features(tr); xva = _features(va); xte = _features(te)
        sc = StandardScaler().fit(xtr)
        rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=1)
        rf.fit(sc.transform(xtr), y_tr)
        vp = rf.predict(sc.transform(xva)); vs = rf.predict_proba(sc.transform(xva))[:, 1]
        tp = rf.predict(sc.transform(xte)); ts = rf.predict_proba(sc.transform(xte))[:, 1]
        dump("rf", name, s, va, te, vp, vs, tp, ts)
        out.setdefault("rf", []).append((macro_f1(vp, y_va), macro_f1(tp, y_te)))

        # ---- HistGBT (15d), nested-in-train hyperparameter selection ----
        xtr = _features(tr, rich=True); xva = _features(va, rich=True); xte = _features(te, rich=True)
        sc = StandardScaler().fit(xtr)
        Xtr, Xva, Xte = sc.transform(xtr), sc.transform(xva), sc.transform(xte)
        hp, cvf1, _ = _cv_select_histgbt(Xtr, y_tr, s)
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp)
        m.fit(Xtr, y_tr)
        vp = m.predict(Xva); vs = m.predict_proba(Xva)[:, 1]
        tp = m.predict(Xte); ts = m.predict_proba(Xte)[:, 1]
        dump("histgbt15", name, s, va, te, vp, vs, tp, ts,
             extra=dict(inner_cv_hp=hp, inner_cv_f1=round(cvf1, 4)))
        out.setdefault("histgbt15", []).append((macro_f1(vp, y_va), macro_f1(tp, y_te)))

        print(f"[{name} seed {s}] " + "  ".join(
            f"{k}=val{v[-1][0]:.3f}/test{v[-1][1]:.3f}" for k, v in out.items())
            + f"  ({time.time()-t0:.1f}s)", flush=True)
    return out


if __name__ == "__main__":
    names = sys.argv[1:] or list(DATASETS)
    summ = {}
    for n in names:
        r = run(n)
        summ[n] = {k: dict(val=[round(a, 4) for a, _ in v], test=[round(b, 4) for _, b in v],
                           val_mean=round(float(np.mean([a for a, _ in v])), 4),
                           test_mean=round(float(np.mean([b for _, b in v])), 4))
                   for k, v in r.items()}
    p = f"{ROOT}/cheap_summary.json"
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(summ); json.dump(old, open(p, "w"), indent=2)
    print(json.dumps(summ, indent=2))
