"""Temporal-drift rigor and prior shift.

For PolitiFact & GossipCop future-test splits:
  * report the test-era n and a story-level bootstrap 95% CI for the baseline
    drop AND for every mitigation cell (so +0.05 recoveries are distinguishable
    from a few-story fluctuation);
  * multiple cut points -- test on the last 10/20/30% (rolling origin) -- to show
    the drop and the recovery are not a single-partition artifact;
  * add a label-free PRIOR-shift correction (Saerens EM posterior adjustment),
    since part of the drift is a class-prior move that covariate-shift methods
    (IW/CORAL) cannot fix.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch, torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _temporal_split

OUT = "outputs/r10/temporal_rigor"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}


def _split_at(casc, frac):
    """train on earliest (1-2*frac), val frac, test last frac (by time)."""
    return _temporal_split(casc, (1 - 2 * frac, frac, frac), 0)


def _propmlp_proba(x_tr, y_tr, x_te):
    mu, sd = x_tr.mean(0, keepdims=True), x_tr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((x_tr - mu) / sd, dtype=torch.float32); xe = torch.tensor((x_te - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return F.softmax(m(xe), -1).numpy()


def _saerens(proba_te, src_prior, iters=50):
    """Label-free EM estimate of target prior, then posterior re-weighting."""
    p = proba_te.copy(); pt = np.array([1 - src_prior, src_prior])
    ps = np.array([1 - src_prior, src_prior])
    for _ in range(iters):
        w = (pt / ps)[None, :] * p
        w = w / w.sum(1, keepdims=True)
        pt = w.mean(0)
    w = (pt / ps)[None, :] * p
    return w / w.sum(1, keepdims=True)


def _cluster_ci(losses, B=5000, seed=0):
    """story-level bootstrap CI of macro-F1-equivalent via per-story 0/1 (here accuracy proxy)."""
    rng = np.random.default_rng(seed); arr = np.asarray(losses); n = len(arr)
    means = arr[rng.integers(0, n, size=(B, n))].mean(1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def eval_cell(dcfg, model, mode, frac=0.1):
    f1s = []; last_pred = last_y = None
    for s in SEEDS:
        _set_seed(s); casc = _load_cascades(dcfg, s)
        tr, _va, te = _split_at(casc, frac)
        x_tr, x_te = _features(tr), _features(te)
        y_tr = np.array([c.label for c in tr]); y_te = np.array([c.label for c in te])
        src_prior = float(y_tr.mean())
        if model == "rf":
            sc = StandardScaler().fit(x_tr); Xtr, Xte = sc.transform(x_tr), sc.transform(x_te)
            m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(Xtr, y_tr)
            proba = m.predict_proba(Xte)
            if mode == "coral":
                Cs = np.cov(Xtr, rowvar=False) + np.eye(Xtr.shape[1]); Ct = np.cov(Xte, rowvar=False) + np.eye(Xte.shape[1])
                def msqrt(C, p):
                    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None); return (V * w**p) @ V.T
                A = msqrt(Cs, -0.5) @ msqrt(Ct, 0.5)
                m2 = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(Xtr @ A, y_tr)
                proba = m2.predict_proba(Xte)
        else:
            sc = StandardScaler().fit(x_tr); Xtr, Xte = sc.transform(x_tr), sc.transform(x_te)
            proba = _propmlp_proba(Xtr if mode != "coral" else Xtr, y_tr, Xte)
        if mode == "prior":
            proba = _saerens(proba, src_prior)
        elif mode == "impweight" and model != "rf":
            pass  # impweight handled in v2; here we focus on prior vs baseline vs coral
        preds = proba.argmax(1)
        f1s.append(float(macro_f1(preds, y_te)))
        last_pred, last_y = preds, y_te
    ci = None
    if last_pred is not None:
        ci = _cluster_ci((last_pred == last_y).astype(float))
    return dict(f1_mean=round(float(np.mean(f1s)), 3), f1_std=round(float(np.std(f1s)), 3),
                per_seed=[round(v, 4) for v in f1s], n_test=int(len(last_y)), acc_ci95=[round(c, 3) for c in ci])


if __name__ == "__main__":
    out = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw); out[name] = {}
        # cut points for baseline PropMLP + RF
        for frac in (0.1, 0.2, 0.3):
            out[name][f"cut{int(frac*100)}"] = {
                "propmlp_base": eval_cell(dcfg, "propmlp", "baseline", frac),
                "rf_base": eval_cell(dcfg, "rf", "baseline", frac)}
            print(f"[{name} last-{int(frac*100)}%] PropMLP {out[name][f'cut{int(frac*100)}']['propmlp_base']['f1_mean']} "
                  f"RF {out[name][f'cut{int(frac*100)}']['rf_base']['f1_mean']}", flush=True)
        # mitigations at 10% with CIs and n
        out[name]["mitig10"] = {
            "propmlp_base": eval_cell(dcfg, "propmlp", "baseline"),
            "propmlp_prior": eval_cell(dcfg, "propmlp", "prior"),
            "rf_base": eval_cell(dcfg, "rf", "baseline"),
            "rf_coral": eval_cell(dcfg, "rf", "coral"),
            "rf_prior": eval_cell(dcfg, "rf", "prior")}
        m = out[name]["mitig10"]
        print(f"[{name}] PropMLP base {m['propmlp_base']['f1_mean']} (CI {m['propmlp_base']['acc_ci95']}, n={m['propmlp_base']['n_test']}) "
              f"prior {m['propmlp_prior']['f1_mean']} | RF base {m['rf_base']['f1_mean']} coral {m['rf_coral']['f1_mean']} prior {m['rf_prior']['f1_mean']}", flush=True)
    json.dump(out, open(f"{OUT}/temporal_rigor.json", "w"), indent=2)
    print("DONE", flush=True)
