"""Power / equivalence analysis.

Separates 'tie' from 'underpowered':
  * report the test-fold n per dataset explicitly;
  * a per-dataset minimum-detectable-effect (MDE) at 80% power for a paired
    accuracy comparison at the achieved n;
  * a TOST equivalence test (two one-sided, pre-specified margin delta=0.05
    accuracy points) for the key '<= PropMLP' comparisons, so 'ties' are backed
    by a powered equivalence argument rather than mere failure-to-reject.

Comparisons handled here are the cheap ones (GATv2, RF, HistGBT, PropMLP+ vs
PropMLP); DocLessFake-vs-PropMLP per-story deltas come from rev2_ssl_tree.py
(which trains the full SSL model) and are merged in the paper by hand.

Method: within each seed every model is evaluated on the SAME stratified test
split (same seed -> same _stratified_split), so per-story predictions are
aligned in test order. We pool per-(story,seed) correctness and use a
STORY-CLUSTERED bootstrap (resample test stories) for the delta CI -- the
conservative unit the paper already adopts.
"""
from __future__ import annotations
import json, os
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch, torch.nn.functional as F

from docless.baselines.propmlp import _MLP, _features
from docless.baselines.gatv2 import train_gatv2
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r10/power"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
MARGIN = 0.05  # pre-specified practical-equivalence margin (accuracy points)
DATASETS = {
    "coaid":      dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}


def _propmlp_preds(tr, te, rich, seed):
    _set_seed(seed)
    xtr, xte = _features(tr, rich=rich), _features(te, rich=rich)
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    xt = torch.tensor((xtr - mu) / sd, dtype=torch.float32); xe = torch.tensor((xte - mu) / sd, dtype=torch.float32)
    yt = torch.tensor([c.label for c in tr], dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(xe).argmax(-1).numpy()


def _tab_preds(tr, te, kind):
    xtr, xte = _features(tr), _features(te)
    sc = StandardScaler().fit(xtr); Xtr, Xte = sc.transform(xtr), sc.transform(xte)
    ytr = np.array([c.label for c in tr])
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=1)
    else:
        m = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=0)
    m.fit(Xtr, ytr); return m.predict(Xte)


def _cluster_boot_delta(story_ids, corr_a, corr_b, B=10000, seed=0):
    """Story-clustered bootstrap of mean(acc_a - acc_b). Positive => A better."""
    by = defaultdict(list)
    for sid, ca, cb in zip(story_ids, corr_a, corr_b):
        by[sid].append(ca - cb)
    sids = list(by); diffs = [np.array(by[s], dtype=float) for s in sids]
    obs = float(np.mean([d.mean() for d in diffs]))
    rng = np.random.default_rng(seed); samp = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(sids), len(sids))
        samp[b] = np.mean([diffs[i].mean() for i in idx])
    se = float(samp.std())
    lo95, hi95 = np.quantile(samp, [.025, .975])
    lo90, hi90 = np.quantile(samp, [.05, .95])
    return obs, se, (float(lo95), float(hi95)), (float(lo90), float(hi90))


def run():
    out = {"margin": MARGIN, "datasets": {}}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        story_ids, corr = [], defaultdict(list)
        seed_f1 = defaultdict(list); n_test = None
        for s in SEEDS:
            _set_seed(s); casc = _load_cascades(dcfg, s)
            tr, _va, te = _stratified_split(casc, dcfg.split_ratios, s)
            y = np.array([c.label for c in te]); sids = [c.story_id for c in te]
            n_test = len(te)
            preds = {
                "propmlp": _propmlp_preds(tr, te, False, s),
                "propmlp+": _propmlp_preds(tr, te, True, s),
                "rf": _tab_preds(tr, te, "rf"),
                "histgbt": _tab_preds(tr, te, "histgbt"),
            }
            g = train_gatv2(s, dcfg, device=torch.device("cpu"))
            preds["gatv2"] = np.array(g["preds"]); yg = np.array(g["labels"])
            # gatv2 evaluates the same split; align by assuming test order matches
            for m, p in preds.items():
                yy = yg if m == "gatv2" else y
                seed_f1[m].append(macro_f1(p, yy))
                corr[m].extend((p == yy).astype(float))
            story_ids.extend(sids)
        ref = np.array(corr["propmlp"])
        comps = {}
        for m in ["propmlp+", "rf", "histgbt", "gatv2"]:
            obs, se, ci95, ci90 = _cluster_boot_delta(story_ids, corr[m], corr["propmlp"])
            mde = 2.80 * se  # (z_.975 + z_.80) * SE for paired two-sided test at 80% power
            # TOST equivalence at +-MARGIN: equivalent if 90% CI within (-MARGIN, MARGIN)
            tost_equiv = bool(ci90[0] > -MARGIN and ci90[1] < MARGIN)
            t_p = float(stats.ttest_rel(np.array(seed_f1[m]), np.array(seed_f1["propmlp"])).pvalue)
            comps[f"{m}_vs_propmlp"] = dict(
                f1_A=round(float(np.mean(seed_f1[m])), 3), f1_propmlp=round(float(np.mean(seed_f1["propmlp"])), 3),
                acc_delta=round(obs, 4), se=round(se, 4), ci95=[round(c, 3) for c in ci95],
                ci90=[round(c, 3) for c in ci90], mde_80=round(mde, 3),
                tost_equiv_at_05=tost_equiv, seed_t_p=round(t_p, 4))
        out["datasets"][name] = dict(n_test=n_test, n_stories=len(set(story_ids)), comparisons=comps)
        print(f"[{name}] n_test={n_test}  " + "  ".join(
            f"{m.split('_')[0]} d={comps[m]['acc_delta']:+.3f} MDE={comps[m]['mde_80']:.3f} TOST={'Y' if comps[m]['tost_equiv_at_05'] else 'N'}"
            for m in comps), flush=True)
        json.dump(out, open(f"{OUT}/power.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
