"""Conservative variance + multiplicity control.

(a) Seed-level paired t-tests (5 independent seeds -> fold-disjoint at the seed
    level) for the feature/tabular comparisons, alongside a story-CLUSTERED
    bootstrap (resample test stories, not (story,seed) pairs) so the estimate is
    not anti-conservative.
(b) Holm-Bonferroni and Benjamini-Hochberg (FDR) correction over the FULL family
    of paired tests reported in the paper, so the celebrated positives are judged
    against a corrected threshold.

DocLessFake / GATv2 cells reuse their already-reported p-values (those models are
expensive); the fast feature/tabular cells are recomputed here with both tests.
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch, torch.nn.functional as F

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r10/rev_stats"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
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
    ytr = torch.tensor([c.label for c in tr], dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(xe).argmax(-1).numpy()


def _tab_preds(tr, te, kind, seed):
    xtr, xte = _features(tr), _features(te)
    sc = StandardScaler().fit(xtr); Xtr, Xte = sc.transform(xtr), sc.transform(xte)
    ytr = np.array([c.label for c in tr])
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=0, n_jobs=1)
    else:
        m = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1, random_state=0)
    m.fit(Xtr, ytr); return m.predict(Xte)


def cluster_bootstrap(story_ids, lossA, lossB, B=10000, seed=0):
    """Resample STORIES (clusters), pooling each story's per-seed loss diffs."""
    from collections import defaultdict
    by = defaultdict(list)
    for sid, la, lb in zip(story_ids, lossA, lossB):
        by[sid].append(lb - la)  # positive => A better (lower loss)
    sids = list(by); rng = np.random.default_rng(seed)
    diffs = [np.array(by[s]) for s in sids]
    obs = np.mean([d.mean() for d in diffs])
    samp = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(sids), len(sids))
        samp[b] = np.mean([diffs[i].mean() for i in idx])
    c = samp - obs
    p = 2 * min((c >= abs(obs)).mean(), (c <= -abs(obs)).mean())
    return float(obs), float(min(p, 1.0))


def run():
    comparisons = {}  # (dataset, "A_vs_B") -> dict
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        per_seed = {m: [] for m in ["propmlp", "propmlp+", "rf", "histgbt"]}
        story_ids, loss = [], {m: [] for m in ["propmlp", "propmlp+", "rf", "histgbt"]}
        for s in SEEDS:
            _set_seed(s); casc = _load_cascades(dcfg, s)
            tr, _va, te = _stratified_split(casc, dcfg.split_ratios, s)
            y = np.array([c.label for c in te]); sids = [c.story_id for c in te]
            preds = {"propmlp": _propmlp_preds(tr, te, False, s),
                     "propmlp+": _propmlp_preds(tr, te, True, s),
                     "rf": _tab_preds(tr, te, "rf", s),
                     "histgbt": _tab_preds(tr, te, "histgbt", s)}
            for m in preds:
                per_seed[m].append(macro_f1(preds[m], y))
                loss[m].extend((preds[m] != y).astype(float))
            story_ids.extend(sids)
        # comparisons vs PropMLP (and DocLessFake+ analogue PropMLP+)
        for A, Bm in [("propmlp+", "propmlp"), ("rf", "propmlp"), ("histgbt", "propmlp")]:
            fa, fb = np.array(per_seed[A]), np.array(per_seed[Bm])
            t_p = float(stats.ttest_rel(fa, fb).pvalue)
            obs, cb_p = cluster_bootstrap(story_ids, loss[A], loss[Bm])
            comparisons[f"{name}:{A}_vs_{Bm}"] = dict(
                f1_A=round(float(fa.mean()), 3), f1_B=round(float(fb.mean()), 3),
                seed_paired_t_p=round(t_p, 4), cluster_bootstrap_p=round(cb_p, 4),
                acc_diff_A_minus_B=round(obs, 4))
        print(f"[{name}] done", flush=True)

    # ---- multiplicity over the full family of paired tests in the paper ----
    family = {  # name -> raw p (paper-reported + newly computed feature/tabular)
        "DLF>PropMLP CoAID": 0.41, "DLF>PropMLP ReCOVery": 0.63,
        "DLF>PropMLP PolitiFact": 0.23, "DLF>PropMLP GossipCop": 0.25,
        "GATv2 vs PropMLP CoAID": 0.68, "GATv2 vs PropMLP ReCOVery": 0.10,
        "GATv2 vs PropMLP PolitiFact": 0.96, "GATv2 vs PropMLP GossipCop": 0.001,
        "hidden128 struct null": 0.87,
        "DLF+ vs PropMLP+ ReCOVery": 0.69, "DLF+ vs PropMLP+ PolitiFact": 0.65,
    }
    for k, v in comparisons.items():
        if "propmlp+_vs" in k or "rf_vs" in k or "histgbt_vs" in k:
            family[k] = v["cluster_bootstrap_p"]
    items = sorted(family.items(), key=lambda kv: kv[1]); ps = np.array([p for _, p in items]); n = len(ps)
    holm = np.minimum.accumulate((n - np.arange(n)) * ps[::-1])[::-1]  # not used; compute properly below
    holm = np.empty(n); run_max = 0.0
    for i in range(n):
        run_max = max(run_max, (n - i) * ps[i]); holm[i] = min(run_max, 1.0)
    bh = np.empty(n); run_min = 1.0
    for i in range(n - 1, -1, -1):
        run_min = min(run_min, ps[i] * n / (i + 1)); bh[i] = min(run_min, 1.0)
    mult = [dict(test=items[i][0], raw_p=round(float(ps[i]), 4), holm_p=round(float(holm[i]), 4),
                 bh_p=round(float(bh[i]), 4), holm_sig=bool(holm[i] < 0.05), bh_sig=bool(bh[i] < 0.05))
            for i in range(n)]
    out = dict(family_size=n, comparisons=comparisons, multiplicity=mult)
    json.dump(out, open(f"{OUT}/rev_stats.json", "w"), indent=2)
    print("\n=== MULTIPLICITY (family size", n, ") ===")
    for m in mult:
        print(f"  {m['test']:34s} raw {m['raw_p']:.4f}  Holm {m['holm_p']:.4f}{'*' if m['holm_sig'] else ' '}  BH {m['bh_p']:.4f}{'*' if m['bh_sig'] else ' '}")
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
