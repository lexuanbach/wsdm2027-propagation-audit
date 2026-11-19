"""Does any neural component beat the boosted tree?

We ask whether the one load-bearing neural ingredient (SSL-pretrained
summary fusion, the only audit branch component that helps, -0.056 F1 if removed)
actually beats a gradient-boosted tree on identical features. We make the
sharpest, cheapest version of that test: the strongest NEURAL summary model
(PropMLP+, a 15-feature MLP -- the same network family the SSL fusion reduces to
on summary features) vs the strongest TREE (HistGBT-15d) on the SAME 15 features,
paired across 5 seeds with a story-clustered bootstrap. We also re-state the
flagship SSL model (DocLessFake) vs HistGBT at the seed level from the main
table. If no neural model beats the tree, the negative result sharpens: no
neural component -- SSL fusion included -- survives the audit against a tree.
"""
from __future__ import annotations
import json, os
from collections import defaultdict
import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import torch, torch.nn.functional as F

from docless.baselines.propmlp import _MLP, _features
from docless.eval.stats import macro_f1
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

OUT = "outputs/r10/ssl_tree"; os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}
# flagship SSL model (DocLessFake) vs HistGBT, paper main-table means (5 seeds)
DLF_MEAN = {"coaid": 0.654, "recovery": 0.663, "politifact": 0.678, "gossipcop": 0.842}


def _mlp_preds(Xtr, ytr, Xte, seed):
    _set_seed(seed)
    xt = torch.tensor(Xtr, dtype=torch.float32); xe = torch.tensor(Xte, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    m = _MLP(in_dim=xt.shape[1], hidden=64); opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(25):
        m.train(); opt.zero_grad(); F.cross_entropy(m(xt), yt).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(xe).argmax(-1).numpy()


def _cluster_p(story_ids, corr_a, corr_b, B=10000, seed=0):
    by = defaultdict(list)
    for sid, ca, cb in zip(story_ids, corr_a, corr_b):
        by[sid].append(ca - cb)
    sids = list(by); diffs = [np.array(by[s], float) for s in sids]
    obs = float(np.mean([d.mean() for d in diffs]))
    rng = np.random.default_rng(seed); samp = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, len(sids), len(sids)); samp[b] = np.mean([diffs[i].mean() for i in idx])
    c = samp - obs; p = 2 * min((c >= abs(obs)).mean(), (c <= -abs(obs)).mean())
    return obs, float(min(p, 1.0)), [float(np.quantile(samp, .025)), float(np.quantile(samp, .975))]


def run():
    out = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        story_ids, c_mlp, c_tree = [], [], []
        f1_mlp, f1_tree = [], []
        for s in SEEDS:
            _set_seed(s); casc = _load_cascades(dcfg, s)
            tr, _va, te = _stratified_split(casc, dcfg.split_ratios, s)
            xtr, xte = _features(tr, rich=True), _features(te, rich=True)   # identical 15 feats
            sc = StandardScaler().fit(xtr); Xtr, Xte = sc.transform(xtr), sc.transform(xte)
            ytr = np.array([c.label for c in tr]); yte = np.array([c.label for c in te])
            pm = _mlp_preds(Xtr, ytr, Xte, s)                       # PropMLP+ (neural)
            tg = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.1,
                                                random_state=s).fit(Xtr, ytr).predict(Xte)  # HistGBT (tree)
            f1_mlp.append(macro_f1(pm, yte)); f1_tree.append(macro_f1(tg, yte))
            c_mlp.extend((pm == yte).astype(float)); c_tree.extend((tg == yte).astype(float))
            story_ids.extend([c.story_id for c in te])
        obs, p, ci = _cluster_p(story_ids, c_mlp, c_tree)  # mlp - tree, positive => neural wins
        seed_t = float(stats.ttest_rel(np.array(f1_mlp), np.array(f1_tree)).pvalue)
        out[name] = dict(
            propmlp_plus_f1=round(float(np.mean(f1_mlp)), 3), histgbt15_f1=round(float(np.mean(f1_tree)), 3),
            neural_minus_tree_acc=round(obs, 4), cluster_p=round(p, 4), ci95=[round(v, 3) for v in ci],
            seed_t_p=round(seed_t, 4), neural_wins=bool(obs > 0 and p < 0.05),
            doclessfake_f1=DLF_MEAN[name], dlf_beats_tree=bool(DLF_MEAN[name] > np.mean(f1_tree)))
        d = out[name]
        print(f"[{name}] PropMLP+ {d['propmlp_plus_f1']} vs HistGBT15 {d['histgbt15_f1']}  "
              f"neural-tree={d['neural_minus_tree_acc']:+.3f} (cluster p={d['cluster_p']}, seed-t {d['seed_t_p']})  "
              f"DLF {d['doclessfake_f1']} beats_tree={d['dlf_beats_tree']}", flush=True)
        json.dump(out, open(f"{OUT}/ssl_tree.json", "w"), indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    run()
