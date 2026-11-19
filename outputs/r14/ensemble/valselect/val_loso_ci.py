"""Story-clustered CI for the leave-one-seed-out validation gain.

The LOSO picks themselves are held fixed (they were made on the four *other*
seeds' val folds and do not depend on the resampled stories). What is resampled
is the held-out val fold itself: for each (corpus, held-out seed) we draw stories
with replacement and reuse the SAME indices for every component, so the
comparison stays paired story-for-story.

VAL fold only. Never reads test.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PRED, SEEDS, DATASETS
from docless.eval.stats import macro_f1

ORDER = ["rf", "propmlp", "propmlp15", "histgbt15", "ebgcn", "gatv2", "dlf", "dlf15"]
B = int(os.environ.get("B", "3000"))

R = {}
for c in DATASETS:
    for f in sorted(glob.glob(f"{PRED}/{c}/*__seed*.json")):
        d = json.load(open(f))
        R.setdefault(c, {}).setdefault(d["model"], {})[d["seed"]] = d
R = {c: {m: v for m, v in ms.items() if set(v) == set(SEEDS)} for c, ms in R.items()}
corp = sorted(R)
pool = {c: [m for m in ORDER if m in R[c]] for c in corp}
common = [m for m in ORDER if all(m in pool[c] for c in corp)]

V = {c: {m: [float(macro_f1(np.asarray(R[c][m][s]["val_preds"]),
                            np.asarray(R[c][m][s]["val_labels"]))) for s in SEEDS]
         for m in pool[c]} for c in corp}
Y = {c: [np.asarray(R[c][pool[c][0]][s]["val_labels"]) for s in SEEDS] for c in corp}
Pd = {c: {m: [np.asarray(R[c][m][s]["val_preds"]) for s in SEEDS] for m in pool[c]}
      for c in corp}


def pick(c, seeds_in):
    return min(pool[c], key=lambda m: (-float(np.mean([V[c][m][i] for i in seeds_in])),
                                       float(np.std([V[c][m][i] for i in seeds_in])),
                                       ORDER.index(m)))


sel = {k: {c: pick(c, [i for i in range(5) if i != k]) for c in corp} for k in range(5)}
uni = {k: min(common, key=lambda m: (-float(np.mean(
    [np.mean([V[c][m][i] for i in range(5) if i != k]) for c in corp])), ORDER.index(m)))
    for k in range(5)}

rng = np.random.default_rng(4242)
draws = {"selection": [], "uniform": [], **{m: [] for m in common}}
for _ in range(B):
    acc = {k: [] for k in draws}
    for c in corp:
        for k in range(5):
            y = Y[c][k]
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if len(set(yy.tolist())) < 2:
                continue
            acc["selection"].append(float(macro_f1(Pd[c][sel[k][c]][k][idx], yy)))
            acc["uniform"].append(float(macro_f1(Pd[c][uni[k]][k][idx], yy)))
            for m in common:
                acc[m].append(float(macro_f1(Pd[c][m][k][idx], yy)))
    for k in draws:
        draws[k].append(float(np.mean(acc[k])))

obs = {"selection": float(np.mean([V[c][sel[k][c]][k] for c in corp for k in range(5)])),
       "uniform": float(np.mean([V[c][uni[k]][k] for c in corp for k in range(5)]))}
for m in common:
    obs[m] = float(np.mean([V[c][m][k] for c in corp for k in range(5)]))

S = np.asarray(draws["selection"])
out = {"B": B, "corpora": corp, "common_pool": common,
       "design": "story-clustered paired bootstrap of the held-out val folds; LOSO picks fixed",
       "observed": {k: round(v, 4) for k, v in obs.items()},
       "contrasts": {}}
for m in ["uniform"] + common:
    d = S - np.asarray(draws[m])
    out["contrasts"][m] = dict(
        observed_gain=round(obs["selection"] - obs[m], 4),
        ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
        p_two_sided=round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 4),
        excludes_zero=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0))
json.dump(out, open(f"{ROOT}/val_loso_ci.json", "w"), indent=2)
print(f"held-out val, selection = {obs['selection']:.4f}   (B={B}, story-clustered, paired)")
for m in ["uniform"] + common:
    ct = out["contrasts"][m]
    lab = "best uniform (re-chosen per fold)" if m == "uniform" else f"{m} everywhere"
    print(f"  vs {lab:34s} {obs[m]:.4f}  gain {ct['observed_gain']:+.4f}  "
          f"95% CI [{ct['ci95'][0]:+.4f}, {ct['ci95'][1]:+.4f}]  p={ct['p_two_sided']}  "
          f"sig={ct['excludes_zero']}")
