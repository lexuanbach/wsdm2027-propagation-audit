"""How much resolving power does each corpus's validation fold actually have?

For every (corpus, component) we story-bootstrap the val fold (resampling
stories within each seed, same indices for every component) and report the SD of
the 5-seed mean val macro-F1. Comparing that SD to the val margin between the
selected component and the runner-up says whether the selection is signal.
Uses the VAL fold only -- no test data is touched here.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PRED, SEEDS, DATASETS
from docless.eval.stats import macro_f1

ORDER = ["rf", "propmlp", "propmlp15", "histgbt15", "ebgcn", "gatv2", "dlf", "dlf15"]
B = int(os.environ.get("B", "1000"))

R = {}
for c in DATASETS:
    for f in sorted(glob.glob(f"{PRED}/{c}/*__seed*.json")):
        d = json.load(open(f))
        R.setdefault(c, {}).setdefault(d["model"], {})[d["seed"]] = d
R = {c: {m: v for m, v in ms.items() if set(v) == set(SEEDS)} for c, ms in R.items()}

out = {}
for c in sorted(R):
    pool = [m for m in ORDER if m in R[c]]
    P = {m: [np.asarray(R[c][m][s]["val_preds"]) for s in SEEDS] for m in pool}
    Y = [np.asarray(R[c][pool[0]][s]["val_labels"]) for s in SEEDS]
    n = [len(y) for y in Y]
    rng = np.random.default_rng(999)
    draws = {m: [] for m in pool}
    for _ in range(B):
        idx = [rng.integers(0, n[i], n[i]) for i in range(5)]
        for m in pool:
            fs = []
            for i in range(5):
                yy = Y[i][idx[i]]
                fs.append(float("nan") if len(set(yy.tolist())) < 2
                          else float(macro_f1(P[m][i][idx[i]], yy)))
            draws[m].append(float(np.nanmean(fs)))
    sd = {m: round(float(np.std(draws[m])), 4) for m in pool}
    means = {m: float(np.mean([macro_f1(P[m][i], Y[i]) for i in range(5)])) for m in pool}
    srt = sorted(pool, key=lambda m: -means[m])
    margin = means[srt[0]] - means[srt[1]]
    # P(the observed val leader is also the val leader in a bootstrap replicate)
    arr = np.array([draws[m] for m in pool])
    lead = np.array(pool)[arr.argmax(0)]
    pl = float((lead == srt[0]).mean())
    out[c] = dict(n_val_per_seed=n[0], pool=pool, val_mean={m: round(means[m], 4) for m in pool},
                  bootstrap_sd_of_val_mean=sd,
                  mean_bootstrap_sd=round(float(np.mean(list(sd.values()))), 4),
                  val_leader=srt[0], val_runner_up=srt[1],
                  val_leader_margin=round(margin, 4),
                  margin_over_sd=round(margin / np.mean(list(sd.values())), 2),
                  p_same_val_leader_under_resampling=round(pl, 3))
    print(f"{c}: n_val={n[0]}  leader {srt[0]} over {srt[1]} by {margin:.4f}; "
          f"bootstrap SD {out[c]['mean_bootstrap_sd']:.4f}  margin/SD={out[c]['margin_over_sd']}  "
          f"P(same val leader)={pl:.3f}", flush=True)
json.dump(out, open(f"{ROOT}/val_resolution.json", "w"), indent=2)
