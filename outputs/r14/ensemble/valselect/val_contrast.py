"""Pre-registered VAL-fold contrast: per-corpus validation selection vs the best
single model applied uniformly to all corpora (also chosen on val).

Story-clustered paired bootstrap over the VAL fold only. This is the only
comparison this combiner is entitled to make before the judge scores test --
and it is deliberately reported because it shows the ceiling of the control.
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
full = {c: [m for m in ORDER if m in R[c]] for c in corp}
common = [m for m in ORDER if all(m in full[c] for c in corp)]

vm = {c: {m: float(np.mean([macro_f1(np.asarray(R[c][m][s]["val_preds"]),
                                     np.asarray(R[c][m][s]["val_labels"])) for s in SEEDS]))
          for m in full[c]} for c in corp}
selA = {c: min(full[c], key=lambda m: (-vm[c][m], ORDER.index(m))) for c in corp}
uni = min(common, key=lambda m: (-float(np.mean([vm[c][m] for c in corp])), ORDER.index(m)))

rng = np.random.default_rng(2027)
diffs = []
for _ in range(B):
    da, du = [], []
    for c in corp:
        for i, s in enumerate(SEEDS):
            y = np.asarray(R[c][selA[c]][s]["val_labels"])
            idx = rng.integers(0, len(y), len(y))
            yy = y[idx]
            if len(set(yy.tolist())) < 2:
                continue
            da.append(float(macro_f1(np.asarray(R[c][selA[c]][s]["val_preds"])[idx], yy)))
            du.append(float(macro_f1(np.asarray(R[c][uni][s]["val_preds"])[idx], yy)))
    diffs.append(float(np.mean(da) - np.mean(du)))
diffs = np.asarray(diffs)
obs = float(np.mean([vm[c][selA[c]] for c in corp]) - np.mean([vm[c][uni] for c in corp]))
out = dict(B=B, corpora=corp, common_pool=common,
           per_corpus_selection=selA, best_uniform_single_on_val=uni,
           val_mean_selection=round(float(np.mean([vm[c][selA[c]] for c in corp])), 4),
           val_mean_uniform=round(float(np.mean([vm[c][uni] for c in corp])), 4),
           observed_val_gain=round(obs, 4),
           ci95=[round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
           p_two_sided=round(float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())), 4),
           note="VAL fold only. Positive gain = per-corpus selection beats the single best "
                "uniform model on the fold selection was made on -- which is partly guaranteed "
                "by construction and is NOT evidence about test.")
json.dump(out, open(f"{ROOT}/val_contrast.json", "w"), indent=2)
print(json.dumps(out, indent=2))
