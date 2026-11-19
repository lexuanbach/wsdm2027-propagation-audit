"""Unbiased, VAL-ONLY estimate of what per-corpus validation selection buys.

The plain val gain in val_contrast.json is optimistic: the component was chosen
by maximising the very val scores it is then credited with.  Here the rule is
cross-validated over seeds.  For each held-out seed k:

    * per-corpus selection picks, for each corpus, the component with the best
      mean val macro-F1 over the FOUR other seeds (same frozen tiebreak), then is
      scored on seed k's val fold;
    * every fixed single component is scored on the same seed-k val fold;
    * the best uniform single model is likewise chosen on the four other seeds.

Nothing here reads the test fold.  Seed k's val fold is out-of-sample for the
selection decision, so per-corpus selection can and does lose -- which is the
point: this is the first comparison in which the control is not guaranteed to win.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PRED, SEEDS, DATASETS
from docless.eval.stats import macro_f1

ORDER = ["rf", "propmlp", "propmlp15", "histgbt15", "ebgcn", "gatv2", "dlf", "dlf15"]

R = {}
for c in DATASETS:
    for f in sorted(glob.glob(f"{PRED}/{c}/*__seed*.json")):
        d = json.load(open(f))
        R.setdefault(c, {}).setdefault(d["model"], {})[d["seed"]] = d
R = {c: {m: v for m, v in ms.items() if set(v) == set(SEEDS)} for c, ms in R.items()}
corp = sorted(R)
POOL_TAG = os.environ.get("POOL", "full")
_full = {c: [m for m in ORDER if m in R[c]] for c in corp}
common = [m for m in ORDER if all(m in _full[c] for c in corp)]
# "full": selection may use every component available on that corpus.
# "common": selection is restricted to the components every corpus has, so the
# comparison against a fixed uniform component is over an identical menu.
pool = _full if POOL_TAG == "full" else {c: list(common) for c in corp}

V = {c: {m: [float(macro_f1(np.asarray(R[c][m][s]["val_preds"]),
                            np.asarray(R[c][m][s]["val_labels"]))) for s in SEEDS]
         for m in pool[c]} for c in corp}


def pick(c, seeds_in):
    return min(pool[c], key=lambda m: (-float(np.mean([V[c][m][i] for i in seeds_in])),
                                       float(np.std([V[c][m][i] for i in seeds_in])),
                                       ORDER.index(m)))


held_sel, held_uni, picks, unis = [], [], {}, []
per_corpus_held = {c: [] for c in corp}
for k in range(5):
    tr_seeds = [i for i in range(5) if i != k]
    p = {c: pick(c, tr_seeds) for c in corp}
    picks[k] = p
    for c in corp:
        per_corpus_held[c].append(V[c][p[c]][k])
    held_sel.append(float(np.mean([V[c][p[c]][k] for c in corp])))
    u = min(common, key=lambda m: (-float(np.mean([np.mean([V[c][m][i] for i in tr_seeds])
                                                   for c in corp])), ORDER.index(m)))
    unis.append(u)
    held_uni.append(float(np.mean([V[c][u][k] for c in corp])))

# every fixed single component, scored on the same held-out val folds
fixed = {m: [float(np.mean([V[c][m][k] for c in corp])) for k in range(5)] for m in common}
# per-corpus: the best fixed component on that corpus, scored the same way
per_corpus_fixed = {c: {m: V[c][m] for m in pool[c]} for c in corp}

sel_mean = float(np.mean(held_sel))
beat = {m: dict(held_out_val_mean=round(float(np.mean(fixed[m])), 4),
                selection_minus=round(sel_mean - float(np.mean(fixed[m])), 4),
                selection_wins_on_k=int(sum(held_sel[k] > fixed[m][k] + 1e-12 for k in range(5))))
        for m in common}

out = dict(
    design="leave-one-seed-out over the 5 paired val folds; test fold never read",
    corpora=corp, pool=pool, common_pool=common,
    loso_picks_per_seed={str(k): picks[k] for k in picks},
    loso_pick_stable_per_corpus={c: bool(len({picks[k][c] for k in picks}) == 1) for c in corp},
    loso_uniform_picks=unis,
    held_out_val_selection_per_seed=[round(x, 4) for x in held_sel],
    held_out_val_selection_mean=round(sel_mean, 4),
    held_out_val_uniform_per_seed=[round(x, 4) for x in held_uni],
    held_out_val_uniform_mean=round(float(np.mean(held_uni)), 4),
    selection_minus_uniform=round(sel_mean - float(np.mean(held_uni)), 4),
    selection_beats_uniform_on_k=int(sum(held_sel[k] > held_uni[k] + 1e-12 for k in range(5))),
    vs_every_fixed_common_component=beat,
    selection_beats_every_fixed_component_in_mean=bool(
        all(sel_mean > np.mean(fixed[m]) + 1e-12 for m in common)),
    per_corpus_held_out_val=
        {c: dict(selection_mean=round(float(np.mean(per_corpus_held[c])), 4),
                 best_fixed_component_on_this_corpus=max(pool[c], key=lambda m: np.mean(V[c][m])),
                 best_fixed_mean=round(float(max(np.mean(V[c][m]) for m in pool[c])), 4),
                 selection_minus_best_fixed=round(
                     float(np.mean(per_corpus_held[c]) - max(np.mean(V[c][m]) for m in pool[c])), 4))
         for c in corp},
    note="Held-out val means are NOT comparable to the main table's test numbers; they are a "
         "val-fold-internal measure of whether the selection rule generalises across seeds.",
)
out["selection_pool_tag"] = POOL_TAG
json.dump(out, open(f"{ROOT}/val_loso_{POOL_TAG}.json", "w"), indent=2)
print(f"LOSO held-out VAL mean over {len(corp)} corpora  [selection pool = {POOL_TAG}]:")
print(f"  per-corpus selection      {out['held_out_val_selection_mean']:.4f}  "
      f"per seed {out['held_out_val_selection_per_seed']}")
print(f"  best uniform single       {out['held_out_val_uniform_mean']:.4f}  "
      f"(picks {unis})  -> selection wins on "
      f"{out['selection_beats_uniform_on_k']}/5 held-out seeds, "
      f"delta {out['selection_minus_uniform']:+.4f}")
print("  vs each fixed common component (held-out val):")
for m in common:
    b = beat[m]
    print(f"    {m:11s} {b['held_out_val_mean']:.4f}  selection{b['selection_minus']:+.4f}  "
          f"wins {b['selection_wins_on_k']}/5")
print(f"  selection beats EVERY fixed common component in held-out val mean: "
      f"{out['selection_beats_every_fixed_component_in_mean']}")
print("  LOSO pick stability:", out["loso_pick_stable_per_corpus"])
print("  LOSO picks per held-out seed:", json.dumps(out["loso_picks_per_seed"]))
