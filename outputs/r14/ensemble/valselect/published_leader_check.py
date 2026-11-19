"""Post-hoc diagnostic tied directly to the paper's claim.

The main table's headline is "a DIFFERENT winner on each corpus".  This asks the
only question a practitioner cares about: could that winner have been identified
WITHOUT looking at the test fold?

Three leader definitions are compared per corpus:
  published_leader   the winner in the paper's frozen main table (test fold)
  regen_test_leader  the winner among the regenerated components (test fold)
  val_pick           what the frozen val-fold selection rule chose (val fold only)

No selection decision depends on anything here; the rule was frozen in
SELECTION_RULE.json before any of this ran.
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, SEEDS
from analyze import ORDER, PRETTY, PUBLISHED, load, f1s, select

# the paper's frozen main-table leader per corpus (test fold, 5-seed mean)
PUBLISHED_LEADER = {"coaid": "dlf", "recovery": "propmlp15",
                    "politifact": "dlf15", "gossipcop": "histgbt15"}
PUBLISHED_RUNNERUP = {"coaid": "histgbt15", "recovery": "ebgcn",
                      "politifact": "propmlp15", "gossipcop": "rf"}

R = load()
val = {c: {m: f1s(R, c, m, "val") for m in R[c]} for c in R}
test = {c: {m: f1s(R, c, m, "test") for m in R[c]} for c in R}
pools = {c: [m for m in ORDER if m in R[c]] for c in R}

out = {"note": "val_pick comes from val fields only; everything else is post-hoc reporting.",
       "per_corpus": {}}
hits_pub = hits_regen = n_eval = 0
for c in sorted(pools):
    pool = pools[c]
    if len(pool) < 2:
        continue
    A, Bm = select(val, R, c, pool)
    regen = max(pool, key=lambda m: np.mean(test[c][m]))
    pub = PUBLISHED_LEADER[c]
    pub_in_pool = pub in pool
    n_eval += 1
    hits_regen += int(A == regen)
    hits_pub += int(pub_in_pool and A == pub)
    out["per_corpus"][c] = dict(
        pool=pool, n_pool=len(pool),
        val_pick=A, val_pick_pretty=PRETTY[A],
        val_pick_val_mean=round(float(np.mean(val[c][A])), 4),
        val_pick_test_mean=round(float(np.mean(test[c][A])), 4),
        published_leader=pub, published_leader_pretty=PRETTY[pub],
        published_leader_value=PUBLISHED[c][pub],
        published_runner_up=PUBLISHED_RUNNERUP[c],
        published_leader_in_regen_pool=bool(pub_in_pool),
        val_pick_equals_published_leader=bool(pub_in_pool and A == pub),
        regen_test_leader=regen, regen_test_leader_pretty=PRETTY[regen],
        regen_test_leader_mean=round(float(np.mean(test[c][regen])), 4),
        val_pick_equals_regen_test_leader=bool(A == regen),
        val_rank_of_regen_test_leader=1 + sorted(
            pool, key=lambda m: -np.mean(val[c][m])).index(regen),
        val_rank_of_published_leader=(1 + sorted(
            pool, key=lambda m: -np.mean(val[c][m])).index(pub)) if pub_in_pool else None,
        test_rank_of_val_pick=1 + sorted(
            pool, key=lambda m: -np.mean(test[c][m])).index(A),
        regret_vs_regen_leader=round(float(np.mean(test[c][regen]) - np.mean(test[c][A])), 4),
    )
out["headline"] = dict(
    n_corpora=n_eval,
    val_pick_recovers_regen_test_leader=f"{hits_regen}/{n_eval}",
    val_pick_recovers_published_leader=f"{hits_pub}/{n_eval}",
    mean_regret_vs_regen_leader=round(float(np.mean(
        [out["per_corpus"][c]["regret_vs_regen_leader"] for c in out["per_corpus"]])), 4),
    distinct_val_picks=len({out["per_corpus"][c]["val_pick"] for c in out["per_corpus"]}),
    distinct_regen_test_leaders=len({out["per_corpus"][c]["regen_test_leader"]
                                     for c in out["per_corpus"]}),
    distinct_published_leaders=len({PUBLISHED_LEADER[c] for c in out["per_corpus"]}),
)
json.dump(out, open(f"{ROOT}/published_leader_check.json", "w"), indent=2)
print(f"{'corpus':11s} {'|pool|':>6s} {'VAL PICK':>11s} {'REGEN LEADER':>13s} {'PUB LEADER':>11s} "
      f"{'valRank(regen)':>15s} {'testRank(pick)':>15s} {'regret':>7s}")
for c in sorted(out["per_corpus"]):
    d = out["per_corpus"][c]
    print(f"{c:11s} {d['n_pool']:6d} {d['val_pick']:>11s} {d['regen_test_leader']:>13s} "
          f"{d['published_leader']:>11s} {d['val_rank_of_regen_test_leader']:>15d} "
          f"{d['test_rank_of_val_pick']:>15d} {d['regret_vs_regen_leader']:+7.4f}")
print()
print(json.dumps(out["headline"], indent=2))
