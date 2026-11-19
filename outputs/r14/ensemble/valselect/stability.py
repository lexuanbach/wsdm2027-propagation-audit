"""Post-hoc: how much of the 'different winner on every corpus' story is signal?

Story-clustered bootstrap (resample test stories within each seed, with
replacement, using the SAME resampled indices for every model so the pairing is
preserved), B resamples. For each resample, recompute every component's mean
macro-F1 over the 5 seeds and record which component leads.

This touches the test fold, but runs AFTER the selection rule is frozen and
no selection decision depends on it. It is a descriptive measure of leader
stability, which is exactly the quantity the paper's methodological complaint
is about.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PRED, SEEDS, DATASETS
from docless.eval.stats import macro_f1

ORDER = ["rf", "propmlp", "propmlp15", "histgbt15", "ebgcn", "gatv2", "dlf", "dlf15"]
B = int(os.environ.get("B", "2000"))


def load():
    R = {}
    for c in DATASETS:
        for f in sorted(glob.glob(f"{PRED}/{c}/*__seed*.json")):
            d = json.load(open(f))
            R.setdefault(c, {}).setdefault(d["model"], {})[d["seed"]] = d
    return {c: {m: v for m, v in ms.items() if set(v) == set(SEEDS)} for c, ms in R.items()}


def boot(R, c, pool, fold):
    P = {m: [np.asarray(R[c][m][s][f"{fold}_preds"]) for s in SEEDS] for m in pool}
    Y = [np.asarray(R[c][pool[0]][s][f"{fold}_labels"]) for s in SEEDS]
    rng = np.random.default_rng(12345)
    n = [len(y) for y in Y]
    wins = Counter()
    for _ in range(B):
        idx = [rng.integers(0, n[i], n[i]) for i in range(5)]
        sc = {}
        for m in pool:
            fs = []
            for i in range(5):
                yy = Y[i][idx[i]]
                if len(set(yy.tolist())) < 2:
                    fs.append(float("nan")); continue
                fs.append(float(macro_f1(P[m][i][idx[i]], yy)))
            sc[m] = float(np.nanmean(fs))
        wins[max(pool, key=lambda m: sc[m])] += 1
    return {m: round(wins[m] / B, 3) for m in pool}


def main():
    R = load()
    out = {}
    for c in sorted(R):
        pool = [m for m in ORDER if m in R[c]]
        if len(pool) < 2:
            continue
        pt = boot(R, c, pool, "test")
        pv = boot(R, c, pool, "val")
        lead_t = max(pool, key=lambda m: pt[m]); lead_v = max(pool, key=lambda m: pv[m])
        out[c] = dict(pool=pool, B=B,
                      p_leads_on_test=pt, p_leads_on_val=pv,
                      modal_test_leader=lead_t, modal_test_leader_prob=pt[lead_t],
                      modal_val_leader=lead_v, modal_val_leader_prob=pv[lead_v],
                      p_val_leader_also_leads_test=pt[lead_v])
        print(f"{c}: test-leader {lead_t} p={pt[lead_t]}   val-leader {lead_v} p={pv[lead_v]}"
              f"   P(val-leader leads test)={pt[lead_v]}", flush=True)
        print(f"   P(leads on test): {pt}", flush=True)
    json.dump(out, open(f"{ROOT}/leader_stability.json", "w"), indent=2)


if __name__ == "__main__":
    main()
