#!/usr/bin/env python3
"""Seed-paired t-tests: PropMLP vs the two fixed-budget GNNs (GCNFN, BiGCN).

Motivation: the paper attaches paired statistics to GATv2 /
EBGCN / DocLessFake / RF / HistGBT but never to GCNFN or BiGCN, which appear 7x
in main.tex with no p-value.

No per-story GNN predictions were dumped, so a story-clustered bootstrap is not
computable without a rerun.  But per-SEED macro-F1 is on disk for all four
corpora, so the seed-paired t-test -- the same secondary test the paper already
declares in the Statistics paragraph and already quotes for EBGCN on ReCOVery
(p=0.50) -- is computable at zero compute cost.

These are seed-level tests, reported OUTSIDE the 23-test Holm/BH family of the
identity-free regime, exactly as the existing EBGCN seed-paired t-test is.

Pairing validity: splits come from train.py:_stratified_split, which is seeded
by `seed` alone, so seed s gives the same split in every run directory.  This is
verified empirically in `check_determinism()` below: the `docless` arm is
bit-identical between outputs/r4 and outputs/r7.

Sources (all read, none synthesised):
  CoAID/ReCOVery GNN   : outputs/r7/main_table5/main_table.json
  CoAID/ReCOVery PropMLP: outputs/r9/propmlp_table/main_table.json
  PolitiFact            : outputs/r9/politifact/politifact_eval.json
  GossipCop (1500/class): outputs/r9/gossipcop_3k/gossipcop_eval.json
"""
import json
import os
from collections import defaultdict

from scipy import stats

ART = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(rel):
    with open(os.path.join(ART, rel)) as fh:
        return json.load(fh)


def from_main_table(rel):
    """-> {(dataset, method): [f1 by ascending seed]}"""
    rows = load(rel)["results"]
    acc = defaultdict(dict)
    for r in rows:
        acc[(r["dataset"], r["method"])][r["seed"]] = r["f1_all"]
    return {k: [v[s] for s in sorted(v)] for k, v in acc.items()}


def check_determinism():
    """seed -> split determinism across run directories, empirically."""
    r4 = from_main_table("outputs/r4/main_table5/main_table.json")
    r7 = from_main_table("outputs/r7/main_table5/main_table.json")
    out = {}
    for ds in ("coaid", "recovery"):
        a = [round(x, 4) for x in r4[(ds, "docless")]]
        b = [round(x, 4) for x in r7[(ds, "docless")]]
        out[ds] = {"r4_docless": a, "r7_docless": b, "identical": a == b}
    return out


def main():
    r7 = from_main_table("outputs/r7/main_table5/main_table.json")
    r9p = from_main_table("outputs/r9/propmlp_table/main_table.json")
    poli = load("outputs/r9/politifact/politifact_eval.json")["main"]
    goss = load("outputs/r9/gossipcop_3k/gossipcop_eval.json")["main"]

    series = {
        "coaid": {
            "propmlp": r9p[("coaid", "propmlp")],
            "gcnfn": r7[("coaid", "gcnfn")],
            "bigcn": r7[("coaid", "bigcn")],
        },
        "recovery": {
            "propmlp": r9p[("recovery", "propmlp")],
            "gcnfn": r7[("recovery", "gcnfn")],
            "bigcn": r7[("recovery", "bigcn")],
        },
        "politifact": {k: poli[k]["per_seed"] for k in ("propmlp", "gcnfn", "bigcn")},
        "gossipcop": {k: goss[k]["per_seed"] for k in ("propmlp", "gcnfn", "bigcn")},
    }

    # Table 2 cells these must reproduce, so we know we picked the right arms.
    table2 = {
        ("coaid", "propmlp"): 0.618, ("coaid", "gcnfn"): 0.442, ("coaid", "bigcn"): 0.539,
        ("recovery", "propmlp"): 0.659, ("recovery", "gcnfn"): 0.390, ("recovery", "bigcn"): 0.338,
        ("politifact", "propmlp"): 0.657, ("politifact", "gcnfn"): 0.550, ("politifact", "bigcn"): 0.536,
        ("gossipcop", "propmlp"): 0.836, ("gossipcop", "gcnfn"): 0.692, ("gossipcop", "bigcn"): 0.787,
    }

    result = {
        "test": "two-sided paired t-test over 5 seeds, macro-F1, PropMLP minus GNN",
        "family": "seed-level secondary test; NOT part of the 23-test Holm/BH "
                  "family of the identity-free regime",
        "determinism_check": check_determinism(),
        "means_reproduce_table2": {},
        "tests": {},
    }

    for ds, arms in series.items():
        for gnn in ("gcnfn", "bigcn"):
            a = arms["propmlp"]
            b = arms[gnn]
            assert len(a) == len(b) == 5, (ds, gnn, len(a), len(b))
            diff = [x - y for x, y in zip(a, b)]
            t, p = stats.ttest_rel(a, b)
            result["tests"][f"{ds}_propmlp_vs_{gnn}"] = {
                "propmlp_mean": round(sum(a) / 5, 4),
                f"{gnn}_mean": round(sum(b) / 5, 4),
                "mean_diff": round(sum(diff) / 5, 4),
                "per_seed_diff": [round(d, 4) for d in diff],
                "t_stat": round(float(t), 4),
                "p_value": float(f"{p:.4g}"),
                "positive": sum(diff) / 5 > 0,
                "significant_at_05": float(p) < 0.05,
            }

    for (ds, m), expected in table2.items():
        got = round(sum(series[ds][m]) / 5, 3)
        result["means_reproduce_table2"][f"{ds}_{m}"] = {
            "computed": got, "table2": expected, "match": abs(got - expected) <= 0.001,
        }

    n_pos = sum(v["positive"] for v in result["tests"].values())
    n_sig = sum(v["significant_at_05"] for v in result["tests"].values())
    result["summary"] = {
        "n_cells": len(result["tests"]),
        "n_positive": n_pos,
        "n_significant_at_05": n_sig,
    }

    outdir = os.path.join(ART, "outputs", "r11", "seed_paired_gnn")
    os.makedirs(outdir, exist_ok=True)
    dest = os.path.join(outdir, "seed_paired_gnn.json")
    with open(dest, "w") as fh:
        json.dump(result, fh, indent=2)

    bad = [k for k, v in result["means_reproduce_table2"].items() if not v["match"]]
    print("determinism (docless bit-identical r4 vs r7):",
          {k: v["identical"] for k, v in result["determinism_check"].items()})
    print("Table-2 reproduction mismatches:", bad or "none")
    for k, v in result["tests"].items():
        print(f"  {k:34s} d={v['mean_diff']:+.4f}  p={v['p_value']:.4g}"
              f"{'  *' if v['significant_at_05'] else ''}")
    print(f"positive in {n_pos}/{len(result['tests'])}, "
          f"significant at 0.05 in {n_sig}/{len(result['tests'])}")
    print("wrote", dest)


if __name__ == "__main__":
    main()
