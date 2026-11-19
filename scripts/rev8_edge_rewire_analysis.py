"""Paper-facing analysis of the EDGE degree-preserving rewire control.

Reads the intact run (outputs/r11/graph_informative_clustered) and the rewired
run (outputs/r13/edge_rewire), and produces

  * the seed-level paired test of rewired vs intact BiGCN on every corpus --
    the same secondary, outside-the-Holm/BH-family test the paper already uses;
  * an internal control: the summary baselines must be bit-identical between
    the two runs, because only the graph model's edge_index was rewired;
  * tree-shape descriptors that say how much arrangement a degree-preserving
    rewire can remove on each corpus in the first place.

Usage: PYTHONPATH=src python scripts/rev8_edge_rewire_analysis.py
Output: outputs/r13/edge_rewire/analysis.json
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import statistics
import sys
import zlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
_spec = importlib.util.spec_from_file_location(
    "rev6_graph_informative", HERE / "rev6_graph_informative.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["rev6_graph_informative"] = G
_spec.loader.exec_module(G)
sys.path.insert(0, str(HERE))
from rev8_edge_rewire import rewire_parent  # noqa: E402

INTACT = ROOT / "outputs" / "r11" / "graph_informative_clustered"
REWIRED = ROOT / "outputs" / "r13" / "edge_rewire"
DATASETS = ["upfd_gossipcop", "upfd_politifact", "twitter15", "twitter16"]
SUMMARY = ["logreg", "rf", "histgbt", "mlp"]


def paired_t(a, b):
    """Two-sided paired t-test on 5 seeds, computed from scratch (no scipy dep)."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    n = len(d)
    sd = d.std(ddof=1)
    if sd == 0:
        return {"t": 0.0, "p": 1.0, "mean_delta": 0.0, "n_seeds": n}
    t = float(d.mean() / (sd / np.sqrt(n)))
    # two-sided p from the t distribution with n-1 df, via the incomplete beta
    from math import lgamma, log

    def betacf(a, b, x, itmax=200, eps=3e-16):
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, dd = 1.0, 1.0 - qab * x / qap
        dd = 1.0 / (1e-300 if abs(dd) < 1e-300 else dd)
        h = dd
        for m in range(1, itmax + 1):
            m2 = 2 * m
            aa = m * (b - m) * x / ((qam + m2) * (a + m2))
            dd = 1.0 + aa * dd; c = 1.0 + aa / c
            dd = 1.0 / (1e-300 if abs(dd) < 1e-300 else dd); h *= dd * c
            aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
            dd = 1.0 + aa * dd; c = 1.0 + aa / c
            dd = 1.0 / (1e-300 if abs(dd) < 1e-300 else dd)
            de = dd * c; h *= de
            if abs(de - 1.0) < eps:
                break
        return h

    def betai(a, b, x):
        if x <= 0: return 0.0
        if x >= 1: return 1.0
        bt = np.exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
        if x < (a + 1) / (a + b + 2):
            return bt * betacf(a, b, x) / a
        return 1.0 - bt * betacf(b, a, 1 - x) / b

    df = n - 1
    p = float(betai(df / 2.0, 0.5, df / (df + t * t)))
    return {"t": round(t, 3), "p": round(p, 4), "mean_delta": round(float(d.mean()), 4),
            "per_seed_delta": [round(float(x), 4) for x in d], "n_seeds": n}


def shape_descriptors(items, limit=None):
    """How star-like is the corpus, and how much does the rewire move it?"""
    keys = ["depth", "mean_depth", "structural_virality", "max_width"]
    its = items if limit is None else items[:limit]
    frac_depth1, moved = [], {k: [] for k in keys}
    d_before = {k: [] for k in keys}
    for it in its:
        key = str(it["id"])
        newp = rewire_parent(it["parent"],
                             seed=int(zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF))[0]
        sb, sa = G.tree_stats(it["parent"]), G.tree_stats(newp)
        n = sb["n_nodes"]
        if n > 0:
            frac_depth1.append(sb["root_fanout"] / n)
        for k in keys:
            d_before[k].append(float(sb[k]))
            moved[k].append(abs(float(sa[k]) - float(sb[k])))
    out = {"n_trees": len(its),
           "mean_frac_nodes_attached_to_root": round(float(np.mean(frac_depth1)), 4)}
    for k in keys:
        out[k] = {"mean_intact": round(float(np.mean(d_before[k])), 4),
                  "mean_abs_change_under_rewire": round(float(np.mean(moved[k])), 4),
                  "frac_trees_changed": round(float(np.mean(np.asarray(moved[k]) > 1e-9)), 4)}
    return out


def main():
    res = {"design": "degree-preserving connected double-edge swap of the graph model's "
                     "edges only; summary baselines, node features, labels, splits and "
                     "seeds unchanged",
           "intact_source": str(INTACT.relative_to(ROOT)),
           "rewired_source": str(REWIRED.relative_to(ROOT)),
           "per_corpus": {}}
    for ds in DATASETS:
        a = json.loads((INTACT / f"{ds}.json").read_text())
        b = json.loads((REWIRED / f"{ds}.json").read_text())
        ctrl = {k: a["results"][k]["per_seed"] == b["results"][k]["per_seed"] for k in SUMMARY}
        ia, ib = a["results"]["bigcn"]["per_seed"], b["results"]["bigcn"]["per_seed"]
        res["per_corpus"][ds] = {
            "bigcn_intact_mean": a["results"]["bigcn"]["f1_mean"],
            "bigcn_rewired_mean": b["results"]["bigcn"]["f1_mean"],
            "bigcn_intact_per_seed": ia,
            "bigcn_rewired_per_seed": ib,
            "bigcn_intact_seed_std": round(float(statistics.pstdev(ia)), 4),
            "histgbt": a["results"]["histgbt"]["f1_mean"],
            "best_summary_intact": a["results"][a["best_summary"]]["f1_mean"],
            "seed_paired_rewired_vs_intact": paired_t(ia, ib),
            "summary_family_bit_identical": all(ctrl.values()),
        }
    # tree shape: how much arrangement exists to destroy
    res["tree_shape"] = {
        "upfd_gossipcop": shape_descriptors(G.load_upfd("gossipcop")["test"]),
        "twitter15": shape_descriptors(G.load_twitter("Twitter15")),
    }
    (REWIRED / "analysis.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["per_corpus"], indent=2))
    print(json.dumps(res["tree_shape"], indent=2))


if __name__ == "__main__":
    main()
