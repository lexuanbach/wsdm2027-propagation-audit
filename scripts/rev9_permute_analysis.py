"""Compare the profile-permutation arms against the intact retweet-tree run.

Mirrors scripts/rev8_edge_rewire_analysis.py: for each permutation variant and
corpus, report the intact and permuted BiGCN means, the paired seed-level
difference, and a two-sided seed-level t-test.  Also checks the invariance the
design promises -- under the 'within' variant the summary baselines must be
bit-identical to the intact run, because mean/std/max over a tree's profile
rows do not depend on their order.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy import stats as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
INTACT = ROOT / "outputs" / "r11" / "graph_informative_clustered"
PERM = ROOT / "outputs" / "r15" / "profile_permute"
OUT = ROOT / "outputs" / "r15" / "profile_permute"

SUMMARY_MODELS = ["logreg", "rf", "histgbt", "mlp"]
GRAPH_MODELS = ["gcn", "gatv2", "bigcn"]


def main():
    res = {"per_variant": {}}
    for variant in ("within", "global"):
        vdirs = [PERM / variant, ROOT / "outputs" / "r15" / "profile_permute_pf" / variant]
        vdirs = [d for d in vdirs if d.exists()]
        if not vdirs:
            continue
        cells = {}
        for f in sorted(f for d in vdirs for f in d.glob("*.json")):
            if f.name in ("summary.json", "permute_audit.json"):
                continue
            ds = f.stem
            pj = json.loads(f.read_text())
            ij = json.loads((INTACT / f"{ds}.json").read_text())
            cell = {"intact_best_graph": ij["best_graph"]}
            for m in GRAPH_MODELS:
                a = np.array(ij["results"][m]["per_seed"], dtype=float)
                b = np.array(pj["results"][m]["per_seed"], dtype=float)
                t = st.ttest_rel(b, a) if len(a) > 1 and np.std(b - a) > 0 else None
                cell[m] = {
                    "intact_mean": round(float(a.mean()), 4),
                    "permuted_mean": round(float(b.mean()), 4),
                    "delta": round(float(b.mean() - a.mean()), 4),
                    "intact_seed_std": round(float(a.std()), 4),
                    "p_seed_paired": round(float(t.pvalue), 4) if t is not None else None,
                    "per_seed_delta": [round(float(x), 4) for x in (b - a)],
                }
            # invariance audit on the summary side
            inv = {}
            for m in SUMMARY_MODELS:
                a = np.array(ij["results"][m]["per_seed"], dtype=float)
                b = np.array(pj["results"][m]["per_seed"], dtype=float)
                inv[m] = {"max_abs_diff": round(float(np.max(np.abs(b - a))), 6),
                          "identical": bool(np.allclose(a, b, atol=5e-4))}
            cell["summary_invariance"] = inv
            cell["summary_bit_identical"] = all(v["identical"] for v in inv.values())
            cells[ds] = cell
        res["per_variant"][variant] = cells

    res["design"] = {
        "within": ("non-root profile rows permuted inside each tree; topology, the tree's "
                   "profile multiset and the news root are preserved, so the summary "
                   "features are invariant by construction"),
        "global": ("non-root profile rows permuted across the whole corpus; topology and "
                   "the corpus-wide profile marginal survive, per-tree composition does not"),
        "applies_to": "graph-model node features only, on train, val and test alike",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(res, indent=2))

    for variant, cells in res["per_variant"].items():
        for ds, c in cells.items():
            bits = " ".join(
                f"{m}:{c[m]['intact_mean']}->{c[m]['permuted_mean']}({c[m]['delta']:+.4f},p={c[m]['p_seed_paired']})"
                for m in GRAPH_MODELS)
            print(f"[{variant}/{ds}] summary_bit_identical={c['summary_bit_identical']} {bits}")


if __name__ == "__main__":
    main()
