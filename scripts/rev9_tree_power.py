"""80%-power MDEs for the five retweet-tree corpora.

scripts/rev2_power.py covers only the four event-list corpora, so Twitter15/16,
UPFD-PolitiFact/GossipCop and Weibo had test-fold sizes but no stated
resolution.  No retraining is needed: rev6_graph_informative.py already stores
the story-clustered 95% and 90% bootstrap intervals for the selected
graph-vs-summary contrast, and the MDE convention of rev2_power.py is
MDE_80 = 2.80 * SE.

SE is recovered from each stored interval under the normal approximation
(SE = width / (2 z)); the 95% and 90% recoveries are reported side by side so
the approximation can be checked rather than assumed.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "r11" / "graph_informative_clustered"
OUT = ROOT / "outputs" / "r15" / "tree_power"
Z95, Z90, KMDE = 1.959964, 1.644854, 2.80

DATASETS = ["upfd_politifact", "upfd_gossipcop", "twitter15", "twitter16", "weibo"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for ds in DATASETS:
        d = json.loads((SRC / f"{ds}.json").read_text())
        cell = d["clustered_best_graph_vs_best_summary"]
        lo95, hi95 = cell["ci95"]
        lo90, hi90 = cell["ci90"]
        se95 = (hi95 - lo95) / (2 * Z95)
        se90 = (hi90 - lo90) / (2 * Z90)
        se = (se95 + se90) / 2
        rows[ds] = {
            "n_test_per_seed": d["n_test_per_seed"][0],
            "n_story_clusters": cell["n_stories"],
            "acc_delta": cell["mean"],
            "ci95": cell["ci95"], "ci90": cell["ci90"],
            "se_from_ci95": round(se95, 4), "se_from_ci90": round(se90, 4),
            "se_used": round(se, 4),
            "mde_80": round(KMDE * se, 3),
            "resolves_0.03": bool(KMDE * se <= 0.03),
            "best_graph": d["best_graph"], "best_summary": d["best_summary"],
        }
    meta = {
        "method": ("MDE_80 = 2.80 * SE, the convention of scripts/rev2_power.py "
                   "((z_.975 + z_.80) for a paired two-sided test at 80% power). "
                   "SE is recovered from the stored story-clustered bootstrap "
                   "intervals under the normal approximation and averaged over the "
                   "95% and 90% recoveries."),
        "source": "outputs/r11/graph_informative_clustered/<dataset>.json "
                  "(clustered_best_graph_vs_best_summary)",
        "selection_caveat": ("these intervals condition on test-fold selection of the "
                            "graph and summary family members, so the MDEs describe the "
                            "resolution of the fold, not a selection-adjusted procedure"),
        "per_corpus": rows,
    }
    (OUT / "tree_power.json").write_text(json.dumps(meta, indent=2))
    for ds, r in rows.items():
        print(f"{ds:18s} n_test={r['n_test_per_seed']:5d} clusters={r['n_story_clusters']:5d} "
              f"delta={r['acc_delta']:+.4f} SE={r['se_used']:.4f} MDE80={r['mde_80']:.3f} "
              f"resolves_0.03={r['resolves_0.03']}")


if __name__ == "__main__":
    main()
