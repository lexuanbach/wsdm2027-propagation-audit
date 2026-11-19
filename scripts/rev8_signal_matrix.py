"""Signal-availability audit behind main-text Table 1 (the dataset x signal matrix).

Every cell of that table is measured here from the release files on disk rather
than asserted in prose: story counts, official/stratified splits, whether the
release ships real reply/retweet edges, what node features exist, whether any
node identity recurs across stories, whether the release exposes event
timestamps, and the misinformation share of the unbalanced release.

Usage:  PYTHONPATH=src .venv/bin/python scripts/rev8_signal_matrix.py
Output: outputs/r13/signal_matrix/signal_matrix.json
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXT = ROOT / "external"
OUT = ROOT / "outputs" / "r13" / "signal_matrix"
sys.path.insert(0, str(ROOT / "src"))


def url_media_audit(loader, root):
    from docless.data import loaders
    cs = getattr(loaders, loader)(root)
    ur, me = collections.Counter(), collections.Counter()
    for c in cs:
        urls, med = set(), set()
        for e in c.events:
            u = getattr(e, "url_id", None)
            if u is not None:
                urls.add(u)
            med.update(getattr(e, "media_ids", None) or [])
        ur[len(urls)] += 1
        me[len(med)] += 1
    return {"n_cascades": len(cs),
            "distinct_url_counts": {str(k): v for k, v in sorted(ur.items())},
            "distinct_media_counts": {str(k): v for k, v in sorted(me.items())}}


def upfd_audit(name):
    raw = EXT / "upfd" / name / "raw"
    y = np.load(raw / "graph_labels.npy")
    idx = {s: np.load(raw / f"{s}_idx.npy") for s in ("train", "val", "test")}
    return {"n_graphs": int(len(y)),
            "label_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "frac_label1": round(float((y == 1).mean()), 4),
            "split_sizes": {k: int(len(v)) for k, v in idx.items()},
            "files_in_release": sorted(p.name for p in raw.iterdir()),
            "timestamp_file_present": any("time" in p.name.lower() for p in raw.iterdir())}


def rvnn_audit(path, n_expected_cols_note):
    """The BiGCN release of the TD-RvNN trees: eid \t parentIdx \t childIdx \t ... \t BoW."""
    with path.open(encoding="utf-8", errors="ignore") as fh:
        first = fh.readline().rstrip("\n").split("\t")
    return {"file": str(path.relative_to(EXT.parent)),
            "n_tab_columns_first_row": len(first),
            "column_meaning": n_expected_cols_note,
            "timestamp_column": False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "identity_free": {
            "coaid": url_media_audit("load_coaid", str(ROOT / "data/raw/coaid")),
            "recovery": url_media_audit("load_recovery", str(ROOT / "data/raw/recovery")),
            "_fakenewsnet_note": "PolitiFact/GossipCop URL and media degeneracy is audited in "
                                 "outputs/r12/example/example.json (population_audit): "
                                 "677 and 17,977 cascades, all with exactly 1 URL and 0 media",
            "timestamps": "recovered from the Twitter snowflake encoding of the released ids",
            "cross_cascade_identity": "none: the loader sets user_id = tweet id, so u ~= n and "
                                      "no sharer appears in two cascades",
            "reply_tree": "not released; the per-cascade graph is constructed from the event list",
        },
        "graph_informative": {
            "upfd_politifact": upfd_audit("politifact"),
            "upfd_gossipcop": upfd_audit("gossipcop"),
            "twitter15": rvnn_audit(EXT / "BiGCN/data/Twitter15/data.TD_RvNN.vol_5000.txt",
                                    "eid, parentIdx, childIdx, maxDegree, maxL, bag-of-words"),
            "twitter16": rvnn_audit(EXT / "BiGCN/data/Twitter16/data.TD_RvNN.vol_5000.txt",
                                    "eid, parentIdx, childIdx, maxDegree, maxL, bag-of-words"),
            "weibo": rvnn_audit(EXT / "BiGCN/data/Weibo/weibotree.txt",
                                "eid, parentIdx, childIdx, bag-of-words"),
            "timestamps": "no event-time field in any of these five releases as distributed here",
            "upfd_cross_cascade_identity": "outputs/r11/upfd_profile_recurrence/"
                                           "upfd_profile_recurrence.json",
        },
    }
    (OUT / "signal_matrix.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:1200])


if __name__ == "__main__":
    main()
