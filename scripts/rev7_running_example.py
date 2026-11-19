"""Extract a REAL running example for main.tex §2 from FakeNewsNet/PolitiFact.

The example that shipped before this script was written could not exist in this
pipeline: it quoted n=312 (the loader caps cascades at 200 events) and two URLs
(the loader assigns exactly one url_id per story, from its news_url). This
script therefore (a) measures the population so the impossibility is on record,
and (b) picks one real cascade from that population and emits every number the
paper prints, so a reader can look the story up.

Run:  cd artifact && PYTHONPATH=src python3 scripts/rev7_running_example.py
Out:  artifact/outputs/r12/example/example.json
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np

from docless.baselines.propmlp import _features
from docless.causal.strata import _echo_chamber, _popularity
from docless.data.loaders import load_fakenewsnet
from docless.train import DataConfig, _load_cascades, _stratified_split

OUT = "outputs/r12/example"
ROOT = "data/raw/fakenewsnet"
CAP = 200


def population_audit() -> dict:
    """Record the facts that make the old vignette impossible."""
    audit = {}
    for src, root in (("politifact", ROOT), ("gossipcop", ROOT)):
        cs = load_fakenewsnet(root, source=src, max_events_per_story=CAP)
        n = [len(c.events) for c in cs]
        audit[src] = {
            "n_cascades": len(cs),
            "max_n": int(max(n)),
            "n_equals_312_exists": bool(any(x == 312 for x in n)),
            "distinct_url_counts": dict(Counter(c.num_urls() for c in cs)),
            "distinct_media_counts": dict(Counter(c.num_media() for c in cs)),
            "n_at_cap": int(sum(1 for x in n if x == CAP)),
        }
    return audit


def describe(c) -> dict:
    ts = sorted(e.timestamp for e in c.events)
    n = len(ts)
    spread = ts[-1] - ts[0]
    t10 = ts[min(9, n - 1)] - ts[0]
    frac_1h = float(sum(1 for t in ts if t - ts[0] <= 3600.0)) / n
    feats = _features([c])[0]
    return {
        "story_id": c.story_id,
        "label": int(c.label),
        "n_events": n,
        "truncated_at_cap": n == CAP,
        "distinct_users": c.num_users(),
        "distinct_urls": c.num_urls(),
        "distinct_media": c.num_media(),
        "popularity_P": int(_popularity(c)),
        "echo_E": round(float(_echo_chamber(c)), 4),
        "spread_seconds": round(float(spread), 1),
        "spread_days": round(float(spread) / 86400.0, 2),
        "time_to_10th_event_seconds": round(float(t10), 1),
        "time_to_10th_event_minutes": round(float(t10) / 60.0, 1),
        "frac_events_within_1h": round(frac_1h, 4),
        "c_s_7d": [round(float(x), 4) for x in feats],
        "c_s_7d_order": ["log1p_n", "log1p_u", "log1p_r", "log1p_m",
                         "log1p_P", "E", "log1p_spread_seconds"],
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    audit = population_audit()

    cs = load_fakenewsnet(ROOT, source="politifact", max_events_per_story=CAP)
    n = np.array([len(c.events) for c in cs])
    print(f"[pop] politifact cascades={len(cs)} max_n={n.max()} median_n={np.median(n):.0f}")

    # Pick from the real distribution, not to a target vector: the
    # misinformation cascade closest (Euclidean, z-scored) to the coordinate-wise
    # MEDIAN of the misinformation half over the four non-degenerate summary
    # features (log n, log P, E, log spread). log r, log m are constant by
    # construction in this release and log u is collinear with log n, so
    # including them would only add zeros to the distance. Ties break on
    # story_id, so the choice is deterministic.
    fake = [c for c in cs if c.label == 1]
    use = [0, 4, 5, 6]
    Z = _features(fake)[:, use]
    Zz = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    dist = np.linalg.norm(Zz - np.median(Zz, axis=0), axis=1)
    order = sorted(range(len(fake)), key=lambda i: (float(dist[i]), fake[i].story_id))
    chosen = fake[order[0]]
    ex = describe(chosen)
    ex["selection_rule"] = (
        "misinformation cascade nearest the coordinate-wise median of the "
        "misinformation half in z-scored (log n, log P, E, log spread) space; "
        "ties broken by story_id")
    ex["n_misinformation_cascades"] = len(fake)
    ex["median_of_misinformation_half"] = {
        k: round(float(v), 4) for k, v in zip(
            ["log1p_n", "log1p_P", "E", "log1p_spread_seconds"],
            np.median(Z, axis=0))}
    ex["runners_up"] = [fake[i].story_id for i in order[1:4]]
    print(f"[example] {ex['story_id']} n={ex['n_events']} P={ex['popularity_P']} "
          f"E={ex['echo_E']} spread={ex['spread_days']}d c_s={ex['c_s_7d']}")

    # Split membership + model predictions under the paper's PolitiFact config.
    dcfg = DataConfig(source="politifact", balanced=True, data_root=ROOT)
    membership = {}
    for seed in range(5):
        bal = _load_cascades(dcfg, seed)
        tr, va, te = _stratified_split(bal, dcfg.split_ratios, seed)
        where = None
        for name, fold in (("train", tr), ("val", va), ("test", te)):
            if any(c.story_id == chosen.story_id for c in fold):
                where = name
        membership[f"seed{seed}"] = where
    ex["split_membership_balanced_5seeds"] = membership

    payload = {
        "corpus": "FakeNewsNet PolitiFact (public CSV release, tweet ids only)",
        "loader": "docless.data.loaders.load_fakenewsnet",
        "max_events_per_story_cap": CAP,
        "population_audit": audit,
        "example": ex,
        "note": (
            "u == n by construction: the loader sets user_id = tweet id, so every "
            "event is its own event proxy. r == 1 by construction: one url_id per "
            "story. m == 0 by construction: media_id is always -1 in this release."),
    }
    path = os.path.join(OUT, "example.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()
