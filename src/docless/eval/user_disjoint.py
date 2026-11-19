"""Leakage audit: train/test user, URL, and media overlap.

We check whether a shared global graph or shared nodes across the
train/val/test splits could leak signal. Two facts answer this:

1. Architecturally, DocLessFake builds a *separate* hypergraph per split
   (`build_hypergraph(tr)`, `build_hypergraph(te)`), so no message passing
   ever crosses split boundaries, in finetuning or SSL pretraining.

2. Empirically, the node identities barely overlap. This script measures, for
   the standard stratified split, how many test stories share ANY user / URL /
   media node with the training set. On CoAID and ReCOVery this is ~zero, so
   the random split is already a user-disjoint split and the reported main-table
   numbers carry no user-identity leakage.

This is a measurement-only audit (no training), run over several seeds.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..train import DataConfig, _load_cascades, _stratified_split


def _audit_split(cascades, ratios, seed) -> dict:
    tr, va, te = _stratified_split(cascades, ratios, seed)
    tr_users = {e.user_id for c in tr for e in c.events}
    tr_urls = {e.url_id for c in tr for e in c.events}
    tr_media = {e.media_id for c in tr for e in c.events if e.media_id >= 0}

    def shares(c, pool, attr):
        if attr == "user":
            s = {e.user_id for e in c.events}
        elif attr == "url":
            s = {e.url_id for e in c.events}
        else:
            s = {e.media_id for e in c.events if e.media_id >= 0}
        return bool(s & pool)

    n_te = len(te)
    user_leak = sum(shares(c, tr_users, "user") for c in te)
    url_leak = sum(shares(c, tr_urls, "url") for c in te)
    media_leak = sum(shares(c, tr_media, "media") for c in te)
    return dict(seed=seed, n_test=n_te,
                user_leak=user_leak, url_leak=url_leak, media_leak=media_leak)


def run(datasets, seeds, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds_name, dcfg in datasets.items():
        for seed in seeds:
            cascades = _load_cascades(dcfg, seed)
            user_story = Counter()
            for c in cascades:
                for u in {e.user_id for e in c.events}:
                    user_story[u] += 1
            mult = np.array(list(user_story.values())) if user_story else np.array([0])
            a = _audit_split(cascades, dcfg.split_ratios, seed)
            a.update(dataset=ds_name,
                     n_users=int(len(mult)),
                     pct_multi_story_users=round(100 * float((mult > 1).mean()), 3),
                     max_stories_per_user=int(mult.max()))
            rows.append(a)
            print(f"  [{ds_name:>8s} seed={seed}] test={a['n_test']} | "
                  f"user-leak={a['user_leak']} url-leak={a['url_leak']} media-leak={a['media_leak']} | "
                  f"multi-story-users={a['pct_multi_story_users']}%")
    (out_dir / "user_disjoint.json").write_text(json.dumps({"rows": rows}, indent=2))
    # aggregate per dataset
    agg = {}
    for r in rows:
        agg.setdefault(r["dataset"], []).append(r)
    cells = []
    for d, rs in agg.items():
        cells.append(dict(
            dataset=d,
            mean_test=round(float(np.mean([r["n_test"] for r in rs])), 1),
            user_leak_total=int(sum(r["user_leak"] for r in rs)),
            url_leak_total=int(sum(r["url_leak"] for r in rs)),
            media_leak_total=int(sum(r["media_leak"] for r in rs)),
            test_total=int(sum(r["n_test"] for r in rs)),
            pct_multi_story_users=round(float(np.mean([r["pct_multi_story_users"] for r in rs])), 3),
            n_seeds=len(rs),
        ))
    (out_dir / "user_disjoint_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("outputs/r8/user_disjoint"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.seeds, args.out)
    print("\nPer-dataset leakage audit:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>8s}: user-leak {c['user_leak_total']}/{c['test_total']} test stories, "
              f"url-leak {c['url_leak_total']}, media-leak {c['media_leak_total']}, "
              f"multi-story users {c['pct_multi_story_users']}%")


if __name__ == "__main__":
    main()
