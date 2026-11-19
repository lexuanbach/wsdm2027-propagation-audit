"""Identity-vs-topology localization on the UPFD profile-tree corpora.

Companion to scripts/rev8_edge_rewire.py.  The rewire control destroys the
*attachment arrangement* while preserving the degree sequence; this control
does the complementary thing: it destroys the association between a user's
released 10-d profile vector and the position that user occupies, while
leaving every edge in place.

Two permutation variants, both applied to the graph models' node features only
(the summary baselines keep the intact features and are bit-identical across
runs, exactly as in the rewire control):

  within  -- permute the non-root profile rows *inside each tree*.  Topology,
             the tree's profile multiset, and the news root are all preserved;
             only which user sits at which position changes.  The summary
             features (mean/std/max over profile rows) are mathematically
             invariant under this permutation.
  global  -- permute the non-root profile rows across the *whole corpus*.  The
             corpus-wide profile marginal and every tree's topology survive;
             per-tree profile composition does not.  This is the stronger
             probe: a graph advantage that survives it cannot rest on released
             node attributes at all.

Node 0 of every UPFD graph is the news item, not a user, so it is held fixed.

Usage:
  PYTHONPATH=src python3 scripts/rev9_profile_permute.py \
      --variant within --datasets upfd_gossipcop upfd_politifact \
      [--selection test|val] [--seeds 0 1 2 3 4]
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location(
    "rev6_graph_informative", HERE / "rev6_graph_informative.py")
G = importlib.util.module_from_spec(spec)
sys.modules["rev6_graph_informative"] = G
spec.loader.exec_module(G)

DEFAULT_OUT = ROOT / "outputs" / "r15" / "profile_permute"

# item-id -> standardised feature matrix that the graph models should receive
PERMUTED: dict[str, np.ndarray] = {}
AUDIT: dict = {}


def build_permutation(dsname: str, variant: str, seed: int = 0) -> dict:
    """Replicate the script's own train-statistic standardisation, then build
    the permuted node-feature matrix for every item of the corpus."""
    off = G.load_upfd(dsname.split("_", 1)[1])
    tr = off["train"]
    all_tr = np.concatenate([it["x"] for it in tr], axis=0)
    nmu = all_tr.mean(0, keepdims=True)
    nsd = all_tr.std(0, keepdims=True) + 1e-6
    items = off["train"] + off["val"] + off["test"]
    std = {it["id"]: ((it["x"] - nmu) / nsd).astype(np.float32) for it in items}

    rng = np.random.default_rng(seed)
    n_nodes = sum(s.shape[0] for s in std.values())
    n_user = sum(max(s.shape[0] - 1, 0) for s in std.values())

    if variant == "within":
        for k, s in std.items():
            if s.shape[0] > 2:
                idx = rng.permutation(s.shape[0] - 1) + 1
                s[1:] = s[idx]
    elif variant == "global":
        # one corpus-wide permutation of every non-root row
        bank = np.concatenate([s[1:] for s in std.values() if s.shape[0] > 1], axis=0)
        order = rng.permutation(bank.shape[0])
        bank = bank[order]
        pos = 0
        for k, s in std.items():
            m = s.shape[0] - 1
            if m > 0:
                s[1:] = bank[pos: pos + m]
                pos += m
        assert pos == bank.shape[0], (pos, bank.shape)
    else:
        raise ValueError(variant)

    PERMUTED.update(std)
    return {"n_items": len(items), "n_nodes": int(n_nodes), "n_user_nodes": int(n_user),
            "n_singleton_trees": int(sum(1 for s in std.values() if s.shape[0] <= 1)),
            "n_trees_unpermutable_within": int(sum(1 for s in std.values() if s.shape[0] <= 2)),
            "standardisation": "train-node mean/std, replicated from rev6 run_dataset",
            "root_held_fixed": True, "rng_seed": seed}


_ORIG_MAKE_GRAPH = G.make_graph


def make_graph_permuted(item, node_feats):
    """Drop-in for rev6_graph_informative.make_graph."""
    key = item.get("id")
    rep = PERMUTED.get(key)
    if rep is None:                     # non-UPFD corpus: leave untouched
        return _ORIG_MAKE_GRAPH(item, node_feats)
    nf = np.asarray(node_feats)
    assert rep.shape == nf.shape, (key, rep.shape, nf.shape)
    # safety: the row we hold fixed must match what the script computed itself
    if not np.allclose(rep[0], nf[0], atol=1e-4):
        raise AssertionError(f"standardisation mismatch on {key}: root row differs")
    AUDIT["n_graphs_permuted"] = AUDIT.get("n_graphs_permuted", 0) + 1
    return _ORIG_MAKE_GRAPH(item, rep)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["within", "global"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["upfd_gossipcop", "upfd_politifact"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--selection", choices=["test", "val"], default="test")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    outdir = pathlib.Path(args.outdir) if args.outdir else DEFAULT_OUT / args.variant
    outdir.mkdir(parents=True, exist_ok=True)

    audits = {}
    for ds in args.datasets:
        if not ds.startswith("upfd_"):
            raise SystemExit("profile permutation needs released profile vectors (UPFD only)")
        audits[ds] = build_permutation(ds, args.variant)

    G.make_graph = make_graph_permuted          # the whole experiment, in one line
    sys.argv = [sys.argv[0], "--datasets", *args.datasets,
                "--seeds", *[str(s) for s in args.seeds],
                "--epochs", str(args.epochs),
                "--selection", args.selection,
                "--outdir", str(outdir)]
    G.main()

    (outdir / "permute_audit.json").write_text(json.dumps(
        {"variant": args.variant, "per_corpus": audits, "counters": AUDIT,
         "permuted_side": "graph model node features only; summary baselines and the "
                          "tree topology are unchanged",
         "note": ("under the 'within' variant the summary features (mean/std/max over "
                  "profile rows) are invariant by construction, so the summary columns "
                  "are bit-identical to the intact run")},
        indent=2))
    print("[profile-permute] audit:", json.dumps(AUDIT), flush=True)


if __name__ == "__main__":
    main()
