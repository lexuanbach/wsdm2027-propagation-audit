"""EDGE control: does the propagation GNN's advantage use tree *arrangement*,
or only the degree statistics the cascade summary already contains?

The one statistically significant graph win in the paper is BiGCN on
UPFD-GossipCop (+0.9 accuracy points over the tuned tree, story-clustered
p=0.011). This script re-runs that comparison with the graph model's edges
replaced by a *degree-preserving* random rewire of the same cascade tree.

Design (deliberately asymmetric, see the docstring of `rewire_parent`):

  * The undirected cascade tree is rewired with a connected double-edge swap
    (networkx `connected_double_edge_swap`), which preserves the node count,
    the edge count, connectivity and the *entire degree sequence*, and is
    therefore still a tree. It is then re-rooted at the original root and the
    parent pointers are re-derived by breadth-first search.
  * Preserved by construction: cascade size, root fan-out, leaf fraction and
    mean branching (all functions of the degree sequence alone).
  * Destroyed: depth, max width, mean depth, structural virality and
    depth-per-node -- i.e. everything that depends on *which* node a reshare
    attached to rather than on how many reshares each node received.
  * Only the `edge_index` consumed by the graph model is rewired. The summary
    baselines keep the intact tree, because the graph-informative summary
    vector is rebuilt from the tree itself; rewiring the shared item would move
    both sides of the comparison and measure nothing. Node features, labels,
    splits and seeds are held fixed.

Caveat recorded in the output JSON: on the corpora whose node features are
structural (Twitter15/16, Weibo), the per-node depth one-hot and log-out-degree
are computed on the *intact* tree and are therefore not rewired, so on those
corpora the control removes arrangement only from message passing, not from the
node inputs. On the UPFD corpora the node features are the 10-d user profile
vector, which is independent of topology, so the control there is complete.

Nothing under outputs/r11/graph_informative* is touched: this script writes to
its own --outdir (default outputs/r13/edge_rewire).

MUST run under a PyTorch-Geometric interpreter:
  PYTHONPATH=src python scripts/rev8_edge_rewire.py \
      --datasets upfd_gossipcop upfd_politifact twitter15 twitter16 \
      --seeds 0 1 2 3 4 --graph-models bigcn
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import zlib
from collections import deque

import networkx as nx
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "rev6_graph_informative", HERE / "rev6_graph_informative.py")
G = importlib.util.module_from_spec(_spec)
sys.modules["rev6_graph_informative"] = G
_spec.loader.exec_module(G)

DEFAULT_OUT = HERE.parent / "outputs" / "r13" / "edge_rewire"

# audit counters, written next to the results
AUDIT = {"n_items": 0, "n_rewired": 0, "n_skipped_small": 0, "n_skipped_swapfail": 0,
         "degree_sequence_preserved": 0, "degree_sequence_violated": 0}


def _to_undirected(parent: list[int]) -> tuple[nx.Graph, int]:
    n = len(parent)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    roots = []
    for c, p in enumerate(parent):
        if p is None or p < 0 or p >= n or p == c:
            roots.append(c)
        else:
            g.add_edge(p, c)
    return g, (roots[0] if roots else 0)


def rewire_parent(parent: list[int], seed: int) -> tuple[list[int], bool]:
    """Degree-preserving random rewire of a cascade tree.

    Returns (new_parent, rewired?). The tree is returned unchanged when it is
    too small to swap (fewer than 4 nodes or fewer than 2 edges) or when the
    swap routine cannot find a connectivity-preserving move.
    """
    n = len(parent)
    g, root = _to_undirected(parent)
    AUDIT["n_items"] += 1
    if n < 4 or g.number_of_edges() < 2 or not nx.is_connected(g):
        AUDIT["n_skipped_small"] += 1
        return list(parent), False
    deg_before = sorted(d for _, d in g.degree())
    try:
        nx.connected_double_edge_swap(g, nswap=max(1, g.number_of_edges()), seed=seed)
    except Exception:
        AUDIT["n_skipped_swapfail"] += 1
        return list(parent), False
    if sorted(d for _, d in g.degree()) == deg_before:
        AUDIT["degree_sequence_preserved"] += 1
    else:                                     # cannot happen; recorded, not silenced
        AUDIT["degree_sequence_violated"] += 1
    new_parent = [-1] * n
    seen = {root}
    q = deque([root])
    while q:
        u = q.popleft()
        for v in g.neighbors(u):
            if v not in seen:
                seen.add(v)
                new_parent[v] = u
                q.append(v)
    AUDIT["n_rewired"] += 1
    return new_parent, True


_CACHE: dict[str, list[int]] = {}
_ORIG_MAKE_GRAPH = G.make_graph


def make_graph_rewired(item, node_feats):
    """Drop-in for rev6_graph_informative.make_graph.

    The rewire is cached per story id, so every seed sees the *same* rewired
    topology and seeds vary only model initialisation -- exactly as in the
    intact run, where the tree is also fixed across seeds.
    """
    key = str(item["id"])
    if key not in _CACHE:
        _CACHE[key] = rewire_parent(item["parent"],
                                    seed=int(zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF))[0]
    shadow = {"parent": _CACHE[key], "y": item["y"], "id": item["id"]}
    return _ORIG_MAKE_GRAPH(shadow, node_feats)


def rewire_audit(items, limit=400):
    """Measure which summary statistics the rewire preserves and which it moves."""
    keys = ["n_nodes", "depth", "max_width", "mean_branching", "leaf_frac",
            "root_fanout", "mean_depth", "structural_virality", "depth_per_node"]
    before, after = {k: [] for k in keys}, {k: [] for k in keys}
    for it in items[:limit]:
        key = str(it["id"])
        if key not in _CACHE:
            _CACHE[key] = rewire_parent(it["parent"],
                                        seed=int(zlib.crc32(key.encode("utf-8")) & 0x7FFFFFFF))[0]
        sb, sa = G.tree_stats(it["parent"]), G.tree_stats(_CACHE[key])
        for k in keys:
            before[k].append(float(sb[k])); after[k].append(float(sa[k]))
    out = {}
    for k in keys:
        b, a = np.asarray(before[k]), np.asarray(after[k])
        out[k] = {"mean_intact": round(float(b.mean()), 4),
                  "mean_rewired": round(float(a.mean()), 4),
                  "frac_unchanged": round(float(np.mean(np.isclose(b, a))), 4)}
    out["_n_trees_audited"] = int(min(limit, len(items)))
    return out


def main():
    argv = sys.argv[1:]
    if not any(a == "--outdir" for a in argv):
        argv += ["--outdir", str(DEFAULT_OUT)]
    outdir = pathlib.Path(argv[argv.index("--outdir") + 1])
    outdir.mkdir(parents=True, exist_ok=True)

    # audit the rewire on the headline corpus before training anything
    audit_tables = {}
    if "upfd_gossipcop" in argv:
        off = G.load_upfd("gossipcop")
        audit_tables["upfd_gossipcop"] = rewire_audit(off["test"])
    if "twitter15" in argv:
        audit_tables["twitter15"] = rewire_audit(G.load_twitter("Twitter15"))

    G.make_graph = make_graph_rewired          # the whole experiment, in one line
    sys.argv = [sys.argv[0]] + argv
    G.main()

    (outdir / "rewire_audit.json").write_text(json.dumps(
        {"counters": AUDIT, "statistic_effect": audit_tables,
         "method": "networkx.connected_double_edge_swap on the undirected tree, "
                   "nswap = |E|, re-rooted at the original root, parents re-derived by BFS; "
                   "cached per story id so every seed sees the same rewired topology",
         "rewired_side": "graph model edge_index only; summary baselines and node features "
                         "stay on the intact tree"},
        indent=2))
    print("[edge-rewire] audit:", json.dumps(AUDIT), flush=True)


if __name__ == "__main__":
    main()
