"""Graph-informative regime audit: does a propagation GNN
beat cascade-summary baselines once the data carry *real* retweet/reply trees
and user-level signal?

Datasets (all public, text-free inputs only):
  * UPFD PolitiFact / GossipCop (Dou et al. 2021; PyTorch Geometric release):
    real retweet trees with per-user *profile* features (10-dim: account age,
    follower/friend/status counts, verification, ...), fixed official
    train/val/test splits. This is the user-identity regime the public
    FakeNewsNet tweet-id release cannot express.
  * Twitter15 / Twitter16 (Ma et al. 2017; BiGCN release of the TD-RvNN
    trees): real reply/retweet trees, 4-class rumor labels
    (non-rumor/false/true/unverified). We discard the bag-of-words node text
    and keep structure only.
  * Weibo (Ma et al. 2016; BiGCN release): real repost trees, binary rumor
    labels, Chinese platform. Structure only.

Models (identical splits and seeds):
  summary family : LogReg, RandomForest, val-tuned HistGBT, 2-layer MLP on
                   structural cascade summaries (size, depth, width, branching,
                   leaf fraction, root fan-out, mean/max depth, structural
                   virality) + [UPFD] mean/std/max of the 10 profile features
  graph family   : GCN (2-layer, mean pool), GATv2 (2-layer), BiGCN-structure
                   (top-down + bottom-up GCN with root enhancement, as in
                   Bian et al. 2020) on the same trees with text-free node
                   features ([UPFD] the profile vector; otherwise depth one-hot,
                   log out-degree, is-root)

Outputs: outputs/r11/graph_informative/<dataset>.json and summary.json with
per-seed macro-F1, paired story-level bootstrap (best graph vs HistGBT),
and TOST equivalence at margins 0.03/0.05 on paired accuracy.
CPU only.  Usage: PYTHONPATH=src python3 scripts/rev6_graph_informative.py
  [--datasets upfd_politifact upfd_gossipcop twitter15 twitter16 weibo]
  [--seeds 0 1 2 3 4] [--epochs 60]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from docless.eval.stats import macro_f1, paired_bootstrap  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXT = ROOT / "external"
OUT = ROOT / "outputs" / "r11" / "graph_informative"
# graph models to train; a reduced list is used when only the clustered delta
# for the already-identified best graph model needs recomputing.
GRAPH_MODELS = ["gcn", "gatv2", "bigcn"]
# Family-member selection rule. "test" reproduces the published (exploratory)
# protocol in which best_graph/best_summary are the argmax of mean TEST
# macro-F1. "val" is the validation-clean arm: the argmax is taken over mean
# VALIDATION macro-F1 and no model is fitted on the validation fold.
SELECTION = "test"
GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 31) for l2 in (0.0, 1.0)]
MAX_DEPTH_ONEHOT = 8


# --------------------------------------------------------------------------- #
# tree utilities
# --------------------------------------------------------------------------- #
def tree_stats(parent: list[int]) -> dict:
    """parent[i] = index of parent or -1 for the root(s)."""
    n = len(parent)
    children = defaultdict(list)
    roots = []
    for i, p in enumerate(parent):
        if p < 0 or p >= n or p == i:
            roots.append(i)
        else:
            children[p].append(i)
    depth = [0] * n
    order = []
    stack = list(roots)
    seen = set(roots)
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            if v not in seen:
                seen.add(v)
                depth[v] = depth[u] + 1
                stack.append(v)
    for i in range(n):               # nodes unreachable from a root (corrupt lines)
        if i not in seen:
            depth[i] = 1
    outdeg = [len(children[i]) for i in range(n)]
    nonleaf = [d for d in outdeg if d > 0]
    width = defaultdict(int)
    for d in depth:
        width[d] += 1
    max_depth = max(depth) if depth else 0
    # structural virality (mean pairwise distance) via depth arithmetic on a tree:
    # exact O(n^2) for small trees, sampled for big ones
    sv = 0.0
    if n > 1:
        idx = list(range(n))
        if n > 400:
            rng = random.Random(0)
            idx = rng.sample(idx, 400)
        # ancestors for LCA: climb parents
        anc = {}
        for i in idx:
            path, u = [], i
            while u >= 0 and u < n and u not in path:
                path.append(u)
                u = parent[u]
            anc[i] = {v: k for k, v in enumerate(path)}
        tot, cnt = 0.0, 0
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1:]:
                best = None
                for v, ki in anc[i].items():
                    if v in anc[j]:
                        best = ki + anc[j][v]
                        break
                if best is not None:
                    tot += best
                    cnt += 1
        sv = tot / cnt if cnt else 0.0
    return {
        "n_nodes": n, "depth": max_depth, "max_width": max(width.values()) if width else 0,
        "mean_branching": float(np.mean(nonleaf)) if nonleaf else 0.0,
        "leaf_frac": float(sum(1 for d in outdeg if d == 0) / n) if n else 0.0,
        "root_fanout": float(np.mean([outdeg[r] for r in roots])) if roots else 0.0,
        "mean_depth": float(np.mean(depth)) if depth else 0.0,
        "structural_virality": sv,
        "depth_per_node": float(max_depth / n) if n else 0.0,
        "_depth": depth, "_outdeg": outdeg, "_roots": roots,
    }


def summary_vector(stats: dict, extra: np.ndarray | None = None) -> np.ndarray:
    v = [np.log1p(stats["n_nodes"]), stats["depth"], np.log1p(stats["max_width"]),
         stats["mean_branching"], stats["leaf_frac"], np.log1p(stats["root_fanout"]),
         stats["mean_depth"], stats["structural_virality"], stats["depth_per_node"]]
    if extra is not None:
        v.extend(extra.tolist())
    return np.asarray(v, dtype=np.float64)


def structural_node_features(stats: dict) -> np.ndarray:
    n = stats["n_nodes"]
    x = np.zeros((n, MAX_DEPTH_ONEHOT + 3), dtype=np.float32)
    for i in range(n):
        d = min(stats["_depth"][i], MAX_DEPTH_ONEHOT - 1)
        x[i, d] = 1.0
        x[i, MAX_DEPTH_ONEHOT] = np.log1p(stats["_outdeg"][i])
        x[i, MAX_DEPTH_ONEHOT + 1] = 1.0 if i in stats["_roots"] else 0.0
        x[i, MAX_DEPTH_ONEHOT + 2] = np.log1p(n)
    return x


# --------------------------------------------------------------------------- #
# dataset loaders -> list of dict(x: [n,f] float32, parent: list[int], y: int)
# --------------------------------------------------------------------------- #
def load_rvnn(label_path: pathlib.Path, tree_path: pathlib.Path, label_map: dict, label_col: int,
              id_col: int):
    labels = {}
    for line in label_path.open(encoding="utf-8", errors="ignore"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) <= max(label_col, id_col):
            continue
        lab = parts[label_col].strip()
        if lab in label_map:
            labels[parts[id_col].strip()] = label_map[lab]
    trees = defaultdict(dict)
    for line in tree_path.open(encoding="utf-8", errors="ignore"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            continue
        eid, ip, ic = parts[0], parts[1], parts[2]
        if eid not in labels:
            continue
        try:
            ic_i = int(ic)
        except ValueError:
            continue
        trees[eid][ic_i] = -1 if ip == "None" else int(ip)
    items = []
    for eid, nodes in trees.items():
        n = max(nodes) if nodes else 0
        if n < 2:
            continue
        parent = [-1] * n
        for ic_i, ip_i in nodes.items():      # indices are 1-based in the RvNN files
            if 1 <= ic_i <= n:
                parent[ic_i - 1] = (ip_i - 1) if ip_i >= 1 else -1
        items.append({"id": eid, "parent": parent, "y": labels[eid]})
    return items


def load_weibo():
    label_path = EXT / "BiGCN" / "data" / "Weibo" / "weibo_id_label.txt"
    tree_path = EXT / "BiGCN" / "data" / "Weibo" / "weibotree.txt"
    labels = {}
    for line in label_path.open():
        parts = line.split()
        if len(parts) >= 2:
            labels[parts[0]] = int(parts[1])
    trees = defaultdict(dict)
    for line in tree_path.open(encoding="utf-8", errors="ignore"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3 or parts[0] not in labels:
            continue
        try:
            ic_i = int(parts[2])
        except ValueError:
            continue
        trees[parts[0]][ic_i] = -1 if parts[1] == "None" else int(parts[1])
    items = []
    for eid, nodes in trees.items():
        n = max(nodes) if nodes else 0
        if n < 2:
            continue
        parent = [-1] * n
        for ic_i, ip_i in nodes.items():
            if 1 <= ic_i <= n:
                parent[ic_i - 1] = (ip_i - 1) if ip_i >= 1 else -1
        items.append({"id": eid, "parent": parent, "y": labels[eid]})
    return items


def load_twitter(name: str):
    base = EXT / "BiGCN" / "data" / name
    label_map = {"non-rumor": 0, "false": 1, "true": 2, "unverified": 3}
    return load_rvnn(base / f"{name}_label_All.txt", base / "data.TD_RvNN.vol_5000.txt",
                     label_map, label_col=0, id_col=2)


def load_upfd(name: str):
    from torch_geometric.datasets import UPFD
    root = str(EXT / "upfd")
    out = {}
    for split in ["train", "val", "test"]:
        ds = UPFD(root, name, "profile", split)
        items = []
        for k in range(len(ds)):
            d = ds[k]
            n = int(d.num_nodes)
            parent = [-1] * n
            ei = d.edge_index.numpy()
            # UPFD edges point from source (earlier) to target; node 0 is the news root
            for s, t in zip(ei[0], ei[1]):
                if t != 0 and parent[t] == -1 and s != t:
                    parent[t] = int(s)
            items.append({"id": f"{name}-{split}-{k}", "parent": parent,
                          "x": d.x.numpy().astype(np.float32), "y": int(d.y.item())})
        out[split] = items
    return out


# --------------------------------------------------------------------------- #
# graph models (PyG)
# --------------------------------------------------------------------------- #
def make_graph(item, node_feats):
    from torch_geometric.data import Data
    parent = item["parent"]
    src, dst = [], []
    for c, p in enumerate(parent):
        if p >= 0:
            src.append(p); dst.append(c)
    if not src:                         # single node or disconnected: self loop
        src, dst = [0], [0]
    ei = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=torch.tensor(node_feats, dtype=torch.float32), edge_index=ei,
                y=torch.tensor([item["y"]], dtype=torch.long))


class GCNNet(nn.Module):
    def __init__(self, in_dim, hid, n_cls, kind="gcn"):
        super().__init__()
        from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
        self.pool = global_mean_pool
        if kind == "gcn":
            self.c1, self.c2 = GCNConv(in_dim, hid), GCNConv(hid, hid)
        else:
            self.c1 = GATv2Conv(in_dim, hid // 4, heads=4)
            self.c2 = GATv2Conv(hid, hid // 4, heads=4)
        # UPFD-style readout: pooled node embedding concatenated with the root
        # (source) node's embedding (Dou et al. 2021)
        self.lin = nn.Linear(2 * hid, n_cls)
        self.drop = nn.Dropout(0.2)

    def forward(self, data):
        x = F.relu(self.c1(data.x, data.edge_index))
        x = self.drop(x)
        x = F.relu(self.c2(x, data.edge_index))
        root = x[data.ptr[:-1]]
        return self.lin(torch.cat([self.pool(x, data.batch), root], dim=1))


class BiGCNStruct(nn.Module):
    """Bian et al. (2020): top-down and bottom-up GCNs with root-feature
    enhancement, concatenated; node features are text-free here."""

    def __init__(self, in_dim, hid, n_cls):
        super().__init__()
        from torch_geometric.nn import GCNConv, global_mean_pool
        self.pool = global_mean_pool
        self.td1, self.td2 = GCNConv(in_dim, hid), GCNConv(hid + in_dim, hid)
        self.bu1, self.bu2 = GCNConv(in_dim, hid), GCNConv(hid + in_dim, hid)
        self.lin = nn.Linear(2 * (hid + in_dim), n_cls)
        self.drop = nn.Dropout(0.2)

    def _branch(self, c1, c2, x, ei, batch, root_x):
        h = F.relu(c1(x, ei))
        h = torch.cat([h, root_x], dim=1)
        h = self.drop(h)
        h = F.relu(c2(h, ei))
        h = torch.cat([h, root_x], dim=1)
        return self.pool(h, batch)

    def forward(self, data):
        x, ei, batch = data.x, data.edge_index, data.batch
        # root (first node of each graph) feature broadcast
        ptr = data.ptr
        root_idx = ptr[:-1]
        root_x = x[root_idx][batch]
        td = self._branch(self.td1, self.td2, x, ei, batch, root_x)
        bu = self._branch(self.bu1, self.bu2, x, ei.flip(0), batch, root_x)
        return self.lin(torch.cat([td, bu], dim=1))


def train_graph(kind, tr, va, te, n_cls, seed, epochs, hid=64, lr=None, batch=64):
    """Validation-selected learning rate in {1e-3, 5e-3}; early stopping on val F1."""
    if lr is None:
        best_lr, best_f1, best_pred = None, -1.0, None
        for cand in (1e-3, 5e-3):
            pred, vf1 = _train_graph_once(kind, tr, va, te, n_cls, seed, epochs, hid, cand, batch)
            if vf1 > best_f1:
                best_lr, best_f1, best_pred = cand, vf1, pred
        return best_pred, best_f1
    return _train_graph_once(kind, tr, va, te, n_cls, seed, epochs, hid, lr, batch)


def _train_graph_once(kind, tr, va, te, n_cls, seed, epochs, hid, lr, batch):
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    in_dim = tr[0].x.shape[1]
    model = BiGCNStruct(in_dim, hid, n_cls) if kind == "bigcn" else GCNNet(in_dim, hid, n_cls, kind)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    ltr = DataLoader(tr, batch_size=batch, shuffle=True)
    lva = DataLoader(va, batch_size=256)
    lte = DataLoader(te, batch_size=256)

    def predict(loader):
        model.eval(); out = []
        with torch.no_grad():
            for b in loader:
                out.append(model(b).argmax(-1))
        return torch.cat(out).numpy()

    best, best_state, patience = -1.0, None, 0
    y_va = np.array([d.y.item() for d in va])
    for ep in range(epochs):
        model.train()
        for b in ltr:
            opt.zero_grad()
            F.cross_entropy(model(b), b.y).backward()
            opt.step()
        f1 = macro_f1(predict(lva), y_va, n_cls)
        if f1 > best:
            best, patience = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return predict(lte), best


# --------------------------------------------------------------------------- #
# tabular / MLP
# --------------------------------------------------------------------------- #
def tuned_histgbt(x_tr, y_tr, x_va, y_va, n_cls):
    best, best_f1 = None, -1.0
    for hp in GRID:
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp).fit(x_tr, y_tr)
        f1 = macro_f1(m.predict(x_va), y_va, n_cls)
        if f1 > best_f1:
            best_f1, best = f1, hp
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **best)
    m.fit(np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))
    return m, best


def innercv_histgbt(x_tr, y_tr, n_cls, folds=5):
    """Validation-clean HistGBT: the same GRID is scored by stratified CV
    *inside the training fold*, then a single model is fitted on train only.
    The validation fold is never touched, so its score can be used to select."""
    from sklearn.model_selection import StratifiedKFold
    k = min(folds, int(np.min(np.bincount(y_tr))))
    best, best_f1 = GRID[0], -1.0
    if k >= 2:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        for hp in GRID:
            sc = []
            for itr, iva in skf.split(x_tr, y_tr):
                m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp)
                m.fit(x_tr[itr], y_tr[itr])
                sc.append(macro_f1(m.predict(x_tr[iva]), y_tr[iva], n_cls))
            f1 = float(np.mean(sc))
            if f1 > best_f1:
                best_f1, best = f1, hp
    m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **best).fit(x_tr, y_tr)
    return m, best


def train_mlp(x_tr, y_tr, x_te, n_cls, seed, hidden=64, epochs=100, lr=1e-2):
    torch.manual_seed(seed)
    xt = torch.tensor(x_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.long)
    xe = torch.tensor(x_te, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(xt.shape[1], hidden), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, n_cls))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(xt), yt).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(xe).argmax(-1).numpy()


def stratified_split(items, seed, ratios=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in items:
        by[it["y"]].append(it)
    tr, va, te = [], [], []
    for y, group in by.items():
        rng.shuffle(group)
        n = len(group); a = int(ratios[0] * n); b = int((ratios[0] + ratios[1]) * n)
        tr += group[:a]; va += group[a:b]; te += group[b:]
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te


def tost(diff, margin, alpha=0.05, n_boot=10000, seed=0):
    """Paired TOST on per-story accuracy differences via percentile bootstrap."""
    d = np.asarray(diff, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha, 1 - alpha])        # 90% CI <=> two one-sided at 5%
    return {"margin": margin, "ci90": [float(lo), float(hi)], "equivalent": bool(lo > -margin and hi < margin),
            "mean": float(d.mean())}


# --------------------------------------------------------------------------- #
def run_dataset(name, seeds, epochs, n_cls, items=None, official=None, profile=False):
    rows, preds_store, labels_store = {}, defaultdict(list), []
    vrows = {}          # per-seed VALIDATION macro-F1, used when SELECTION == "val"
    story_store = []  # per-seed array of story ids (stable across seeds)
    story_index = {}
    timing = {}
    hp_log = []
    for seed in seeds:
        if official is not None:
            tr_i, va_i, te_i = official["train"], official["val"], official["test"]
        else:
            tr_i, va_i, te_i = stratified_split(items, seed)
        stats = {id(it): tree_stats(it["parent"]) for it in tr_i + va_i + te_i}

        def feat(its):
            X = []
            for it in its:
                st = stats[id(it)]
                extra = None
                if profile:
                    px = it["x"]
                    extra = np.concatenate([px.mean(0), px.std(0), px.max(0)])
                X.append(summary_vector(st, extra))
            return np.asarray(X)

        x_tr, x_va, x_te = feat(tr_i), feat(va_i), feat(te_i)
        y_tr = np.array([it["y"] for it in tr_i]); y_va = np.array([it["y"] for it in va_i])
        y_te = np.array([it["y"] for it in te_i])
        sc = StandardScaler().fit(x_tr)
        xs_tr, xs_va, xs_te = sc.transform(x_tr), sc.transform(x_va), sc.transform(x_te)
        labels_store.append(y_te)
        story_store.append(np.array([story_index.setdefault(id(it), len(story_index)) for it in te_i]))
        t0 = time.time()
        preds, vpreds = {}, {}
        _lr = LogisticRegression(max_iter=2000, random_state=seed).fit(xs_tr, y_tr)
        preds["logreg"], vpreds["logreg"] = _lr.predict(xs_te), _lr.predict(xs_va)
        _rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=1).fit(xs_tr, y_tr)
        preds["rf"], vpreds["rf"] = _rf.predict(xs_te), _rf.predict(xs_va)
        if SELECTION == "val":
            hg, hp = innercv_histgbt(xs_tr, y_tr, n_cls)
        else:
            hg, hp = tuned_histgbt(xs_tr, y_tr, xs_va, y_va, n_cls)
        hp_log.append(hp)
        preds["histgbt"], vpreds["histgbt"] = hg.predict(xs_te), hg.predict(xs_va)
        # one MLP fit; validation and test rows are scored from the same model
        _mlp_out = train_mlp(xs_tr, y_tr, np.vstack([xs_te, xs_va]), n_cls, seed)
        preds["mlp"], vpreds["mlp"] = _mlp_out[: len(xs_te)], _mlp_out[len(xs_te):]
        timing.setdefault("summary_family_s", []).append(time.time() - t0)
        # graph models; node features are z-scored with training-node statistics
        # (the summary models receive the same standardisation)
        if profile:
            all_tr = np.concatenate([it["x"] for it in tr_i], axis=0)
            nmu, nsd = all_tr.mean(0, keepdims=True), all_tr.std(0, keepdims=True) + 1e-6
        else:
            nmu, nsd = None, None

        def graphs(its):
            out = []
            for it in its:
                st = stats[id(it)]
                nf = ((it["x"] - nmu) / nsd).astype(np.float32) if profile else structural_node_features(st)
                out.append(make_graph(it, nf))
            return out
        g_tr, g_va, g_te = graphs(tr_i), graphs(va_i), graphs(te_i)
        for kind in GRAPH_MODELS:
            t1 = time.time()
            preds[kind], _vf1 = train_graph(kind, g_tr, g_va, g_te, n_cls, seed, epochs)
            vrows.setdefault(kind, []).append(float(_vf1))
            timing.setdefault(f"{kind}_s", []).append(time.time() - t1)
        for k, p in vpreds.items():
            vrows.setdefault(k, []).append(macro_f1(p, y_va, n_cls))
        for k, p in preds.items():
            rows.setdefault(k, []).append(macro_f1(p, y_te, n_cls))
            preds_store[k].append(p)
        print(f"[{name}] seed {seed}: " + " ".join(f"{k}={rows[k][-1]:.3f}" for k in rows), flush=True)
    # aggregate + paired tests (pooled over seeds, story-level 0/1 loss)
    labels_all = np.concatenate(labels_store)
    losses = {k: np.concatenate([(p != y).astype(float) for p, y in zip(preds_store[k], labels_store)])
              for k in preds_store}
    agg = {k: {"f1_mean": round(float(np.mean(v)), 4), "f1_std": round(float(np.std(v)), 4),
               "per_seed": [round(float(x), 4) for x in v]} for k, v in rows.items()}
    for k, v in vrows.items():
        if k in agg:
            agg[k]["val_f1_mean"] = round(float(np.mean(v)), 4)
            agg[k]["val_f1_std"] = round(float(np.std(v)), 4)
            agg[k]["val_per_seed"] = [round(float(x), 4) for x in v]
    graph_kinds = list(GRAPH_MODELS)
    summary_kinds = ["logreg", "rf", "histgbt", "mlp"]
    _key = "val_f1_mean" if SELECTION == "val" else "f1_mean"
    # deterministic tiebreak: higher score, then lower std, then declared order
    def _pick(cands):
        return min(cands, key=lambda k: (-agg[k][_key],
                                         agg[k]["val_f1_std" if SELECTION == "val" else "f1_std"],
                                         cands.index(k)))
    best_graph = _pick(graph_kinds)
    best_summary = _pick(summary_kinds)
    pb = paired_bootstrap(losses[best_graph], losses[best_summary])
    pb_hist = paired_bootstrap(losses[best_graph], losses["histgbt"])
    diff = losses["histgbt"] - losses[best_graph]          # acc(graph) - acc(histgbt)
    # story-clustered bootstrap: resample stories, every seed copy of a story moves together
    sid = np.concatenate(story_store)
    def clustered(d, n_boot=10000, seed=0):
        rng = np.random.default_rng(seed)
        uniq = np.unique(sid)
        groups = [np.where(sid == u)[0] for u in uniq]
        sums = np.array([d[g].sum() for g in groups]); cnts = np.array([len(g) for g in groups])
        means = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(groups), len(groups))
            means.append(sums[pick].sum() / cnts[pick].sum())
        means = np.array(means)
        m = float(d.mean()); lo, hi = np.percentile(means, [2.5, 97.5]); lo90, hi90 = np.percentile(means, [5, 95])
        p = float(2 * min((means <= 0).mean(), (means >= 0).mean()))
        return {"mean": round(m, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "p": round(min(1.0, p), 4),
                "ci90": [round(float(lo90), 4), round(float(hi90), 4)],
                "tost_equivalent_0.03": bool(lo90 > -0.03 and hi90 < 0.03),
                "tost_equivalent_0.05": bool(lo90 > -0.05 and hi90 < 0.05),
                "n_stories": int(len(groups)), "n_rows": int(len(d))}
    diff_summary = losses[best_summary] - losses[best_graph]
    # seed-level paired t-test on per-seed accuracy differences
    from scipy import stats as _st
    acc_graph = [1 - float(l.mean()) for l in [(p != y).astype(float) for p, y in zip(preds_store[best_graph], labels_store)]]
    acc_tree = [1 - float(l.mean()) for l in [(p != y).astype(float) for p, y in zip(preds_store["histgbt"], labels_store)]]
    tt = _st.ttest_rel(acc_graph, acc_tree) if len(acc_graph) > 1 else None
    return {
        "clustered_best_graph_vs_histgbt": clustered(diff),
        "clustered_best_graph_vs_best_summary": clustered(diff_summary),
        "seed_paired_t_graph_vs_histgbt": {"t": round(float(tt.statistic), 3), "p": round(float(tt.pvalue), 4),
                                           "per_seed_acc_graph": [round(a, 4) for a in acc_graph],
                                           "per_seed_acc_histgbt": [round(a, 4) for a in acc_tree]} if tt is not None else None,
        "dataset": name, "n_classes": n_cls, "seeds": seeds, "epochs": epochs,
        "n_test_per_seed": [int(len(y)) for y in labels_store],
        "official_splits": official is not None, "profile_features": profile,
        "summary_features": ["log_size", "depth", "log_max_width", "mean_branching", "leaf_frac",
                             "log_root_fanout", "mean_depth", "structural_virality", "depth_per_node"]
        + (["profile_mean(10)", "profile_std(10)", "profile_max(10)"] if profile else []),
        "node_features": "UPFD 10-d user profile" if profile else "depth one-hot(8), log out-degree, is-root, log size",
        "results": agg,
        "best_graph": best_graph, "best_summary": best_summary,
        "selection_rule": SELECTION,
        "selection_metric": ("mean validation macro-F1 over seeds; no model is fitted on the "
                             "validation fold (HistGBT uses stratified CV inside train)"
                             if SELECTION == "val" else
                             "mean TEST macro-F1 over seeds (exploratory; as published)"),
        "test_selected_best_graph": max(graph_kinds, key=lambda k: agg[k]["f1_mean"]),
        "test_selected_best_summary": max(summary_kinds, key=lambda k: agg[k]["f1_mean"]),
        "story_predictions": {k: [[int(x) for x in a] for a in v] for k, v in preds_store.items()},
        "story_ids_per_seed": [[int(x) for x in a] for a in story_store],
        "story_labels_per_seed": [[int(x) for x in a] for a in labels_store],
        "paired_best_graph_vs_best_summary": {"acc_gain_graph": round(pb.mean_diff, 4), "p": round(pb.p_value, 4),
                                             "ci95": [round(pb.ci_lo, 4), round(pb.ci_hi, 4)]},
        "paired_best_graph_vs_histgbt": {"acc_gain_graph": round(pb_hist.mean_diff, 4), "p": round(pb_hist.p_value, 4),
                                        "ci95": [round(pb_hist.ci_lo, 4), round(pb_hist.ci_hi, 4)]},
        "tost_graph_vs_histgbt": {str(m): tost(diff, m) for m in (0.03, 0.05)},
        "histgbt_selected_hparams": hp_log,
        "timing_s_mean": {k: round(float(np.mean(v)), 1) for k, v in timing.items()},
    }


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["upfd_politifact", "upfd_gossipcop", "twitter15", "twitter16", "weibo"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max-items", type=int, default=None, help="debug cap per dataset")
    ap.add_argument("--outdir", type=str, default=None, help="override output directory")
    ap.add_argument("--graph-models", nargs="+", default=None, help="subset of gcn/gatv2/bigcn to train")
    ap.add_argument("--selection", choices=["test", "val"], default="test",
                    help="family-member selection rule: 'test' reproduces the published "
                         "exploratory protocol, 'val' is the validation-clean arm")
    args = ap.parse_args()
    global GRAPH_MODELS, SELECTION
    SELECTION = args.selection
    if args.graph_models:
        GRAPH_MODELS = list(args.graph_models)
    if args.outdir:
        OUT = pathlib.Path(args.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in args.datasets:
        t0 = time.time()
        if name.startswith("upfd_"):
            off = load_upfd(name.split("_", 1)[1])
            if args.max_items:
                off = {k: v[: args.max_items] for k, v in off.items()}
            res = run_dataset(name, args.seeds, args.epochs, 2, official=off, profile=True)
            res["n_graphs"] = {k: len(v) for k, v in off.items()}
        else:
            if name == "weibo":
                items, n_cls = load_weibo(), 2
            else:
                items, n_cls = load_twitter("Twitter15" if name == "twitter15" else "Twitter16"), 4
            if args.max_items:
                items = items[: args.max_items]
            res = run_dataset(name, args.seeds, args.epochs, n_cls, items=items)
            res["n_graphs"] = len(items)
            res["label_counts"] = {int(k): int(v) for k, v in zip(*np.unique([it["y"] for it in items], return_counts=True))}
        res["wall_s"] = round(time.time() - t0, 1)
        (OUT / f"{name}.json").write_text(json.dumps(res, indent=2))
        summary[name] = {k: res["results"][k]["f1_mean"] for k in res["results"]}
        summary[name].update({"best_graph": res["best_graph"], "best_summary": res["best_summary"],
                              "selection_rule": res["selection_rule"],
                              "test_selected_best_graph": res["test_selected_best_graph"],
                              "val_f1": {k: res["results"][k].get("val_f1_mean") for k in res["results"]},
                              "p_graph_vs_histgbt": res["paired_best_graph_vs_histgbt"]["p"],
                              "acc_gain_graph_vs_histgbt": res["paired_best_graph_vs_histgbt"]["acc_gain_graph"]})
        print(f"[{name}] done in {res['wall_s']}s: {json.dumps(summary[name])}", flush=True)
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
