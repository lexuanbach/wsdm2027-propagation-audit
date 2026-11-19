"""GNN tuning parity + EBGCN comparator.

Gives GCNFN, BiGCN, RGCN the SAME validation-selected (epochs x lr) search GATv2
received, and adds an EBGCN-style edge-enhanced bidirectional GCN. Same per-split
hypergraph, node features, burst readout, 5 seeds, validation-only selection.
Tests whether 'PropMLP beats every propagation GNN' survives equal tuning and a
dedicated rumor-detection model.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import torch, torch.nn.functional as F

from docless.baselines.gcnfn import _PropGCN
from docless.baselines.bigcn import _BiPropGCN
from docless.baselines.rgcn import _RGCN
from docless.baselines.ebgcn import _EBGCN
from docless.eval.stats import macro_f1
from docless.hypergraph import build_hypergraph
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

# GNN_PARITY_OUT lets a second dataset run in parallel without two writers
# racing on the same JSON; merge the shards afterwards.
OUT = os.environ.get("GNN_PARITY_OUT", "outputs/r10/gnn_parity"); os.makedirs(OUT, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
CONFIGS = [dict(epochs=25, lr=5e-4), dict(epochs=40, lr=1e-3), dict(epochs=40, lr=5e-3), dict(epochs=80, lr=5e-3)]
MODELS = {"gcnfn": _PropGCN, "bigcn": _BiPropGCN, "rgcn": _RGCN, "ebgcn": _EBGCN}
DATASETS = {
    "coaid":      dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}
dev = torch.device("cpu")


def _fit_eval(Cls, h_tr, y_tr, h_ev, y_ev, epochs, lr, seed):
    _set_seed(seed)
    m = Cls(in_dim=16, hidden_dim=64).to(dev); opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        F.cross_entropy(m(h_tr, h_tr.story_to_burst_edges, h_tr.story_ids), y_tr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        preds = m(h_ev, h_ev.story_to_burst_edges, h_ev.story_ids).argmax(-1).cpu().numpy()
    return macro_f1(preds, y_ev.cpu().numpy())


def run(name, kw):
    dcfg = DataConfig(**kw); res = {mn: [] for mn in MODELS}
    for s in SEEDS:
        _set_seed(s); casc = _load_cascades(dcfg, s); tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
        h_tr = build_hypergraph(tr, feature_dim=16, seed=s)
        h_va = build_hypergraph(va, feature_dim=16, seed=s + 1)
        h_te = build_hypergraph(te, feature_dim=16, seed=s + 2)
        y_tr, y_va, y_te = h_tr.story_labels, h_va.story_labels, h_te.story_labels
        for mn, Cls in MODELS.items():
            best, bestv = None, -1
            for cfg in CONFIGS:
                vf1 = _fit_eval(Cls, h_tr, y_tr, h_va, y_va, cfg["epochs"], cfg["lr"], s)
                if vf1 > bestv: bestv, best = vf1, cfg
            tf1 = _fit_eval(Cls, h_tr, y_tr, h_te, y_te, best["epochs"], best["lr"], s)
            res[mn].append(round(float(tf1), 4))
    agg = {mn: dict(f1_mean=round(float(np.mean(v)), 3), f1_std=round(float(np.std(v)), 3), per_seed=v) for mn, v in res.items()}
    print(f"[{name}] " + "  ".join(f"{mn}={agg[mn]['f1_mean']}" for mn in MODELS), flush=True)
    return agg


if __name__ == "__main__":
    import sys
    # optional dataset filter + resumable output, so the matched-tuning sweep can
    # be extended to the powered corpora without redoing CoAID
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(DATASETS)
    t0 = time.time()
    out_path = f"{OUT}/gnn_parity.json"
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for name, kw in ((n, DATASETS[n]) for n in wanted):
        out[name] = run(name, kw)
        json.dump(out, open(out_path, "w"), indent=2)
    print(f"ALL DONE in {round(time.time()-t0)}s", flush=True)
