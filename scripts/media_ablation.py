"""Media-node ablation: do media objects contribute?

Compares the fused model with vs without media nodes + media-mediated coexpose
edges in the hypergraph (summary features held fixed), isolating the structural
contribution of media objects. Fused model, hidden-64, 4+25, 5 seeds.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from docless.causal.strata import StratumEstimator
from docless.devices import move_hdata
from docless.eval.stats import macro_f1
from docless.hypergraph import build_hypergraph
from docless.model import DocLessFake
from docless.ssl import CascadeMLM, cascade_ssl_loss, mask_cascade_slices
from docless.train import (
    DataConfig, _load_cascades, _set_seed, _stratified_split, _story_summary,
)

OUT = "outputs/r9/media_ablation"
os.makedirs(OUT, exist_ok=True)
DSETS = {"coaid": "data/raw/coaid", "recovery": "data/raw/recovery"}


def _attach(h, casc, mu, sd, device):
    h.story_summary = torch.tensor((_story_summary(casc) - mu) / sd, dtype=torch.float32, device=device)


def run_seed(ds, root, seed, drop_media, device):
    _set_seed(seed)
    casc = _load_cascades(DataConfig(source=ds, balanced=True, data_root=root), seed)
    tr, va, te = _stratified_split(casc, (0.8, 0.1, 0.1), seed)
    est = StratumEstimator(kp=4, ke=4).fit(tr)
    q = est.q_tensor().to(device)
    h_tr = build_hypergraph(tr, feature_dim=16, seed=seed, drop_media=drop_media)
    h_te = build_hypergraph(te, feature_dim=16, seed=seed + 2, drop_media=drop_media)
    for h in (h_tr, h_te):
        move_hdata(h, device)
    s_tr = _story_summary(tr); mu = s_tr.mean(0, keepdims=True); sd = s_tr.std(0, keepdims=True) + 1e-6
    _attach(h_tr, tr, mu, sd, device); _attach(h_te, te, mu, sd, device)
    p_tr, e_tr = est.transform(tr); p_te, e_te = est.transform(te)
    p_tr, e_tr, p_te, e_te = (x.to(device) for x in (p_tr, e_tr, p_te, e_te))
    y_tr = h_tr.story_labels

    model = DocLessFake(in_dim=16, hidden_dim=64, num_layers=2, num_heads=4, dropout=0.2, kp=4, ke=4).to(device)
    ssl = CascadeMLM(hidden_dim=64).to(device)
    sp = set(model.summary_proj.parameters())
    base = [p for p in model.parameters() if p not in sp] + list(ssl.parameters())
    opt = torch.optim.Adam([{"params": base, "lr": 5e-4}, {"params": list(sp), "lr": 1e-2}])
    for ep in range(4):
        opt.zero_grad()
        node = model.encoder(h_tr.x, h_tr.edge_index, h_tr.edge_type, h_tr.node_type)
        b = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4, seed=seed * 100 + ep)
        for n in ("anchor_idx", "positive_idx", "negative_idx"):
            setattr(b, n, getattr(b, n).to(device))
        cascade_ssl_loss(node, ssl, b).backward(); opt.step()
    for ep in range(25):
        opt.zero_grad()
        F.cross_entropy(model(h_tr, p_tr, e_tr, q)["logits_do"], y_tr).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(h_te, p_te, e_te, q)["logits_do"].argmax(-1).cpu().numpy()
    return macro_f1(preds, h_te.story_labels.cpu().numpy())


if __name__ == "__main__":
    res = {}
    for ds, root in DSETS.items():
        res[ds] = {}
        for name, dm in (("full", False), ("no_media", True)):
            f = [run_seed(ds, root, s, dm, torch.device("cpu")) for s in range(5)]
            res[ds][name] = {"f1_mean": round(float(np.mean(f)), 3), "f1_std": round(float(np.std(f)), 3)}
            print(ds, name, res[ds][name], flush=True)
        json.dump(res, open(f"{OUT}/media_ablation.json", "w"), indent=2)
    print("DONE", flush=True)
