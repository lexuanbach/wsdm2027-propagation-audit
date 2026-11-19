"""Abstention-baseline audit.

Does the proposed TV-disagreement abstention beat a trivial calibrated
softmax-confidence selector at matched coverage? Both thresholds are calibrated
on validation to coverage c=0.8; we report test F1-kept and achieved coverage.
Fused model, hidden-64, 4+25, 5 seeds.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from docless.abstain.selector import tv_disagreement
from docless.causal.strata import StratumEstimator
from docless.devices import move_hdata
from docless.eval.stats import macro_f1
from docless.hypergraph import build_hypergraph
from docless.model import DocLessFake
from docless.ssl import CascadeMLM, cascade_ssl_loss, mask_cascade_slices
from docless.train import (
    DataConfig, _load_cascades, _set_seed, _stratified_split, _story_summary,
)

COV = 0.8
OUT = "outputs/r9/abstention_baseline"
os.makedirs(OUT, exist_ok=True)
DSETS = {"coaid": "data/raw/coaid", "recovery": "data/raw/recovery"}


def _attach_summary(h, casc, mu, sd, device):
    s = (_story_summary(casc) - mu) / sd
    h.story_summary = torch.tensor(s, dtype=torch.float32, device=device)


def _f1_kept(preds, labels, keep):
    keep = keep.cpu().numpy().astype(bool)
    if keep.sum() == 0:
        return 0.0, 0.0
    return macro_f1(preds[keep], labels[keep]), float(keep.mean())


def run_seed(ds, root, seed, device):
    _set_seed(seed)
    casc = _load_cascades(DataConfig(source=ds, balanced=True, data_root=root), seed)
    tr, va, te = _stratified_split(casc, (0.8, 0.1, 0.1), seed)
    est = StratumEstimator(kp=4, ke=4).fit(tr)
    q = est.q_tensor().to(device)
    h_tr = build_hypergraph(tr, feature_dim=16, seed=seed)
    h_va = build_hypergraph(va, feature_dim=16, seed=seed + 1)
    h_te = build_hypergraph(te, feature_dim=16, seed=seed + 2)
    for h in (h_tr, h_va, h_te):
        move_hdata(h, device)
    s_tr = _story_summary(tr); mu = s_tr.mean(0, keepdims=True); sd = s_tr.std(0, keepdims=True) + 1e-6
    _attach_summary(h_tr, tr, mu, sd, device)
    _attach_summary(h_va, va, mu, sd, device)
    _attach_summary(h_te, te, mu, sd, device)
    p_tr, e_tr = est.transform(tr); p_va, e_va = est.transform(va); p_te, e_te = est.transform(te)
    p_tr, e_tr, p_va, e_va, p_te, e_te = (x.to(device) for x in (p_tr, e_tr, p_va, e_va, p_te, e_te))
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
        out = model(h_tr, p_tr, e_tr, q)
        F.cross_entropy(out["logits_do"], y_tr).backward(); opt.step()

    model.eval()
    with torch.no_grad():
        ov = model(h_va, p_va, e_va, q); ot = model(h_te, p_te, e_te, q)
    yte = h_te.story_labels.cpu().numpy()
    preds = ot["logits_do"].argmax(-1).cpu().numpy()

    # selector A: TV disagreement (keep small Δ)
    dv = tv_disagreement(ov["logits_obs"], ov["logits_do"])
    dt = tv_disagreement(ot["logits_obs"], ot["logits_do"])
    tau = float(np.quantile(dv.cpu().numpy(), COV))
    f1_tv, cov_tv = _f1_kept(preds, yte, dt <= tau)

    # selector B: softmax confidence (keep high confidence)
    cv = ov["logits_obs"].softmax(-1).max(-1).values
    ct = ot["logits_obs"].softmax(-1).max(-1).values
    thr = float(np.quantile(cv.cpu().numpy(), 1 - COV))
    f1_cf, cov_cf = _f1_kept(preds, yte, ct >= thr)

    f1_all = macro_f1(preds, yte)
    return f1_all, f1_tv, cov_tv, f1_cf, cov_cf


if __name__ == "__main__":
    res = {}
    for ds, root in DSETS.items():
        rows = [run_seed(ds, root, s, torch.device("cpu")) for s in range(5)]
        a = np.array(rows)
        res[ds] = {
            "f1_all": round(float(a[:, 0].mean()), 3),
            "tv_abstain_f1kept": round(float(a[:, 1].mean()), 3),
            "tv_coverage": round(float(a[:, 2].mean()), 3),
            "confidence_f1kept": round(float(a[:, 3].mean()), 3),
            "confidence_coverage": round(float(a[:, 4].mean()), 3),
        }
        print(ds, res[ds], flush=True)
        json.dump(res, open(f"{OUT}/abstention_baseline.json", "w"), indent=2)
    print("DONE", flush=True)
