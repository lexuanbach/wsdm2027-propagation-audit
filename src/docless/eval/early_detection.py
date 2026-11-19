"""Early-detection evaluation.

Trains DocLessFake once on full training cascades, then evaluates on test
cascades truncated to the first {25, 50, 75, 100}% of their events, simulating
early-warning scenarios where only the beginning of a cascade is observed.
The early-window confounders P and E are recomputed on the truncated stream,
so this exercises exactly the early-window design the method is built around.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..causal.strata import StratumEstimator
from ..data.schema import Cascade
from ..devices import move_hdata, select_device
from ..eval.stats import macro_f1
from ..hypergraph import build_hypergraph
from ..model import DocLessFake
from ..ssl import CascadeMLM, cascade_ssl_loss, mask_cascade_slices
from ..train import (
    DataConfig,
    _load_cascades,
    _set_seed,
    _stratified_split,
    _story_summary,
)


def _truncate(cascades: list[Cascade], frac: float) -> list[Cascade]:
    """Keep the first ceil(frac * n) events of each cascade; recompute early window."""
    out = []
    for c in cascades:
        n = len(c.events)
        k = max(1, math.ceil(frac * n))
        ev = c.events[:k]
        t0 = ev[0].timestamp
        early_end = min(ev[min(9, len(ev) - 1)].timestamp, t0 + 3600.0)
        out.append(Cascade(
            story_id=c.story_id, label=c.label, events=ev,
            language=c.language, early_window_end=early_end, timestamp=t0,
        ))
    return out


def _train_full(tr, seed, device, hidden_dim, pretrain_epochs, finetune_epochs):
    estimator = StratumEstimator(kp=4, ke=4).fit(tr)
    q_hat = estimator.q_tensor().to(device)
    h_tr = build_hypergraph(tr, feature_dim=16, seed=seed)
    p_tr, e_tr = estimator.transform(tr)
    move_hdata(h_tr, device)
    p_tr, e_tr = p_tr.to(device), e_tr.to(device)
    y_tr = h_tr.story_labels
    # Fused model: attach train-standardized cascade-summary features.
    s_tr = _story_summary(tr)
    mu = s_tr.mean(0, keepdims=True); sd = s_tr.std(0, keepdims=True) + 1e-6
    h_tr.story_summary = torch.tensor((s_tr - mu) / sd, dtype=torch.float32, device=device)
    model = DocLessFake(in_dim=16, hidden_dim=hidden_dim, num_layers=2, num_heads=4,
                        dropout=0.2, kp=4, ke=4).to(device)
    ssl_head = CascadeMLM(hidden_dim=hidden_dim).to(device)
    summary_params = set(model.summary_proj.parameters())
    base = [p for p in model.parameters() if p not in summary_params] + list(ssl_head.parameters())
    opt = torch.optim.Adam([{"params": base, "lr": 5e-4},
                            {"params": list(summary_params), "lr": 1e-2}])
    for ep in range(pretrain_epochs):
        opt.zero_grad()
        node = model.encoder(h_tr.x, h_tr.edge_index, h_tr.edge_type, h_tr.node_type)
        b = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4, seed=seed * 100 + ep)
        for n in ("anchor_idx", "positive_idx", "negative_idx"):
            setattr(b, n, getattr(b, n).to(device))
        cascade_ssl_loss(node, ssl_head, b).backward()
        opt.step()
    for ep in range(finetune_epochs):
        opt.zero_grad()
        out = model(h_tr, p_tr, e_tr, q_hat)
        F.cross_entropy(out["logits_do"], y_tr).backward()
        opt.step()
    return model, estimator, q_hat, (mu, sd)


def _eval_trunc(model, estimator, q_hat, te, frac, seed, device, summ_stats):
    te_t = _truncate(te, frac)
    h_te = build_hypergraph(te_t, feature_dim=16, seed=seed + 2)
    move_hdata(h_te, device)
    p_te, e_te = estimator.transform(te_t)
    p_te, e_te = p_te.to(device), e_te.to(device)
    y_te = h_te.story_labels
    mu, sd = summ_stats
    s_te = (_story_summary(te_t) - mu) / sd
    h_te.story_summary = torch.tensor(s_te, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        out = model(h_te, p_te, e_te, q_hat)
    preds = out["logits_do"].argmax(-1).cpu().numpy()
    return macro_f1(preds, y_te.cpu().numpy())


def run(datasets, seeds, fracs, out_dir: Path, hidden_dim=64, pretrain_epochs=4, finetune_epochs=25) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for ds_name, dcfg in datasets.items():
        for seed in seeds:
            _set_seed(seed)
            cascades = _load_cascades(dcfg, seed)
            tr, va, te = _stratified_split(cascades, dcfg.split_ratios, seed)
            model, estimator, q_hat, summ_stats = _train_full(tr, seed, device, hidden_dim, pretrain_epochs, finetune_epochs)
            for frac in fracs:
                f1 = _eval_trunc(model, estimator, q_hat, te, frac, seed, device, summ_stats)
                rows.append(dict(dataset=ds_name, seed=seed, frac=frac, f1=f1))
                print(f"  [{ds_name:>8s} seed={seed} frac={frac:.2f}] F1={f1:.3f}")
    (out_dir / "early_detection.json").write_text(json.dumps({"rows": rows, "wall_s": time.time() - t0}, indent=2))
    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["frac"]), []).append(r["f1"])
    cells = [dict(dataset=d, frac=fr,
                  f1_mean=round(float(np.mean(v)), 3),
                  f1_std=round(float(np.std(v, ddof=1)), 3) if len(v) > 1 else 0.0,
                  n_seeds=len(v))
             for (d, fr), v in agg.items()]
    (out_dir / "early_detection_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--pretrain-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=25)
    ap.add_argument("--out", type=Path, default=Path("outputs/r8/early_detection"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.seeds, args.fracs, args.out, hidden_dim=args.hidden_dim,
              pretrain_epochs=args.pretrain_epochs, finetune_epochs=args.finetune_epochs)
    print("\nCells:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>8s} frac={c['frac']:.2f}  F1 {c['f1_mean']:.3f} +/- {c['f1_std']:.3f}")


if __name__ == "__main__":
    main()
