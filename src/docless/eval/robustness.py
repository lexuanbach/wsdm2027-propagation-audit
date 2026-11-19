"""Label-noise + popularity-shift robustness sweeps on real data.

Two perturbations, each with the same train-time / test-time split as the
main table, run on both CoAID and ReCOVery.

  label_noise   flip a fraction nu in {0, .1, .2, .3} of training labels
  popularity_shift   resample test set by the high-popularity stratum
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..abstain.selector import calibrate_threshold, selective_predict
from ..causal.strata import StratumEstimator, story_p_e_features
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
)


def _flip_labels(cascades: list[Cascade], nu: float, seed: int) -> list[Cascade]:
    if nu <= 0:
        return cascades
    rng = np.random.default_rng(seed)
    out = []
    for c in cascades:
        if rng.random() < nu:
            out.append(Cascade(
                story_id=c.story_id,
                label=1 - c.label,
                events=c.events,
                language=c.language,
                early_window_end=c.early_window_end,
                timestamp=c.timestamp,
            ))
        else:
            out.append(c)
    return out


def _popularity_shift(cascades: list[Cascade], hi_frac: float, seed: int) -> list[Cascade]:
    """Re-weight the test population toward high-popularity stratum."""
    if hi_frac <= 0:
        return cascades
    rng = np.random.default_rng(seed)
    p_vals, _ = story_p_e_features(cascades)
    median = np.median(p_vals)
    hi = [c for c, p in zip(cascades, p_vals) if p >= median]
    lo = [c for c, p in zip(cascades, p_vals) if p < median]
    n = len(cascades)
    n_hi = int(round(n * hi_frac))
    n_lo = n - n_hi
    rng.shuffle(hi)
    rng.shuffle(lo)
    out = (hi * ((n_hi // max(1, len(hi))) + 1))[:n_hi] + (lo * ((n_lo // max(1, len(lo))) + 1))[:n_lo]
    rng.shuffle(out)
    return out


def _train_once(tr, va, te, seed: int, device: torch.device) -> tuple[DocLessFake, StratumEstimator, torch.Tensor]:
    estimator = StratumEstimator(kp=4, ke=4).fit(tr)
    q_hat = estimator.q_tensor().to(device)
    h_tr = build_hypergraph(tr, feature_dim=16, seed=seed)
    p_tr, e_tr = estimator.transform(tr)
    move_hdata(h_tr, device)
    p_tr, e_tr = p_tr.to(device), e_tr.to(device)
    y_tr = h_tr.story_labels
    model = DocLessFake(in_dim=16, hidden_dim=64, num_layers=2, num_heads=4,
                         dropout=0.2, kp=4, ke=4).to(device)
    ssl_head = CascadeMLM(hidden_dim=64).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(ssl_head.parameters()),
                            lr=5e-4)
    for _ in range(2):
        opt.zero_grad()
        h_nodes = model.encoder(h_tr.x, h_tr.edge_index, h_tr.edge_type, h_tr.node_type)
        b = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4, seed=seed)
        for n in ("anchor_idx","positive_idx","negative_idx"):
            setattr(b, n, getattr(b,n).to(device))
        cascade_ssl_loss(h_nodes, ssl_head, b).backward()
        opt.step()
    for _ in range(15):
        opt.zero_grad()
        out = model(h_tr, p_tr, e_tr, q_hat)
        F.cross_entropy(out["logits_do"], y_tr).backward()
        opt.step()
    return model, estimator, q_hat


def _eval(model, estimator, q_hat, eval_cascades, seed, device):
    h_te = build_hypergraph(eval_cascades, feature_dim=16, seed=seed + 7)
    move_hdata(h_te, device)
    p_te, e_te = estimator.transform(eval_cascades)
    p_te, e_te = p_te.to(device), e_te.to(device)
    y_te = h_te.story_labels
    model.eval()
    with torch.no_grad():
        out = model(h_te, p_te, e_te, q_hat)
    preds = out["logits_do"].argmax(-1).cpu().numpy()
    return macro_f1(preds, y_te.cpu().numpy())


def run(datasets, seeds, out_dir: Path) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for ds_name, dcfg in datasets.items():
        for seed in seeds:
            _set_seed(seed)
            cascades = _load_cascades(dcfg, seed)
            tr, va, te = _stratified_split(cascades, dcfg.split_ratios, seed)
            for nu in [0.0, 0.10, 0.20, 0.30]:
                tr_noisy = _flip_labels(tr, nu, seed * 13)
                model, estim, q_hat = _train_once(tr_noisy, va, te, seed, device)
                f1 = _eval(model, estim, q_hat, te, seed, device)
                rows.append(dict(dataset=ds_name, seed=seed, perturb="label_noise", level=nu, f1=f1))
                print(f"  [{ds_name:>8s} seed={seed} noise={nu:.2f}] F1={f1:.3f}")
            for hi in [0.5, 0.7, 0.9]:
                te_shift = _popularity_shift(te, hi, seed * 17)
                model, estim, q_hat = _train_once(tr, va, te, seed, device)
                f1 = _eval(model, estim, q_hat, te_shift, seed, device)
                rows.append(dict(dataset=ds_name, seed=seed, perturb="pop_shift", level=hi, f1=f1))
                print(f"  [{ds_name:>8s} seed={seed} pop_hi={hi:.2f}] F1={f1:.3f}")

    (out_dir / "robustness.json").write_text(json.dumps({"rows": rows, "wall_s": time.time() - t0}, indent=2))

    agg = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["perturb"], r["level"]), []).append(r["f1"])
    cells = [
        dict(dataset=d, perturb=p, level=l,
             f1_mean=float(np.mean(v)), f1_std=float(np.std(v, ddof=1)) if len(v)>1 else 0.0,
             n_seeds=len(v))
        for (d, p, l), v in agg.items()
    ]
    (out_dir / "robustness_agg.json").write_text(json.dumps(cells, indent=2))
    return {"rows": rows, "cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=Path("outputs/r4/robustness"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.seeds, args.out)
    print("\nCells:")
    for c in out["cells"]:
        print(f"  {c['dataset']:>8s} {c['perturb']:>10s} L={c['level']:.2f}  "
              f"F1 {c['f1_mean']:.3f} ± {c['f1_std']:.3f}")


if __name__ == "__main__":
    main()
