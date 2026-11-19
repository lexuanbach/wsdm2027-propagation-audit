"""Stratum-wise calibration diagnostic.

For each (K_P, K_E) stratum, reports the predicted positive rate vs the
actual positive rate on the test set, giving a direct diagnostic of the
backdoor adjustment estimator's quality.  Also computes stratum-conditional
error rate and a global Expected Calibration Error (ECE) across strata.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..abstain.selector import calibrate_threshold, selective_predict
from ..causal.strata import StratumEstimator
from ..devices import move_hdata, select_device
from ..hypergraph import build_hypergraph
from ..model import DocLessFake
from ..ssl import CascadeMLM, cascade_ssl_loss, mask_cascade_slices
from ..train import (
    DataConfig,
    ModelConfig,
    TrainConfig,
    _load_cascades,
    _set_seed,
    _stratified_split,
    train_one_seed,
)


def _train_and_collect(
    seed: int,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: torch.device,
) -> dict:
    """Train DocLessFake and return per-stratum calibration info on the test set."""
    _set_seed(seed)
    cascades = _load_cascades(data_cfg, seed)
    tr, va, te = _stratified_split(cascades, data_cfg.split_ratios, seed)

    estimator = StratumEstimator(kp=model_cfg.kp, ke=model_cfg.ke).fit(tr)
    q_hat = estimator.q_tensor().to(device)

    h_tr = build_hypergraph(tr, model_cfg.delta_minutes, model_cfg.k_max, feature_dim=16, seed=seed)
    h_te = build_hypergraph(te, model_cfg.delta_minutes, model_cfg.k_max, feature_dim=16, seed=seed + 2)
    p_tr, e_tr = estimator.transform(tr)
    p_te, e_te = estimator.transform(te)

    move_hdata(h_tr, device); move_hdata(h_te, device)
    p_tr, e_tr = p_tr.to(device), e_tr.to(device)
    p_te, e_te = p_te.to(device), e_te.to(device)
    y_tr, y_te = h_tr.story_labels, h_te.story_labels

    model = DocLessFake(
        in_dim=16, hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers, num_heads=model_cfg.num_heads,
        dropout=model_cfg.dropout, kp=model_cfg.kp, ke=model_cfg.ke,
    ).to(device)
    ssl_head = CascadeMLM(hidden_dim=model_cfg.hidden_dim).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(ssl_head.parameters()),
        lr=train_cfg.learning_rate,
    )

    for ep in range(train_cfg.pretrain_epochs):
        model.train(); opt.zero_grad()
        node_repr = model.encoder(h_tr.x, h_tr.edge_index, h_tr.edge_type, h_tr.node_type)
        batch = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4, seed=seed * 100 + ep)
        for attr in ("anchor_idx", "positive_idx", "negative_idx"):
            setattr(batch, attr, getattr(batch, attr).to(device))
        cascade_ssl_loss(node_repr, ssl_head, batch).backward()
        opt.step()

    import torch.nn.functional as F
    anneal_steps = max(1, int(train_cfg.finetune_epochs * train_cfg.kl_anneal_frac))
    for ep in range(train_cfg.finetune_epochs):
        lam_kl = train_cfg.lambda_kl_max * min(1.0, ep / anneal_steps)
        model.train(); opt.zero_grad()
        out_tr = model(h_tr, p_tr, e_tr, q_hat)
        ce = F.cross_entropy(out_tr["logits_do"], y_tr)
        batch = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4, seed=seed * 1000 + ep)
        for attr in ("anchor_idx", "positive_idx", "negative_idx"):
            setattr(batch, attr, getattr(batch, attr).to(device))
        l_ssl = cascade_ssl_loss(out_tr["node_repr"], ssl_head, batch)
        p_do = out_tr["logits_do"].exp()
        p_obs_logsoft = out_tr["logits_obs"].log_softmax(-1)
        l_kl = F.kl_div(p_obs_logsoft, p_do, reduction="batchmean")
        (ce + train_cfg.lambda_ssl * l_ssl + lam_kl * l_kl).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out_te = model(h_te, p_te, e_te, q_hat)

    probs_do = out_te["logits_do"].softmax(-1)[:, 1].cpu().numpy()
    preds = out_te["logits_do"].argmax(-1).cpu().numpy()
    labels = y_te.cpu().numpy()
    p_bins = p_te.cpu().numpy()
    e_bins = e_te.cpu().numpy()

    strata = {}
    for pi in range(model_cfg.kp):
        for ei in range(model_cfg.ke):
            mask = (p_bins == pi) & (e_bins == ei)
            n = int(mask.sum())
            if n == 0:
                strata[f"p{pi}_e{ei}"] = {
                    "n": 0, "pred_pos_rate": None, "actual_pos_rate": None,
                    "error_rate": None, "mean_prob": None,
                }
                continue
            pred_pos = float(preds[mask].mean())
            actual_pos = float(labels[mask].mean())
            err = float((preds[mask] != labels[mask]).mean())
            mean_prob = float(probs_do[mask].mean())
            strata[f"p{pi}_e{ei}"] = {
                "n": n,
                "pred_pos_rate": round(pred_pos, 4),
                "actual_pos_rate": round(actual_pos, 4),
                "error_rate": round(err, 4),
                "mean_prob": round(mean_prob, 4),
            }

    # Global ECE across strata
    ece = 0.0
    total = 0
    for s in strata.values():
        if s["n"] > 0 and s["mean_prob"] is not None:
            ece += s["n"] * abs(s["mean_prob"] - s["actual_pos_rate"])
            total += s["n"]
    ece = round(ece / max(total, 1), 4)

    return {
        "seed": seed,
        "strata": strata,
        "ece": ece,
        "n_test": int(labels.shape[0]),
    }


def run(
    datasets: dict[str, DataConfig],
    seeds: list[int],
    out_dir: Path,
    pretrain_epochs: int = 4,
    finetune_epochs: int = 25,
    hidden_dim: int = 64,
    kp: int = 4,
    ke: int = 4,
) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()

    mcfg = ModelConfig(hidden_dim=hidden_dim, num_layers=2, kp=kp, ke=ke)
    tcfg = TrainConfig(pretrain_epochs=pretrain_epochs, finetune_epochs=finetune_epochs)

    for ds_name, dcfg in datasets.items():
        for seed in seeds:
            result = _train_and_collect(seed, dcfg, mcfg, tcfg, device)
            result["dataset"] = ds_name
            rows.append(result)
            print(f"  [{ds_name:>10s} seed={seed}] ECE={result['ece']:.4f} "
                  f"n_test={result['n_test']}")

    (out_dir / "stratum_calibration.json").write_text(
        json.dumps({"results": rows, "wall_s": time.time() - t0}, indent=2)
    )

    # Aggregate ECE across seeds per dataset
    agg = {}
    for r in rows:
        agg.setdefault(r["dataset"], []).append(r["ece"])
    summary = [
        {"dataset": d, "ece_mean": round(float(np.mean(v)), 4),
         "ece_std": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else 0.0,
         "n_seeds": len(v)}
        for d, v in agg.items()
    ]
    (out_dir / "stratum_calibration_agg.json").write_text(json.dumps(summary, indent=2))
    return {"rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--pretrain-epochs", type=int, default=4)
    ap.add_argument("--finetune-epochs", type=int, default=25)
    ap.add_argument("--kp", type=int, default=4)
    ap.add_argument("--ke", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("outputs/r7/stratum_calibration"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.seeds, args.out,
              pretrain_epochs=args.pretrain_epochs,
              finetune_epochs=args.finetune_epochs,
              hidden_dim=args.hidden_dim, kp=args.kp, ke=args.ke)
    print("\nSummary:")
    for s in out["summary"]:
        print(f"  {s['dataset']:>10s} ECE {s['ece_mean']:.4f} +/- {s['ece_std']:.4f}")


if __name__ == "__main__":
    main()
