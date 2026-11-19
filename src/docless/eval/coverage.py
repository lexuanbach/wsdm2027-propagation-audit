"""Coverage runner — Fig 4 shape.

Sweeps coverage budgets c ∈ [0.5, 1.0] and emits accuracy + F1 at each
threshold. For each (dataset, seed) we train DocLessFake once and then
sweep τ over a grid of quantiles of the val-set disagreement signal.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ..abstain.selector import calibrate_threshold, selective_predict
from ..causal.strata import StratumEstimator
from ..devices import move_hdata, select_device
from ..eval.stats import macro_f1
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
)


def _train_then_sweep(seed: int, dcfg: DataConfig, mcfg: ModelConfig, tcfg: TrainConfig,
                     budgets: list[float], device: torch.device) -> list[dict]:
    """Train DocLessFake once, then sweep coverage budgets at eval time."""
    _set_seed(seed)
    cascades = _load_cascades(dcfg, seed)
    tr, va, te = _stratified_split(cascades, dcfg.split_ratios, seed)

    estimator = StratumEstimator(kp=mcfg.kp, ke=mcfg.ke).fit(tr)
    q_hat = estimator.q_tensor()

    h_tr = build_hypergraph(tr, mcfg.delta_minutes, mcfg.k_max, feature_dim=16, seed=seed)
    h_va = build_hypergraph(va, mcfg.delta_minutes, mcfg.k_max, feature_dim=16, seed=seed + 1)
    h_te = build_hypergraph(te, mcfg.delta_minutes, mcfg.k_max, feature_dim=16, seed=seed + 2)
    p_tr, e_tr = estimator.transform(tr)
    p_va, e_va = estimator.transform(va)
    p_te, e_te = estimator.transform(te)

    for h, p, e in [(h_tr, p_tr, e_tr), (h_va, p_va, e_va), (h_te, p_te, e_te)]:
        move_hdata(h, device)
    p_tr, e_tr = p_tr.to(device), e_tr.to(device)
    p_va, e_va = p_va.to(device), e_va.to(device)
    p_te, e_te = p_te.to(device), e_te.to(device)
    q_hat = q_hat.to(device)
    y_tr, y_va, y_te = h_tr.story_labels, h_va.story_labels, h_te.story_labels

    model = DocLessFake(in_dim=16, hidden_dim=mcfg.hidden_dim,
                         num_layers=mcfg.num_layers, num_heads=mcfg.num_heads,
                         dropout=mcfg.dropout, kp=mcfg.kp, ke=mcfg.ke).to(device)
    ssl_head = CascadeMLM(hidden_dim=mcfg.hidden_dim).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(ssl_head.parameters()),
                            lr=tcfg.learning_rate)

    for _ in range(tcfg.pretrain_epochs):
        model.train(); opt.zero_grad()
        node_repr = model.encoder(h_tr.x, h_tr.edge_index, h_tr.edge_type, h_tr.node_type)
        batch = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4, seed=seed)
        for n in ("anchor_idx", "positive_idx", "negative_idx"):
            setattr(batch, n, getattr(batch, n).to(device))
        cascade_ssl_loss(node_repr, ssl_head, batch).backward(); opt.step()

    import torch.nn.functional as F
    anneal = max(1, int(tcfg.finetune_epochs * tcfg.kl_anneal_frac))
    for ep in range(tcfg.finetune_epochs):
        lam_kl = tcfg.lambda_kl_max * min(1.0, ep / anneal)
        model.train(); opt.zero_grad()
        out_tr = model(h_tr, p_tr, e_tr, q_hat)
        ce = F.cross_entropy(out_tr["logits_do"], y_tr)
        batch = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4, seed=seed * 7 + ep)
        for n in ("anchor_idx", "positive_idx", "negative_idx"):
            setattr(batch, n, getattr(batch, n).to(device))
        l_ssl = cascade_ssl_loss(out_tr["node_repr"], ssl_head, batch)
        p_do = out_tr["logits_do"].exp()
        l_kl = F.kl_div(out_tr["logits_obs"].log_softmax(-1), p_do, reduction="batchmean")
        (ce + tcfg.lambda_ssl * l_ssl + lam_kl * l_kl).backward(); opt.step()

    # ---- sweep coverage budgets ----
    model.eval()
    with torch.no_grad():
        out_va = model(h_va, p_va, e_va, q_hat)
        out_te = model(h_te, p_te, e_te, q_hat)

    rows: list[dict] = []
    for c in budgets:
        tau = calibrate_threshold(out_va["delta"], c)
        preds, keep = selective_predict(out_te["logits_obs"], out_te["logits_do"], tau)
        cov = float(keep.float().mean().item())
        if keep.any():
            f1_k = macro_f1(preds[keep].cpu().numpy(), y_te[keep].cpu().numpy())
            acc_k = float((preds[keep] == y_te[keep]).float().mean().item())
        else:
            f1_k = float("nan"); acc_k = float("nan")
        rows.append({"seed": seed, "budget": c, "tau": tau,
                     "coverage": cov,
                     "acc_kept": acc_k, "f1_kept": f1_k,
                     "acc_all": float((preds == y_te).float().mean().item()),
                     "f1_all": macro_f1(preds.cpu().numpy(), y_te.cpu().numpy())})
    return rows


def run(datasets: dict[str, DataConfig], seeds: list[int],
        budgets: list[float], out_dir: Path,
        pretrain_epochs: int = 4, finetune_epochs: int = 20,
        hidden_dim: int = 64) -> dict:
    device = select_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    t0 = time.time()
    for ds_name, dcfg in datasets.items():
        for seed in seeds:
            t = time.time()
            tcfg = TrainConfig(pretrain_epochs=pretrain_epochs,
                                finetune_epochs=finetune_epochs)
            mcfg = ModelConfig(hidden_dim=hidden_dim, num_layers=2)
            for row in _train_then_sweep(seed, dcfg, mcfg, tcfg, budgets, device):
                row["dataset"] = ds_name
                rows.append(row)
            print(f"  [{ds_name:>10s} | seed {seed}] swept {len(budgets)} budgets "
                  f"in {time.time() - t:.1f}s")

    (out_dir / "coverage.json").write_text(json.dumps(
        {"results": rows, "wall_s_total": time.time() - t0,
         "budgets": budgets}, indent=2))

    # curve per (dataset, budget): mean over seeds
    agg = {}
    for r in rows:
        key = (r["dataset"], r["budget"])
        agg.setdefault(key, []).append((r["coverage"], r["f1_kept"]))
    curves = []
    for (d, b), pts in agg.items():
        covs = [p[0] for p in pts]
        f1s = [p[1] for p in pts if p[1] == p[1]]   # nan filter
        curves.append({"dataset": d, "budget": b,
                       "coverage_mean": round(sum(covs) / len(covs), 3),
                       "f1_kept_mean": round(sum(f1s) / max(1, len(f1s)), 3),
                       "n_seeds": len(pts)})
    curves.sort(key=lambda r: (r["dataset"], r["budget"]))
    (out_dir / "coverage_curve.json").write_text(json.dumps(curves, indent=2))
    return {"rows": rows, "curves": curves}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["coaid", "recovery"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--pretrain-epochs", type=int, default=2)
    ap.add_argument("--finetune-epochs", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("outputs/coverage"))
    args = ap.parse_args()
    ds = {}
    if "coaid" in args.datasets:
        ds["coaid"] = DataConfig(source="coaid", balanced=True)
    if "recovery" in args.datasets:
        ds["recovery"] = DataConfig(source="recovery", balanced=True, data_root="data/raw/recovery")
    out = run(ds, args.seeds, args.budgets, args.out,
              pretrain_epochs=args.pretrain_epochs,
              finetune_epochs=args.finetune_epochs,
              hidden_dim=args.hidden_dim)
    print("\nCoverage curve:")
    for r in out["curves"]:
        print(f"  {r['dataset']:>10s} | budget {r['budget']:.2f} "
              f"-> cov {r['coverage_mean']:.3f}, f1_kept {r['f1_kept_mean']:.3f}")


if __name__ == "__main__":
    main()
