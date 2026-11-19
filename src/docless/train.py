"""DLF-Full training loop (pretraining, fine-tuning, diagnostics) end to end.

Quick synthetic run (no downloads, under a minute on CPU):
    PYTHONPATH=src python -m docless.train --out outputs/smoke.json
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .abstain.selector import calibrate_threshold, selective_predict
from .causal.strata import StratumEstimator, _echo_chamber, _popularity
from .data.loaders import generate_synthetic, load_coaid
from .data.schema import Cascade
from .devices import move_hdata, select_device
from .hypergraph import build_hypergraph
from .model import DocLessFake
from .ssl import CascadeMLM, cascade_ssl_loss, mask_cascade_slices


# -------- config --------

@dataclass
class TrainConfig:
    pretrain_epochs: int = 2
    finetune_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-4
    lambda_ssl: float = 0.5
    lambda_kl_max: float = 0.1
    kl_anneal_frac: float = 0.5
    coverage_budget: float = 0.9
    ssl_label_free: bool = True
    # The cascade-summary branch is a shallow MLP over 7 low-dim features; it
    # needs a higher rate than the GNN encoder to converge in-budget (matching
    # the standalone PropMLP baseline's lr=1e-2). Trained with its own param
    # group so the encoder/head keep the tuned 5e-4 rate.
    summary_lr: float = 1e-2
    # Diagnostic control: randomly permute the per-story (P, E_hash) stratum
    # assignment. This tests whether the adjustment head uses these strata; it
    # does not establish that either variable is a confounder.
    permute_confounders: bool = False
    num_seeds: int = 1


@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    kp: int = 4
    ke: int = 4
    delta_minutes: float = 5.0
    k_max: int = 128
    # Cascade-summary fusion branch (paper §IV-A). use_summary=False reverts to
    # the structure-only readout; use_structure=False isolates the summary
    # branch (a learned PropMLP analogue) for the structure-vs-summary ablation.
    use_summary: bool = True
    use_structure: bool = True
    summary_dim: int = 7
    # When True, the summary branch consumes the 15-dim temporal-dynamics-enriched
    # feature vector (_story_summary_rich) instead of the base 7-dim one. The
    # model's summary_dim is set to 15 accordingly. Used to address the
    # concern that the fused model never significantly beats PropMLP.
    rich_summary: bool = False


@dataclass
class DataConfig:
    source: str = "synthetic"               # `synthetic` | `coaid` | `recovery`
    # synthetic-only
    n_stories: int = 400
    n_users: int = 600
    n_urls: int = 100
    n_media: int = 50
    # real-data fields (used by coaid + recovery)
    data_root: str = "data/raw/coaid"
    balanced: bool = True
    min_events: int = 3
    max_events: int = 200
    # back-compat aliases (kept so older configs keep loading)
    coaid_root: str | None = None
    coaid_balanced: bool | None = None
    coaid_min_events: int | None = None
    coaid_max_events: int | None = None
    # optional cap on balanced stories per class (used to subsample GossipCop
    # to a tractable, ReCOVery-comparable size while keeping it public)
    max_per_class: int | None = None
    # both
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)

    def __post_init__(self) -> None:
        # honour legacy field names if explicitly set
        if self.coaid_root is not None:
            self.data_root = self.coaid_root
        if self.coaid_balanced is not None:
            self.balanced = self.coaid_balanced
        if self.coaid_min_events is not None:
            self.min_events = self.coaid_min_events
        if self.coaid_max_events is not None:
            self.max_events = self.coaid_max_events
        # default root per source
        if self.source == "recovery" and self.data_root == "data/raw/coaid":
            self.data_root = "data/raw/recovery"


# -------- utils --------

def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _story_summary_rich(cascades: list[Cascade]) -> np.ndarray:
    """Temporal-dynamics-enriched cascade-summary features per story [n, 15].

    The base 7 are size/event-proxies/URLs/media/popularity/hash-concentration/spread (matches the
    plain summary). The 8 added features capture propagation timing that
    aggregate counts miss: arrival-rate central tendency and dispersion, early
    velocity at multiple horizons, time-to-quartile, early-vs-late distinct-
    user concentration, and a burst-count proxy.
    """
    rows = []
    for c in cascades:
        ts = np.asarray([e.timestamp for e in c.events], dtype=np.float64)
        ts.sort()
        n = len(ts)
        spread = float(ts[-1] - ts[0]) if n > 1 else 0.0
        # inter-arrival times (in seconds)
        if n > 1:
            dts = np.diff(ts)
            med_dt = float(np.median(dts))
            iqr_dt = float(np.quantile(dts, 0.75) - np.quantile(dts, 0.25))
        else:
            med_dt = 0.0; iqr_dt = 0.0
        # velocity: proportion of events in first 1h / 24h
        t0 = ts[0]
        prop_1h  = float((ts - t0 <= 3600.0).mean()) if n > 0 else 0.0
        prop_24h = float((ts - t0 <= 86400.0).mean()) if n > 0 else 0.0
        # time to 50% / 75% of events (in seconds, log-scaled)
        if n >= 2:
            t_q50 = float(ts[max(0, int(0.50 * n) - 1)] - t0)
            t_q75 = float(ts[max(0, int(0.75 * n) - 1)] - t0)
        else:
            t_q50 = 0.0; t_q75 = 0.0
        # early/late distinct-proxy ratio (first half vs second half)
        half = max(1, n // 2)
        users = [e.user_id for e in c.events]
        early_u = len(set(users[:half]))
        late_u  = len(set(users[half:])) if n > half else 0
        ratio = (early_u + 1) / (late_u + 1)
        # burst count: number of disjoint 5-min windows with >=5 events
        if n > 0:
            bins = np.floor((ts - t0) / 300.0).astype(np.int64)
            counts = np.bincount(bins)
            bursts = int((counts >= 5).sum())
        else:
            bursts = 0
        rows.append([
            np.log1p(n),
            np.log1p(c.num_users()),
            np.log1p(c.num_urls()),
            np.log1p(c.num_media()),
            np.log1p(_popularity(c)),
            _echo_chamber(c),
            np.log1p(spread),
            np.log1p(med_dt),
            np.log1p(iqr_dt),
            prop_1h,
            prop_24h,
            np.log1p(t_q50),
            np.log1p(t_q75),
            np.log1p(ratio),
            np.log1p(bursts),
        ])
    return np.asarray(rows, dtype=np.float64)


def _story_summary(cascades: list[Cascade]) -> np.ndarray:
    """Text-free cascade-summary features per story [n, 7].

    Identical feature set to the PropMLP baseline so the fusion branch consumes
    the same aggregate signal: log cascade size, log distinct event proxies,
    URLs/media, log early-window popularity, normalized hash-bucket
    concentration, and log temporal spread.
    """
    rows = []
    for c in cascades:
        ts = [e.timestamp for e in c.events]
        spread = (max(ts) - min(ts)) if len(ts) > 1 else 0.0
        rows.append([
            np.log1p(len(c.events)),
            np.log1p(c.num_users()),
            np.log1p(c.num_urls()),
            np.log1p(c.num_media()),
            np.log1p(_popularity(c)),
            _echo_chamber(c),
            np.log1p(spread),
        ])
    return np.asarray(rows, dtype=np.float64)


def _split(cascades: list[Cascade], ratios: tuple[float, float, float], seed: int) -> tuple[list, list, list]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cascades))
    rng.shuffle(idx)
    n = len(cascades)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    tr = [cascades[i] for i in idx[:n_train]]
    va = [cascades[i] for i in idx[n_train : n_train + n_val]]
    te = [cascades[i] for i in idx[n_train + n_val :]]
    return tr, va, te


def _accuracy(preds: Tensor, labels: Tensor, mask: Tensor | None = None) -> float:
    if mask is not None:
        preds = preds[mask]
        labels = labels[mask]
    if preds.numel() == 0:
        return 0.0
    return float((preds == labels).float().mean().item())


def _macro_f1(preds: Tensor, labels: Tensor, num_classes: int = 2) -> float:
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum().item()
        fp = ((preds == c) & (labels != c)).sum().item()
        fn = ((preds != c) & (labels == c)).sum().item()
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


# -------- main --------

def _load_cascades(data_cfg: DataConfig, seed: int) -> list[Cascade]:
    if data_cfg.source == "coaid":
        return load_coaid(
            data_cfg.data_root,
            min_events_per_story=data_cfg.min_events,
            max_events_per_story=data_cfg.max_events,
            balanced=data_cfg.balanced,
            seed=seed,
        )
    if data_cfg.source == "recovery":
        from .data.loaders import load_recovery
        return load_recovery(
            data_cfg.data_root,
            min_events_per_story=data_cfg.min_events,
            max_events_per_story=data_cfg.max_events,
            balanced=data_cfg.balanced,
            seed=seed,
        )
    if data_cfg.source in ("politifact", "fakenewsnet", "gossipcop"):
        from .data.loaders import load_fakenewsnet
        src = "gossipcop" if data_cfg.source == "gossipcop" else "politifact"
        return load_fakenewsnet(
            data_cfg.data_root,
            source=src,
            min_events_per_story=data_cfg.min_events,
            max_events_per_story=data_cfg.max_events,
            balanced=data_cfg.balanced,
            seed=seed,
            max_per_class=data_cfg.max_per_class,
        )
    return generate_synthetic(
        n_stories=data_cfg.n_stories,
        n_users=data_cfg.n_users,
        n_urls=data_cfg.n_urls,
        n_media=data_cfg.n_media,
        seed=seed,
    )


def _stratified_split(cascades: list[Cascade], ratios: tuple[float, float, float], seed: int):
    """Class-stratified split (essential for unbalanced data like CoAID)."""
    rng = np.random.default_rng(seed)
    by_label: dict[int, list[int]] = {}
    for i, c in enumerate(cascades):
        by_label.setdefault(c.label, []).append(i)
    tr_idx, va_idx, te_idx = [], [], []
    for _, idx in by_label.items():
        idx = list(idx)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(ratios[0] * n)
        n_va = int(ratios[1] * n)
        tr_idx.extend(idx[:n_tr])
        va_idx.extend(idx[n_tr : n_tr + n_va])
        te_idx.extend(idx[n_tr + n_va :])
    rng.shuffle(tr_idx); rng.shuffle(va_idx); rng.shuffle(te_idx)
    return ([cascades[i] for i in tr_idx],
            [cascades[i] for i in va_idx],
            [cascades[i] for i in te_idx])


def _temporal_split(cascades: list[Cascade], ratios: tuple[float, float, float], seed: int):
    """Deployment-realistic temporal split: order stories by creation time and
    assign the earliest ``ratios[0]`` to train, the next ``ratios[1]`` to val,
    and the latest remainder to test, so the test fold is strictly in the future
    of training. ``seed`` is accepted for a uniform call signature but the split
    is deterministic (a story's earliest event time, falling back to ``timestamp``,
    is the sort key)."""
    def _key(c: Cascade) -> float:
        ts = [e.timestamp for e in c.events]
        if ts:
            return min(ts)
        return c.timestamp if c.timestamp is not None else 0.0
    order = sorted(range(len(cascades)), key=lambda i: _key(cascades[i]))
    n = len(order)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)
    tr_idx = order[:n_tr]
    va_idx = order[n_tr : n_tr + n_va]
    te_idx = order[n_tr + n_va :]
    return ([cascades[i] for i in tr_idx],
            [cascades[i] for i in va_idx],
            [cascades[i] for i in te_idx])


def train_one_seed(
    seed: int,
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    verbose: bool = True,
    device: torch.device | None = None,
    split_fn=None,
    summary_override=None,   # callable(feat_fn, tr, va, te, seed) -> (std_tr, std_va, std_te); for drift-mitigation audit
    ce_weight_fn=None,       # callable(std_tr, std_te) -> [N_tr] per-sample CE weights (importance weighting)
) -> dict:
    _set_seed(seed)
    device = device or select_device()
    if verbose:
        print(f"[device] {device}")

    # --- data ---
    cascades = _load_cascades(data_cfg, seed)
    split_fn = split_fn or _stratified_split
    tr, va, te = split_fn(cascades, data_cfg.split_ratios, seed)

    # --- strata fit on TRAIN ONLY ---
    estimator = StratumEstimator(kp=model_cfg.kp, ke=model_cfg.ke).fit(tr)
    q_hat = estimator.q_tensor()

    # --- build hypergraphs per split ---
    h_tr = build_hypergraph(tr, model_cfg.delta_minutes, model_cfg.k_max, feature_dim=16, seed=seed)
    h_va = build_hypergraph(va, model_cfg.delta_minutes, model_cfg.k_max, feature_dim=16, seed=seed + 1)
    h_te = build_hypergraph(te, model_cfg.delta_minutes, model_cfg.k_max, feature_dim=16, seed=seed + 2)
    p_tr, e_tr = estimator.transform(tr)
    p_va, e_va = estimator.transform(va)
    p_te, e_te = estimator.transform(te)

    if train_cfg.permute_confounders:
        g = torch.Generator().manual_seed(seed + 9973)
        for pv, ev in ((p_tr, e_tr), (p_va, e_va), (p_te, e_te)):
            perm = torch.randperm(pv.numel(), generator=g)
            pv.copy_(pv[perm]); ev.copy_(ev[perm])

    move_hdata(h_tr, device); move_hdata(h_va, device); move_hdata(h_te, device)
    p_tr, e_tr = p_tr.to(device), e_tr.to(device)
    p_va, e_va = p_va.to(device), e_va.to(device)
    p_te, e_te = p_te.to(device), e_te.to(device)
    y_tr, y_va, y_te = h_tr.story_labels, h_va.story_labels, h_te.story_labels

    # --- cascade-summary features, standardized on TRAIN only (no leakage) ---
    ce_sample_w = None
    if model_cfg.use_summary:
        feat_fn = _story_summary_rich if model_cfg.rich_summary else _story_summary
        if summary_override is not None:
            std_tr, std_va, std_te = summary_override(feat_fn, tr, va, te, seed)
        else:
            s_tr_raw = feat_fn(tr)
            mu = s_tr_raw.mean(0, keepdims=True)
            sd = s_tr_raw.std(0, keepdims=True) + 1e-6
            std_tr = (feat_fn(tr) - mu) / sd
            std_va = (feat_fn(va) - mu) / sd
            std_te = (feat_fn(te) - mu) / sd
        for h_split, s in ((h_tr, std_tr), (h_va, std_va), (h_te, std_te)):
            h_split.story_summary = torch.tensor(s, dtype=torch.float32, device=device)
        if ce_weight_fn is not None:
            ce_sample_w = torch.tensor(ce_weight_fn(std_tr, std_te), dtype=torch.float32, device=device)

    # --- model ---
    eff_summary_dim = 15 if model_cfg.rich_summary else model_cfg.summary_dim
    model = DocLessFake(
        in_dim=16,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        dropout=model_cfg.dropout,
        kp=model_cfg.kp,
        ke=model_cfg.ke,
        use_summary=model_cfg.use_summary,
        use_structure=model_cfg.use_structure,
        summary_dim=eff_summary_dim,
    ).to(device)
    ssl_head = CascadeMLM(hidden_dim=model_cfg.hidden_dim).to(device)
    q_hat = q_hat.to(device)
    summary_params = set(model.summary_proj.parameters())
    base_params = [p for p in model.parameters() if p not in summary_params] + list(ssl_head.parameters())
    param_groups = [{"params": base_params, "lr": train_cfg.learning_rate}]
    if model_cfg.use_summary:
        param_groups.append({"params": list(summary_params), "lr": train_cfg.summary_lr})
    opt = torch.optim.Adam(param_groups)

    # --- (a) pretrain ---
    for ep in range(train_cfg.pretrain_epochs):
        model.train()
        opt.zero_grad()
        node_repr = model.encoder(h_tr.x, h_tr.edge_index, h_tr.edge_type, h_tr.node_type)
        batch = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4,
                                     seed=seed * 100 + ep, label_free=train_cfg.ssl_label_free)
        batch.anchor_idx = batch.anchor_idx.to(device)
        batch.positive_idx = batch.positive_idx.to(device)
        batch.negative_idx = batch.negative_idx.to(device)
        loss = cascade_ssl_loss(node_repr, ssl_head, batch)
        loss.backward()
        opt.step()
        if verbose:
            print(f"[pre {ep+1}/{train_cfg.pretrain_epochs}] ssl={loss.item():.4f}")

    # --- (b) finetune with annealed KL ---
    anneal_steps = max(1, int(train_cfg.finetune_epochs * train_cfg.kl_anneal_frac))
    for ep in range(train_cfg.finetune_epochs):
        lam_kl = train_cfg.lambda_kl_max * min(1.0, ep / anneal_steps)

        model.train()
        opt.zero_grad()
        out_tr = model(h_tr, p_tr, e_tr, q_hat)
        # supervised CE on labeled (here, all training stories)
        if ce_sample_w is not None:
            ce = (F.cross_entropy(out_tr["logits_do"], y_tr, reduction="none") * ce_sample_w).mean()
        else:
            ce = F.cross_entropy(out_tr["logits_do"], y_tr)
        # SSL on the same forward (reuse node_repr)
        batch = mask_cascade_slices(h_tr, tr, mask_ratio=0.3, num_negatives=4,
                                     seed=seed * 1000 + ep, label_free=train_cfg.ssl_label_free)
        batch.anchor_idx = batch.anchor_idx.to(device)
        batch.positive_idx = batch.positive_idx.to(device)
        batch.negative_idx = batch.negative_idx.to(device)
        l_ssl = cascade_ssl_loss(out_tr["node_repr"], ssl_head, batch)
        # (batch above already uses label_free per train_cfg)
        # KL between f_obs and f_do (used on the SAME set here; in M5 a true
        # unlabeled stream will be wired in)
        p_do = out_tr["logits_do"].exp()
        p_obs_logsoft = out_tr["logits_obs"].log_softmax(-1)
        l_kl = F.kl_div(p_obs_logsoft, p_do, reduction="batchmean")

        loss = ce + train_cfg.lambda_ssl * l_ssl + lam_kl * l_kl
        loss.backward()
        opt.step()

        if verbose:
            print(
                f"[ft  {ep+1}/{train_cfg.finetune_epochs}] "
                f"ce={ce.item():.3f} ssl={l_ssl.item():.3f} kl={l_kl.item():.3f} "
                f"lam_kl={lam_kl:.3f}"
            )

    # --- (c) calibrate τ on val ---
    model.eval()
    with torch.no_grad():
        out_va = model(h_va, p_va, e_va, q_hat)
    tau = calibrate_threshold(out_va["delta"], train_cfg.coverage_budget)

    # --- (d) evaluate on test ---
    with torch.no_grad():
        out_te = model(h_te, p_te, e_te, q_hat)
    preds, keep = selective_predict(out_te["logits_obs"], out_te["logits_do"], tau)
    coverage = float(keep.float().mean().item())
    metrics = {
        "seed": seed,
        "tau": tau,
        "coverage": coverage,
        "acc_all": _accuracy(preds, y_te),
        "acc_kept": _accuracy(preds, y_te, mask=keep),
        "f1_all": _macro_f1(preds, y_te),
        "f1_kept": _macro_f1(preds[keep], y_te[keep]),
        "n_test": int(y_te.numel()),
        "preds": preds.detach().cpu().tolist(),
        "labels": y_te.detach().cpu().tolist(),
        # Probability-level adjusted/observed outputs for the inertness
        # diagnostic. Legacy JSON keys are retained for output compatibility.
        "p_do": out_te["logits_do"].exp()[:, 1].detach().cpu().tolist(),
        "p_obs": out_te["logits_obs"].softmax(-1)[:, 1].detach().cpu().tolist(),
        "argmax_obs": out_te["logits_obs"].argmax(-1).detach().cpu().tolist(),
        "argmax_do": out_te["logits_do"].argmax(-1).detach().cpu().tolist(),
    }
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--pretrain-epochs", type=int, default=2)
    ap.add_argument("--finetune-epochs", type=int, default=3)
    ap.add_argument("--n-stories", type=int, default=400)
    ap.add_argument("--source", choices=["synthetic", "coaid"], default="synthetic")
    ap.add_argument("--coaid-root", type=Path, default=Path("data/raw/coaid"))
    ap.add_argument("--coaid-balanced", action="store_true")
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--device", default=None, help="explicit device override, e.g. cpu/mps")
    ap.add_argument("--out", type=Path, default=Path("outputs/smoke.json"))
    args = ap.parse_args()

    device = select_device(args.device)

    data_cfg = DataConfig(
        source=args.source,
        n_stories=args.n_stories,
        coaid_root=str(args.coaid_root),
        coaid_balanced=args.coaid_balanced,
    )
    model_cfg = ModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )
    train_cfg = TrainConfig(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        num_seeds=args.seeds,
    )

    results = []
    for s in range(args.seeds):
        m = train_one_seed(s, data_cfg, model_cfg, train_cfg, device=device)
        print(json.dumps(m, indent=2))
        results.append(m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"results": results,
         "model": asdict(model_cfg),
         "train": asdict(train_cfg),
         "data": asdict(data_cfg),
         "device": str(device)},
        indent=2,
    ))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
