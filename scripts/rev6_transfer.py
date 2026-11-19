"""Cross-dataset (cross-corpus / cross-domain) transfer matrix: train a summary model on corpus A, test it on corpus B.

Models: PropMLP (7-d), PropMLP+ (15-d), RandomForest (7-d / 15-d), val-tuned
HistGBT (7-d / 15-d). Features are the same cascade-summary vectors as the main
paper; standardisation statistics come from the *source* training split only.
For each (A, B, seed) the model is trained on A's training split (seed-specific
stratified 80/10/10 as in the main tables; HistGBT tunes on A's validation split)
and evaluated on B's test split, so the diagonal reproduces the in-domain
protocol and off-diagonal cells are zero-shot transfer. We report macro-F1,
balanced accuracy, and the class-prior-only floor (B's majority class).

Outputs: outputs/r11/transfer/transfer.json and transfer_rows.tex.
Usage: PYTHONPATH=src python3 scripts/rev6_transfer.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from docless.baselines.propmlp import _MLP, _features  # noqa: E402
from docless.eval.stats import macro_f1  # noqa: E402
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r11" / "transfer"
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid": dict(source="coaid", balanced=True, data_root="data/raw/coaid"),
    "recovery": dict(source="recovery", balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop": dict(source="gossipcop", balanced=True, data_root="data/raw/fakenewsnet", max_per_class=1500),
}
GRID = [dict(learning_rate=lr, max_leaf_nodes=ml, l2_regularization=l2)
        for lr in (0.05, 0.1) for ml in (15, 31) for l2 in (0.0, 1.0)]


def tuned_histgbt(x_tr, y_tr, x_va, y_va):
    best, best_f1 = None, -1.0
    for hp in GRID:
        m = HistGradientBoostingClassifier(max_iter=300, random_state=0, **hp).fit(x_tr, y_tr)
        f1 = macro_f1(m.predict(x_va), y_va)
        if f1 > best_f1:
            best_f1, best = f1, hp
    return HistGradientBoostingClassifier(max_iter=300, random_state=0, **best).fit(
        np.vstack([x_tr, x_va]), np.concatenate([y_tr, y_va]))


def train_mlp(x_tr, y_tr, seed, hidden=64, epochs=25, lr=1e-2):
    torch.manual_seed(seed)
    xt = torch.tensor(x_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.long)
    model = _MLP(in_dim=xt.shape[1], hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        F.cross_entropy(model(xt), yt).backward(); opt.step()
    model.eval()
    return model


def balanced_accuracy(preds, labels):
    accs = []
    for c in np.unique(labels):
        m = labels == c
        accs.append(float((preds[m] == c).mean()))
    return float(np.mean(accs))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # cache splits + features per (dataset, seed)
    cache = {}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        for s in SEEDS:
            _set_seed(s)
            casc = _load_cascades(dcfg, s)
            tr, va, te = _stratified_split(casc, dcfg.split_ratios, s)
            cache[(name, s)] = {
                "tr": (_features(tr), _features(tr, rich=True), np.array([c.label for c in tr])),
                "va": (_features(va), _features(va, rich=True), np.array([c.label for c in va])),
                "te": (_features(te), _features(te, rich=True), np.array([c.label for c in te])),
            }
        print(f"[cache] {name} loaded", flush=True)
    models = ["propmlp7", "propmlp15", "rf7", "rf15", "histgbt7", "histgbt15"]
    result = {m: {a: {b: [] for b in DATASETS} for a in DATASETS} for m in models}
    bal = {m: {a: {b: [] for b in DATASETS} for a in DATASETS} for m in models}
    floor = {b: [] for b in DATASETS}
    for s in SEEDS:
        for a in DATASETS:
            x7, x15, y = cache[(a, s)]["tr"]
            v7, v15, yv = cache[(a, s)]["va"]
            sc7, sc15 = StandardScaler().fit(x7), StandardScaler().fit(x15)
            fitted = {
                "propmlp7": ("mlp", train_mlp(sc7.transform(x7), y, s), sc7),
                "propmlp15": ("mlp", train_mlp(sc15.transform(x15), y, s), sc15),
                "rf7": ("sk", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(sc7.transform(x7), y), sc7),
                "rf15": ("sk", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=s, n_jobs=1).fit(sc15.transform(x15), y), sc15),
                "histgbt7": ("sk", tuned_histgbt(sc7.transform(x7), y, sc7.transform(v7), yv), sc7),
                "histgbt15": ("sk", tuned_histgbt(sc15.transform(x15), y, sc15.transform(v15), yv), sc15),
            }
            for b in DATASETS:
                t7, t15, yt = cache[(b, s)]["te"]
                if a == list(DATASETS)[0]:
                    maj = np.bincount(cache[(b, s)]["tr"][2]).argmax()
                    floor[b].append(macro_f1(np.full_like(yt, maj), yt))
                for m, (kind, model, sc) in fitted.items():
                    xb = sc.transform(t7 if m.endswith("7") else t15)
                    if kind == "mlp":
                        with torch.no_grad():
                            p = model(torch.tensor(xb, dtype=torch.float32)).argmax(-1).numpy()
                    else:
                        p = model.predict(xb)
                    result[m][a][b].append(macro_f1(p, yt))
                    bal[m][a][b].append(balanced_accuracy(p, yt))
        print(f"[seed {s}] done", flush=True)
    agg = {m: {a: {b: {"f1_mean": round(float(np.mean(v)), 4), "f1_std": round(float(np.std(v)), 4),
                       "bal_acc_mean": round(float(np.mean(bal[m][a][b])), 4)}
                   for b, v in row.items()} for a, row in mat.items()} for m, mat in result.items()}
    agg["majority_floor_f1"] = {b: round(float(np.mean(v)), 4) for b, v in floor.items()}
    # transfer gap: in-domain minus mean off-diagonal, per model and source
    gaps = {}
    for m in models:
        gaps[m] = {}
        for a in DATASETS:
            diag = agg[m][a][a]["f1_mean"]
            off = [agg[m][a][b]["f1_mean"] for b in DATASETS if b != a]
            gaps[m][a] = {"in_domain": diag, "mean_off_diagonal": round(float(np.mean(off)), 4),
                          "gap": round(diag - float(np.mean(off)), 4)}
    agg["transfer_gap"] = gaps
    (OUT / "transfer.json").write_text(json.dumps(agg, indent=2))
    # tex rows: one block per model, rows = source, cols = target
    names = {"coaid": "CoAID", "recovery": "ReCOVery", "politifact": "PolitiFact", "gossipcop": "GossipCop"}
    lines = []
    for m, label in [("histgbt15", "HistGBT (15d)"), ("rf15", "RandomForest (15d)"), ("propmlp15", "PropMLP+ (15d)"), ("propmlp7", "PropMLP (7d)")]:
        for i, a in enumerate(DATASETS):
            cells = []
            for b in DATASETS:
                v = agg[m][a][b]["f1_mean"]
                cells.append(f"\\textbf{{{v:.3f}}}" if a == b else f"{v:.3f}")
            lines.append(f"{label if i == 0 else ''} & {names[a]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    lines.append("\\multicolumn{2}{l}{majority floor (target)} & " + " & ".join(f"{agg['majority_floor_f1'][b]:.3f}" for b in DATASETS) + " \\\\")
    (OUT / "transfer_rows.tex").write_text("\n".join(lines) + "\n")
    for m in models:
        print(m, {a: {b: agg[m][a][b]["f1_mean"] for b in DATASETS} for a in DATASETS})
    print("floor", agg["majority_floor_f1"])


if __name__ == "__main__":
    main()
