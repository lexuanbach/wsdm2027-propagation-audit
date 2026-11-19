"""Render figures from the JSON output of each runner.

Uses matplotlib (CPU only). Saves PNGs under <out_dir>/figures/.

Figures rendered:
    fig2_main_bar.png        Average F1 across datasets, with 95% bootstrap CIs.
                             Mirrors paper Fig 2 (without-text only — we don't have
                             a text path here).
    fig4_coverage.png        Coverage vs. F1_kept per dataset.
    figV_kp_sweep.png        F1 vs. K_P bin count per dataset.
    fig_ablation_bar.png     Ablation variant comparison per dataset.

Each figure refers to the runner JSON it consumes; missing files are
warned about and skipped, so the same script can be run incrementally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: Path):
    if not path.exists():
        print(f"  [skip] {path} missing")
        return None
    return json.loads(path.read_text())


def fig_main_bar(main_table: dict | None, out: Path) -> None:
    if not main_table:
        return
    cells = main_table.get("cells") or _agg_main(main_table.get("results", []))
    if not cells:
        return
    # group: dataset -> {method -> (mean, std)}
    by_ds: dict[str, dict] = {}
    methods_order = ["majority", "gcnfn", "upfd_prop", "docless"]
    for c in cells:
        by_ds.setdefault(c["dataset"], {})[c["method"]] = (c["f1_mean"], c["f1_std"])

    datasets = sorted(by_ds.keys())
    n_methods = len(methods_order)
    width = 0.8 / n_methods
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, m in enumerate(methods_order):
        means = [by_ds[d].get(m, (0, 0))[0] for d in datasets]
        stds = [by_ds[d].get(m, (0, 0))[1] for d in datasets]
        x = [j + i * width for j in range(len(datasets))]
        ax.bar(x, means, width=width, label=m, yerr=stds, capsize=3)
    ax.set_xticks([j + width * (n_methods - 1) / 2 for j in range(len(datasets))])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Macro-F1 (without text)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_title("Fig 2 — Main F1 (Tier 2 local)")
    fig.tight_layout()
    fig.savefig(out / "fig2_main_bar.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out / 'fig2_main_bar.png'}")


def _agg_main(results: list[dict]) -> list[dict]:
    by = {}
    for r in results:
        by.setdefault((r["dataset"], r["method"], r.get("regime", "")), []).append(r["f1_all"])
    out = []
    for (d, m, rg), v in by.items():
        mu = sum(v) / len(v)
        sd = (sum((x - mu) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 if len(v) > 1 else 0
        out.append(dict(dataset=d, method=m, regime=rg, f1_mean=mu, f1_std=sd, n_seeds=len(v)))
    return out


def fig_coverage(coverage: dict | None, out: Path) -> None:
    if not coverage:
        return
    curves = coverage.get("curves") or []
    by_ds = {}
    for r in curves:
        by_ds.setdefault(r["dataset"], []).append(r)
    fig, ax = plt.subplots(figsize=(6, 4))
    for d, pts in by_ds.items():
        pts.sort(key=lambda r: r["coverage_mean"])
        ax.plot([p["coverage_mean"] for p in pts],
                [p["f1_kept_mean"] for p in pts], marker="o", label=d)
    ax.set_xlabel("Coverage (kept / total)")
    ax.set_ylabel("Macro-F1 on kept stories")
    ax.set_title("Fig 4 — Coverage-accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig4_coverage.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out / 'fig4_coverage.png'}")


def fig_kp_sweep(confshift: dict | None, out: Path) -> None:
    if not confshift:
        return
    cells = confshift.get("cells") or []
    by_ds: dict[str, list[tuple[int, float, float]]] = {}
    for c in cells:
        by_ds.setdefault(c["dataset"], []).append((c["kp"], c["f1_mean"], c["f1_std"]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for d, pts in by_ds.items():
        pts.sort()
        kps = [p[0] for p in pts]
        means = [p[1] for p in pts]
        stds = [p[2] for p in pts]
        ax.errorbar(kps, means, yerr=stds, marker="o", capsize=3, label=d)
    ax.set_xlabel("K_P (= K_E) stratification bins")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Table V — Stratification sensitivity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figV_kp_sweep.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out / 'figV_kp_sweep.png'}")


def fig_ablation(ablation: dict | None, out: Path) -> None:
    if not ablation:
        return
    cells = ablation.get("cells") or []
    if not cells:
        return
    by_ds = {}
    variants_order = ["full", "no_ssl", "no_causal", "no_abstain"]
    for c in cells:
        by_ds.setdefault(c["dataset"], {})[c["variant"]] = (c["f1_mean"], c["f1_std"])
    datasets = sorted(by_ds.keys())
    n_var = len(variants_order)
    width = 0.8 / n_var
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, v in enumerate(variants_order):
        means = [by_ds[d].get(v, (0, 0))[0] for d in datasets]
        stds = [by_ds[d].get(v, (0, 0))[1] for d in datasets]
        x = [j + i * width for j in range(len(datasets))]
        ax.bar(x, means, width=width, label=v, yerr=stds, capsize=3)
    ax.set_xticks([j + width * (n_var - 1) / 2 for j in range(len(datasets))])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title("Table III — Ablation")
    fig.tight_layout()
    fig.savefig(out / "fig_ablation_bar.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out / 'fig_ablation_bar.png'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("outputs/tier2"))
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    main_table = _load(args.results_dir / "main_table" / "main_table_agg.json")
    ablation = _load(args.results_dir / "ablation" / "ablation_agg.json")
    coverage = _load(args.results_dir / "coverage" / "coverage_curve.json")
    confshift = _load(args.results_dir / "confshift" / "confshift_agg.json")

    # main_table comes as a list of cells; wrap to match the helper
    fig_main_bar({"cells": main_table} if isinstance(main_table, list) else main_table, args.out)
    fig_ablation({"cells": ablation} if isinstance(ablation, list) else ablation, args.out)
    fig_coverage({"curves": coverage} if isinstance(coverage, list) else coverage, args.out)
    fig_kp_sweep({"cells": confshift} if isinstance(confshift, list) else confshift, args.out)


if __name__ == "__main__":
    main()
