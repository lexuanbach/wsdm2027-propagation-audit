"""Aggregate the per-runner JSON outputs into a single markdown report.

Used by `make tier2-report` after all four runners have finished.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stats import paired_bootstrap


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("outputs/tier2"))
    ap.add_argument("--out", type=Path, default=Path("results/tier2.md"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    main_table = _load(args.results_dir / "main_table" / "main_table.json")
    main_agg = _load(args.results_dir / "main_table" / "main_table_agg.json")
    ablation = _load(args.results_dir / "ablation" / "ablation.json")
    ablation_agg = _load(args.results_dir / "ablation" / "ablation_agg.json")
    coverage = _load(args.results_dir / "coverage" / "coverage_curve.json")
    confshift_agg = _load(args.results_dir / "confshift" / "confshift_agg.json")

    out = ["# Tier 2 — Measured results\n",
           "Real, measured numbers from the full local Tier-2 sweep.",
           "Hardware: Apple Silicon (arm64), 128 GB RAM, no CUDA.\n"]

    # --- Table I shape ---
    out.append("## Main table — F1 ± std across seeds")
    if main_agg:
        datasets = sorted({c["dataset"] for c in main_agg})
        methods = ["majority", "gcnfn", "upfd_prop", "docless"]
        header = "| Method | " + " | ".join(datasets) + " |"
        sep = "|---|" + "|".join(["---"] * len(datasets)) + "|"
        out += ["", header, sep]
        for m in methods:
            row = [m]
            for d in datasets:
                cells = [c for c in main_agg if c["method"] == m and c["dataset"] == d]
                if cells:
                    c = cells[0]
                    row.append(f"{c['f1_mean']:.3f} ± {c['f1_std']:.3f}")
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")

        # paired-bootstrap docless vs gcnfn per dataset
        if main_table:
            out.append("### Paired-bootstrap, DocLessFake vs GCNFN (per-seed F1 deltas)")
            out += ["", "| Dataset | Δ F1 (docless − gcnfn) | p-value | 95% CI |",
                    "|---|---|---|---|"]
            results = main_table.get("results", [])
            for d in datasets:
                docless_f1 = sorted(
                    [r["f1_all"] for r in results if r["dataset"] == d and r["method"] == "docless"]
                )
                gcnfn_f1 = sorted(
                    [r["f1_all"] for r in results if r["dataset"] == d and r["method"] == "gcnfn"]
                )
                n = min(len(docless_f1), len(gcnfn_f1))
                if n < 2:
                    continue
                # paired_bootstrap is loss-oriented (lower is better). Convert F1 to "loss" by 1-F1.
                import numpy as np
                la = 1 - np.array(docless_f1[:n])
                lb = 1 - np.array(gcnfn_f1[:n])
                r = paired_bootstrap(la, lb, n_resamples=10_000, seed=0)
                # mean_diff is mean(lb - la) = mean(F1_a - F1_b) when reversed
                out.append(
                    f"| {d} | {r.mean_diff:+.3f} | {r.p_value:.3f} | "
                    f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] |"
                )
            out.append("")

    else:
        out += ["", "_(main_table not yet run)_", ""]

    # --- Ablation ---
    out.append("## Ablation (DocLessFake variants)")
    if ablation_agg:
        datasets = sorted({c["dataset"] for c in ablation_agg})
        variants = ["full", "no_ssl", "no_causal", "no_abstain"]
        header = "| Variant | " + " | ".join(datasets) + " |"
        sep = "|---|" + "|".join(["---"] * len(datasets)) + "|"
        out += ["", header, sep]
        for v in variants:
            row = [v]
            for d in datasets:
                cells = [c for c in ablation_agg if c["variant"] == v and c["dataset"] == d]
                if cells:
                    c = cells[0]
                    row.append(f"{c['f1_mean']:.3f} ± {c['f1_std']:.3f}")
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    else:
        out += ["", "_(ablation not yet run)_", ""]

    # --- Coverage ---
    out.append("## Coverage-accuracy curve")
    if coverage:
        rows = sorted(coverage, key=lambda r: (r["dataset"], r["budget"]))
        out += ["", "| Dataset | Budget | Coverage | F1 kept |", "|---|---|---|---|"]
        for r in rows:
            out.append(
                f"| {r['dataset']} | {r['budget']:.2f} | "
                f"{r['coverage_mean']:.3f} | {r['f1_kept_mean']:.3f} |"
            )
        out.append("")
    else:
        out += ["", "_(coverage not yet run)_", ""]

    # --- Confshift ---
    out.append("## Confounder-shift sensitivity (K_P, K_E sweep)")
    if confshift_agg:
        datasets = sorted({c["dataset"] for c in confshift_agg})
        kps = sorted({c["kp"] for c in confshift_agg})
        header = "| K_P=K_E | " + " | ".join(datasets) + " |"
        sep = "|---|" + "|".join(["---"] * len(datasets)) + "|"
        out += ["", header, sep]
        for kp in kps:
            row = [str(kp)]
            for d in datasets:
                cells = [c for c in confshift_agg if c["dataset"] == d and c["kp"] == kp]
                if cells:
                    c = cells[0]
                    row.append(f"{c['f1_mean']:.3f} ± {c['f1_std']:.3f}")
                else:
                    row.append("—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    else:
        out += ["", "_(confshift not yet run)_", ""]

    out.append("\n## Figures")
    out.append("Rendered by `python -m docless.eval.make_figures` under `results/figures/`.")

    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
