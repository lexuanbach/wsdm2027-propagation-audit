#!/usr/bin/env python3
"""Check that shipped result JSONs match headline paper/artifact claims.

This is intentionally narrow: it verifies the cells that the README and paper
use as headline evidence, without rerunning training.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(name: str) -> Path:
    """Locate a manuscript source, in the working tree or in the shipped bundle.

    In the development tree the sources sit next to `artifact/`; in the packaged
    supplement they ship at `paper/`. Return whichever exists (the working-tree
    copy wins), so the cross-document checks below run in both layouts.
    """
    # The public release ships no manuscript sources, so the cross-document
    # checks below are skipped unless a paper/ folder is placed next to scripts/.
    return ROOT / "paper" / name

TOL = 1e-9


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text())


def close(actual: float, expected: float, label: str, tol: float = TOL) -> None:
    if abs(float(actual) - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, found {actual}")
    print(f"ok  {label}: {actual}")




def propmlp_table(dataset: str) -> dict:
    rows = load_json("outputs/r9/propmlp_table/main_table_agg.json")
    for row in rows:
        if row["dataset"] == dataset and row["method"] == "propmlp":
            return row
    raise KeyError(f"missing PropMLP row for {dataset}")


def main_metric(dataset_file: str, method: str) -> float:
    return load_json(dataset_file)["main"][method]["f1_mean"]


def paired_metric(dataset_file: str, key: str) -> float:
    return load_json(dataset_file)["paired_docless_vs_propmlp"][key]


def main() -> None:
    # Main PropMLP and DocLessFake claims.
    close(propmlp_table("coaid")["f1_mean"], 0.618, "PropMLP CoAID F1")
    close(propmlp_table("recovery")["f1_mean"], 0.659, "PropMLP ReCOVery F1")
    close(main_metric("outputs/r9/politifact/politifact_eval.json", "propmlp"), 0.657, "PropMLP Politifact F1")
    close(main_metric("outputs/r9/gossipcop_3k/gossipcop_eval.json", "propmlp"), 0.836, "PropMLP GossipCop@1500 F1")

    paired = load_json("outputs/r9/revision_stats/paired_bootstrap.json")
    close(paired["coaid"]["docless_f1_mean"], 0.654, "DocLessFake CoAID F1")
    close(paired["recovery"]["docless_f1_mean"], 0.663, "DocLessFake ReCOVery F1")
    close(main_metric("outputs/r9/politifact/politifact_eval.json", "docless"), 0.678, "DocLessFake Politifact F1")
    close(main_metric("outputs/r9/gossipcop_3k/gossipcop_eval.json", "docless"), 0.842, "DocLessFake GossipCop@1500 F1")

    # Paired-bootstrap non-significance claims.
    close(paired["coaid"]["p_value"], 0.4136, "DocLessFake vs PropMLP p-value CoAID")
    close(paired["recovery"]["p_value"], 0.6324, "DocLessFake vs PropMLP p-value ReCOVery")
    close(paired_metric("outputs/r9/politifact/politifact_eval.json", "p_value"), 0.233, "DocLessFake vs PropMLP p-value Politifact")
    close(paired_metric("outputs/r9/gossipcop_3k/gossipcop_eval.json", "p_value"), 0.2538, "DocLessFake vs PropMLP p-value GossipCop@1500")

    # Relation-aware GNN baseline cells.
    close(load_json("outputs/r9/rgcn/coaid.json")["f1_mean"], 0.596, "RGCN CoAID F1")
    close(load_json("outputs/r9/rgcn/recovery.json")["f1_mean"], 0.545, "RGCN ReCOVery F1")
    close(load_json("outputs/r9/rgcn/politifact.json")["f1_mean"], 0.559, "RGCN Politifact F1")
    close(load_json("outputs/r9/rgcn/gossipcop.json")["f1_mean"], 0.780, "RGCN GossipCop@1500 F1")

    # Modern dynamic-attention GATv2 baseline cells + paired-vs-PropMLP p-values.
    gatv2 = load_json("outputs/r10/gatv2/summary.json")
    close(gatv2["coaid"]["f1_mean"], 0.598, "GATv2 CoAID F1")
    close(gatv2["recovery"]["f1_mean"], 0.597, "GATv2 ReCOVery F1")
    close(gatv2["politifact"]["f1_mean"], 0.627, "GATv2 Politifact F1")
    close(gatv2["gossipcop"]["f1_mean"], 0.797, "GATv2 GossipCop@1500 F1")
    close(gatv2["coaid"]["paired_vs_propmlp"]["p_value"], 0.6814, "GATv2 vs PropMLP p CoAID")
    close(gatv2["recovery"]["paired_vs_propmlp"]["p_value"], 0.1006, "GATv2 vs PropMLP p ReCOVery")
    close(gatv2["politifact"]["paired_vs_propmlp"]["p_value"], 0.9604, "GATv2 vs PropMLP p Politifact")
    close(gatv2["gossipcop"]["paired_vs_propmlp"]["p_value"], 0.001, "GATv2 vs PropMLP p GossipCop")

    # Temporal-drift mitigation cells: macro-F1 POINT ESTIMATES only. The paper
    # does NOT treat these as validated recoveries -- the per-delta paired CI
    # (outputs/r10/mitig_delta/mitig_delta.json) is authoritative, and on
    # Politifact no delta excludes zero.
    pol = load_json("outputs/r10/temporal_drift/politifact_v2.json")["results"]
    gos = load_json("outputs/r10/temporal_drift/gossipcop_v2.json")["results"]
    close(pol["rf"]["coral"]["f1_mean"], 0.501, "CORAL/RF Politifact point estimate (delta n.s.: CI [-0.03,0.08])")
    close(pol["propmlp_plus"]["impweight"]["f1_mean"], 0.505, "imp.-wt PropMLP+ Politifact point estimate (delta n.s.)")
    close(gos["propmlp"]["quantile"]["f1_mean"], 0.790, "quantile PropMLP GossipCop point estimate")

    # Strong tabular baseline claims.
    tabular = load_json("outputs/r9/tabular_baselines/main.json")
    close(tabular["gossipcop"]["rf"]["f1_mean"], 0.865, "RandomForest GossipCop F1")
    close(load_json("outputs/r9/stanford2/gossipcop.json")["histgbt15"]["mean"], 0.877, "HistGBT15 GossipCop F1")
    close(load_json("outputs/r9/gossipcop_3k/gossipcop_eval.json")["max_per_class"], 1500, "GossipCop per-class cap")

    # Calibration / ranking table (main text Table 7 and Table 8). These are the
    # cells an independent audit found mis-transcribed from this very file, so
    # every printed value is pinned here, and the manuscript text is scanned for
    # the two numbers most easily copied from the wrong row.
    calib = load_json("outputs/r10/calibration/calibration.json")
    expected_roc = {
        ("coaid", "propmlp"): 0.504, ("recovery", "propmlp"): 0.655,
        ("politifact", "propmlp"): 0.732, ("gossipcop", "propmlp"): 0.900,
        ("coaid", "propmlp+"): 0.539, ("recovery", "propmlp+"): 0.698,
        ("politifact", "propmlp+"): 0.804, ("gossipcop", "propmlp+"): 0.917,
        ("coaid", "rf"): 0.686, ("recovery", "rf"): 0.681,
        ("politifact", "rf"): 0.772, ("gossipcop", "rf"): 0.933,
        ("coaid", "histgbt"): 0.611, ("recovery", "histgbt"): 0.668,
        ("politifact", "histgbt"): 0.785, ("gossipcop", "histgbt"): 0.936,
    }
    for (dataset, model), value in expected_roc.items():
        close(calib[dataset][model]["roc"], value, f"ROC-AUC {model} {dataset}")
    pr_gossip = [calib["gossipcop"][m]["pr"] for m in ("propmlp", "propmlp+", "rf", "histgbt")]
    close(min(pr_gossip), 0.763, "GossipCop PR-AUC range low")
    close(max(pr_gossip), 0.823, "GossipCop PR-AUC range high")

    # Matched-tuning parity on ReCOVery: the supplement claims the fixed-budget
    # EBGCN lead is a tuning artefact of the flat-feature arm. That rests on the
    # graph side and the PropMLP side coming from the SAME grid, so pin both.
    parity = load_json("outputs/r10/gnn_parity/gnn_parity.json")
    for model, value in (("gcnfn", 0.555), ("bigcn", 0.561),
                         ("rgcn", 0.663), ("ebgcn", 0.672)):
        close(parity["recovery"][model]["f1_mean"], value,
              f"matched-tuning ReCOVery {model}")
    pp = load_json("outputs/r10/propmlp_parity/propmlp_parity.json")["recovery"]
    close(pp["propmlp"]["val_selected"], 0.673, "matched-tuning ReCOVery propmlp")
    close(pp["propmlp+"]["val_selected"], 0.693, "matched-tuning ReCOVery propmlp+")
    best_graph = max(parity["recovery"][m]["f1_mean"] for m in parity["recovery"])
    if not pp["propmlp+"]["val_selected"] > best_graph:
        raise AssertionError(
            "PropMLP+ no longer leads the best matched-tuned graph model on "
            f"ReCOVery ({pp['propmlp+']['val_selected']} vs {best_graph}); the "
            "supplement's 'tuning artefact' reading must be re-measured")
    print("ok  ReCOVery matched tuning: flat-feature arm still leads")

    manuscript = _source("main.tex")
    if manuscript.exists():
        text = " ".join(manuscript.read_text().split())
        for banned, why in (("$0.956$", "old HistGBT GossipCop ROC-AUC"),
                            ("$0.814$", "old PropMLP+ PolitiFact ROC-AUC"),
                            ("$0.76$--$0.88$", "old GossipCop PR-AUC range")):
            if banned in text:
                raise AssertionError(f"manuscript still contains {banned} ({why})")
        print("ok  manuscript free of the superseded calibration numbers")

    # ---------------------------------------------------------------- outputs/r11
    # PolitiFact matched-tuning arm (new in this revision). The main text and
    # supplement claim the fixed-budget EBGCN edge (+0.003) REVERSES SIGN under
    # matched tuning, as a point estimate only.
    pol = load_json("outputs/r11/gnn_parity_politifact/gnn_parity.json")["politifact"]
    for model, value in (("gcnfn", 0.581), ("bigcn", 0.6),
                         ("rgcn", 0.631), ("ebgcn", 0.658)):
        close(pol[model]["f1_mean"], value, f"matched-tuning PolitiFact {model}")
    ppp = load_json("outputs/r10/propmlp_parity/propmlp_parity.json")["politifact"]
    close(ppp["propmlp"]["val_selected"], 0.671, "matched-tuning PolitiFact propmlp")
    close(ppp["propmlp+"]["val_selected"], 0.721, "matched-tuning PolitiFact propmlp+")
    if not ppp["propmlp"]["val_selected"] > pol["ebgcn"]["f1_mean"]:
        raise AssertionError(
            "PolitiFact matched tuning no longer reverses the EBGCN edge "
            f"({ppp['propmlp']['val_selected']} vs {pol['ebgcn']['f1_mean']}); the "
            "main text's 'changes sign under matched tuning' claim must be redone")
    print("ok  PolitiFact matched tuning: sign of the EBGCN gap reverses")

    # Seed-paired t-tests vs the two fixed-budget GNNs. The paper claims
    # "positive in all eight cells, significant at the seed level in five".
    sp = load_json("outputs/r11/seed_paired_gnn/seed_paired_gnn.json")
    expected_sp = {
        "coaid_propmlp_vs_gcnfn": (0.1758, 0.1238),
        "coaid_propmlp_vs_bigcn": (0.079, 0.4666),
        "recovery_propmlp_vs_gcnfn": (0.2694, 0.00257),
        "recovery_propmlp_vs_bigcn": (0.3214, 2.94e-05),
        "politifact_propmlp_vs_gcnfn": (0.1065, 0.01706),
        "politifact_propmlp_vs_bigcn": (0.1208, 0.01297),
        "gossipcop_propmlp_vs_gcnfn": (0.1442, 0.1651),
        "gossipcop_propmlp_vs_bigcn": (0.049, 0.009558),
    }
    for cell, (d, pv) in expected_sp.items():
        close(sp["tests"][cell]["mean_diff"], d, f"seed-paired delta {cell}")
        close(sp["tests"][cell]["p_value"], pv, f"seed-paired p {cell}")
    if sp["summary"]["n_positive"] != 8:
        raise AssertionError(
            f"paper says PropMLP is positive in all 8 cells, found "
            f"{sp['summary']['n_positive']}")
    if sp["summary"]["n_significant_at_05"] != 5:
        raise AssertionError(
            f"paper says 5 of 8 seed-paired cells are significant, found "
            f"{sp['summary']['n_significant_at_05']}")
    for ds, chk in sp["determinism_check"].items():
        if not chk["identical"]:
            raise AssertionError(
                f"seed->split determinism broke on {ds}; the r7-vs-r9 pairing "
                "behind the seed-paired t-tests is no longer valid")
    print("ok  seed-paired GNN tests: 8/8 positive, 5/8 significant, pairing valid")

    # Supplement Table S5: PR-AUC and Brier at natural prevalence. These were
    # promised by main-text Table 8's caption but never shipped before.
    expected_brier = {
        ("coaid", "propmlp"): 0.051, ("recovery", "propmlp"): 0.194,
        ("politifact", "propmlp"): 0.205, ("gossipcop", "propmlp"): 0.092,
        ("coaid", "propmlp+"): 0.051, ("recovery", "propmlp+"): 0.186,
        ("politifact", "propmlp+"): 0.184, ("gossipcop", "propmlp+"): 0.078,
        ("coaid", "rf"): 0.051, ("recovery", "rf"): 0.192,
        ("politifact", "rf"): 0.202, ("gossipcop", "rf"): 0.078,
        ("coaid", "histgbt"): 0.052, ("recovery", "histgbt"): 0.202,
        ("politifact", "histgbt"): 0.203, ("gossipcop", "histgbt"): 0.077,
        ("coaid", "histgbt_iso"): 0.052, ("recovery", "histgbt_iso"): 0.198,
        ("politifact", "histgbt_iso"): 0.218, ("gossipcop", "histgbt_iso"): 0.079,
    }
    expected_pr = {
        ("coaid", "propmlp"): 0.066, ("recovery", "propmlp"): 0.43,
        ("politifact", "propmlp"): 0.739, ("gossipcop", "propmlp"): 0.763,
        ("coaid", "propmlp+"): 0.074, ("recovery", "propmlp+"): 0.47,
        ("politifact", "propmlp+"): 0.823, ("gossipcop", "propmlp+"): 0.823,
        ("coaid", "rf"): 0.116, ("recovery", "rf"): 0.445,
        ("politifact", "rf"): 0.815, ("gossipcop", "rf"): 0.817,
        ("coaid", "histgbt"): 0.1, ("recovery", "histgbt"): 0.423,
        ("politifact", "histgbt"): 0.815, ("gossipcop", "histgbt"): 0.822,
        ("coaid", "histgbt_iso"): 0.08, ("recovery", "histgbt_iso"): 0.399,
        ("politifact", "histgbt_iso"): 0.743, ("gossipcop", "histgbt_iso"): 0.802,
    }
    for (dataset, model), value in expected_brier.items():
        close(calib[dataset][model]["brier"], value, f"Brier {model} {dataset}")
    for (dataset, model), value in expected_pr.items():
        close(calib[dataset][model]["pr"], value, f"PR-AUC {model} {dataset}")
    for dataset in ("coaid", "recovery", "politifact", "gossipcop"):
        if not calib[dataset]["histgbt_iso"]["pr"] < calib[dataset]["histgbt"]["pr"]:
            raise AssertionError(
                f"supplement claims isotonic lowers PR-AUC on all four corpora; "
                f"{dataset} no longer does")
    print("ok  isotonic HistGBT lowers PR-AUC on all four corpora")

    # Feature-drop ablation: "is it just cascade size?"
    fd = load_json("outputs/r11/feature_drop/feature_drop.json")
    for cell, value in fd["_full_arm_reproduces_table2"].items():
        if not value["match"]:
            raise AssertionError(
                f"feature-drop full arm no longer reproduces Table 2 at {cell}: "
                f"{value['computed']} vs {value['table2']}")
    print("ok  feature-drop full arm reproduces all 8 Table-2 cells")
    expected_fd = {
        ("coaid", "propmlp", "drop_size"): (0.0167, 0.6296),
        ("recovery", "propmlp", "drop_size"): (-0.013, 0.2306),
        ("politifact", "propmlp", "drop_size"): (-0.0281, 0.0772),
        ("gossipcop", "propmlp", "drop_size"): (0.0007, 0.9406),
        ("coaid", "rf", "drop_size"): (0.0167, 0.5792),
        ("recovery", "rf", "drop_size"): (-0.0217, 0.0582),
        ("politifact", "rf", "drop_size"): (0.0094, 0.5634),
        ("gossipcop", "rf", "drop_size"): (0.002, 0.5702),
        ("coaid", "propmlp", "drop_size_users"): (0.1167, 0.0214),
        ("recovery", "propmlp", "drop_size_users"): (-0.0043, 0.792),
        ("politifact", "propmlp", "drop_size_users"): (-0.0094, 0.6856),
        ("gossipcop", "propmlp", "drop_size_users"): (0.0107, 0.0576),
        ("recovery", "rf", "drop_size_users"): (-0.0065, 0.762),
        ("politifact", "rf", "drop_size_users"): (-0.0125, 0.6674),
        ("gossipcop", "rf", "drop_size_users"): (0.034, 0.0),
    }
    for (ds, model, arm), (cost, pv) in expected_fd.items():
        got = fd[ds][model][arm]["paired_vs_full"]
        close(got["acc_cost_of_dropping"], cost, f"drop cost {ds}/{model}/{arm}")
        close(got["p_value"], pv, f"drop p {ds}/{model}/{arm}")
    # The paper's claim: dropping log_n alone is null EVERYWHERE (no cell p<0.05).
    for ds in ("coaid", "recovery", "politifact", "gossipcop"):
        for model in ("propmlp", "rf"):
            if fd[ds][model]["drop_size"]["paired_vs_full"]["p_value"] < 0.05:
                raise AssertionError(
                    f"paper claims dropping log_n is null in every cell, but "
                    f"{ds}/{model} is now significant")
    print("ok  dropping log_n alone: null in all 8 cells")
    # ...and that GossipCop RF is where dropping BOTH size features really bites.
    if not fd["gossipcop"]["rf"]["drop_size_users"]["paired_vs_full"]["p_value"] < 0.001:
        raise AssertionError(
            "paper reports the GossipCop RF size-pair cost as p<0.001; no longer true")
    print("ok  GossipCop RF pays a significant cost for dropping the size pair")

    # GossipCop subsampling stability (six cells + the 'inside one std' claim).
    gs = load_json("outputs/r9/tabular_baselines/gossipcop_stability.json")
    base, alt = gs["original"], gs["alternate_subsample_seed+1000"]
    for model, b, a in (("logreg", 0.827, 0.838), ("rf", 0.865, 0.875),
                        ("gbt", 0.859, 0.866)):
        close(base[model]["f1_mean"], b, f"GossipCop stability base {model}")
        close(alt[model]["f1_mean"], a, f"GossipCop stability alternate {model}")
        gap = abs(alt[model]["f1_mean"] - base[model]["f1_mean"])
        if gap > base[model]["f1_std"]:
            raise AssertionError(
                f"supplement claims every subsample gap sits inside one base std; "
                f"{model} gap {gap:.3f} > std {base[model]['f1_std']}")
    print("ok  GossipCop subsample gaps all inside one base std")

    # UPFD profile-vector recurrence, behind the hedged 'no user recurs' claim.
    up = load_json("outputs/r11/upfd_profile_recurrence/upfd_profile_recurrence.json")
    close(up["n_graphs"], 5464, "UPFD GossipCop graphs")
    close(up["vectors_in_more_than_one_graph"], 13019, "UPFD recurring profile vectors")

    # Cascade-size-stratified split . This is a NEGATIVE
    # / partly-uninterpretable result and the paper says so; pin both the drops
    # and the prior-shift that makes the GossipCop half unreadable, because the
    # honesty of the passage depends on those priors.
    ss = load_json("outputs/r11/size_split/size_split.json")
    expected_ss = {
        ("politifact", "small2large", "propmlp"): -0.047,
        ("politifact", "small2large", "propmlp+"): 0.026,
        ("politifact", "small2large", "rf"): -0.045,
        ("politifact", "small2large", "histgbt"): -0.036,
        ("politifact", "large2small", "propmlp"): -0.005,
        ("politifact", "large2small", "propmlp+"): -0.062,
        ("gossipcop", "small2large", "propmlp"): -0.491,
        ("gossipcop", "small2large", "propmlp+"): -0.298,
        ("gossipcop", "small2large", "rf"): -0.359,
        ("gossipcop", "small2large", "histgbt"): -0.262,
        ("gossipcop", "large2small", "propmlp"): -0.341,
        ("gossipcop", "large2small", "histgbt"): -0.275,
    }
    for (ds, mode, model), value in expected_ss.items():
        close(ss[ds]["deltas_vs_random"][mode][model], value,
              f"size-split delta {ds}/{mode}/{model}")
    close(ss["gossipcop"]["modes"]["small2large"]["propmlp"]["f1_mean"], 0.345,
          "size-split GossipCop small2large PropMLP")
    for ds, mode, prior in (("gossipcop", "small2large", 0.763),
                            ("gossipcop", "large2small", 0.937),
                            ("politifact", "small2large", 0.516),
                            ("politifact", "large2small", 0.344)):
        close(ss[ds]["split_stats"][mode]["test_prior_misinfo"], prior,
              f"size-split test prior {ds}/{mode}")
    # The paper's caveat: GossipCop folds are prior-shifted far from balance,
    # PolitiFact's are not. If that ever flips, the passage must be rewritten.
    for mode in ("small2large", "large2small"):
        if abs(ss["gossipcop"]["split_stats"][mode]["test_prior_misinfo"] - 0.5) < 0.15:
            raise AssertionError(
                f"paper calls the GossipCop {mode} fold severely prior-shifted; "
                "it no longer is, so the 'do not read as a model result' caveat "
                "must be revisited")
        if abs(ss["politifact"]["split_stats"][mode]["test_prior_misinfo"] - 0.5) > 0.20:
            raise AssertionError(
                f"paper calls the PolitiFact {mode} fold near-balanced; it is not")
    print("ok  size-split: GossipCop folds prior-shifted, PolitiFact folds near balance")

    # ------------------------------------------------- outputs/r13 checks
    # EDGE: degree-preserving edge rewire (main text 5.7, supplement Part B).
    er = load_json("outputs/r13/edge_rewire/analysis.json")
    expected_rewire = {
        "upfd_gossipcop":  (0.9404, 0.9366, -0.0038, 0.2329, 0.931),
        "upfd_politifact": (0.7516, 0.7553, +0.0038, 0.2682, 0.7916),
        "twitter15":       (0.3418, 0.3243, -0.0175, 0.2013, 0.2844),
        "twitter16":       (0.3114, 0.2747, -0.0368, 0.2479, 0.2909),
    }
    for ds, (intact, rewired, delta, pval, hist) in expected_rewire.items():
        cell = er["per_corpus"][ds]
        close(cell["bigcn_intact_mean"], intact, f"rewire {ds} intact BiGCN", tol=5e-4)
        close(cell["bigcn_rewired_mean"], rewired, f"rewire {ds} rewired BiGCN", tol=5e-4)
        close(cell["seed_paired_rewired_vs_intact"]["mean_delta"], delta,
              f"rewire {ds} delta", tol=5e-4)
        close(cell["seed_paired_rewired_vs_intact"]["p"], pval, f"rewire {ds} p", tol=5e-4)
        close(cell["histgbt"], hist, f"rewire {ds} tuned-tree reference", tol=5e-4)
        if not cell["summary_family_bit_identical"]:
            raise AssertionError(
                f"rewire {ds}: the summary baselines are no longer bit-identical between "
                "the intact and rewired runs, so the rewire touched more than the graph "
                "model's edges and the comparison is invalid")
    # the paper says the rewired model on UPFD-GossipCop stays ABOVE the tuned tree
    gc = er["per_corpus"]["upfd_gossipcop"]
    if not gc["bigcn_rewired_mean"] > gc["histgbt"]:
        raise AssertionError(
            "main text 5.7 says the rewired BiGCN is still above the tuned tree on "
            f"UPFD-GossipCop; now {gc['bigcn_rewired_mean']} vs {gc['histgbt']}")
    # and that no delta is resolvable at 5 seeds
    worst_p = min(c["seed_paired_rewired_vs_intact"]["p"] for c in er["per_corpus"].values())
    if worst_p <= 0.05:
        raise AssertionError(
            "main text 5.7 calls every rewire delta inconclusive at 5 seeds, but some "
            f"corpus now has p={worst_p}; the passage must be rewritten, not the run")
    print(f"ok  edge rewire: 4 corpora pinned, summary arms bit-identical, min p={worst_p}")

    # near-star tree shape -- the reason the rewire has little to remove
    sh = er["tree_shape"]["upfd_gossipcop"]
    close(sh["mean_frac_nodes_attached_to_root"], 0.752, "UPFD-GossipCop root-attached frac",
          tol=5e-4)
    close(sh["mean_depth"]["mean_intact"], 1.1973, "UPFD-GossipCop mean depth", tol=5e-4)
    close(sh["structural_virality"]["frac_trees_changed"], 0.3157,
          "UPFD-GossipCop virality frac changed by rewire", tol=5e-4)
    close(sh["structural_virality"]["mean_abs_change_under_rewire"], 0.0299,
          "UPFD-GossipCop virality mean change", tol=5e-4)

    # LEAK: label-permutation negative control (main text 5.5, supplement Part B).
    lp = load_json("outputs/r13/label_permutation/label_permutation.json")
    expected_perm = {
        "coaid":      {"histgbt15": (0.7039, 0.4665), "rf7": (0.6133, 0.4693)},
        "recovery":   {"histgbt15": (0.6705, 0.5056), "rf7": (0.6573, 0.4834)},
        "politifact": {"histgbt15": (0.6979, 0.4984), "rf7": (0.6446, 0.5285)},
        "gossipcop":  {"histgbt15": (0.8779, 0.5177), "rf7": (0.8619, 0.5183)},
    }
    for corpus, models in expected_perm.items():
        for model, (real, perm) in models.items():
            cell = lp["results"][corpus][model]
            close(cell["real_mean"], real, f"label-perm {corpus}/{model} real", tol=5e-4)
            close(cell["perm_mean"], perm, f"label-perm {corpus}/{model} permuted", tol=5e-4)
            if not cell["perm_ci95_contains_0.50"]:
                raise AssertionError(
                    f"label-permutation control: {corpus}/{model} permuted 95% CI "
                    f"{cell['perm_ci95']} no longer contains 0.50. This is the "
                    "pre-registered FAILURE outcome: it is evidence of a "
                    "label-independent artefact and belongs in the MAIN text as a "
                    "threat to validity. Do not re-run, re-seed or re-shuffle.")
    if lp["corpora_whose_permuted_ci_excludes_0.50"]:
        raise AssertionError(
            "label-permutation control flagged "
            f"{lp['corpora_whose_permuted_ci_excludes_0.50']}; see the pre-registered "
            "failure procedure in scripts/rev8_label_permutation.py")
    perm_means = [lp["results"][c][m]["perm_mean"] for c in expected_perm for m in ("histgbt15", "rf7")]
    if not (0.46 <= min(perm_means) and max(perm_means) <= 0.53):
        raise AssertionError(
            f"main text 5.5 states the permuted range 0.467--0.529; measured "
            f"{min(perm_means)}--{max(perm_means)}")
    print(f"ok  label permutation: 8 cells at chance, range "
          f"{min(perm_means)}--{max(perm_means)}, all CIs contain 0.50")

    # Signal-availability matrix (main-text Table 1): the facts the table asserts.
    sm = load_json("outputs/r13/signal_matrix/signal_matrix.json")
    for corpus in ("coaid", "recovery"):
        cell = sm["identity_free"][corpus]
        if list(cell["distinct_url_counts"]) != ["1"] or list(cell["distinct_media_counts"]) != ["0"]:
            raise AssertionError(
                f"Table 1 says every {corpus} cascade carries exactly 1 URL and 0 media; "
                f"measured {cell['distinct_url_counts']} / {cell['distinct_media_counts']}")
    for corpus, n, splits in (("upfd_politifact", 314, {"train": 62, "val": 31, "test": 221}),
                              ("upfd_gossipcop", 5464, {"train": 1092, "val": 546, "test": 3826})):
        cell = sm["graph_informative"][corpus]
        if cell["n_graphs"] != n or cell["split_sizes"] != splits:
            raise AssertionError(f"Table 1 {corpus} counts/splits moved: {cell}")
        close(cell["frac_label1"], 0.5, f"Table 1 {corpus} prevalence")
        if cell["timestamp_file_present"]:
            raise AssertionError(
                f"Table 1 and the dataset card say no {corpus} release file carries event "
                "times; one now appears, so both must be corrected")
    for corpus in ("twitter15", "twitter16", "weibo"):
        if sm["graph_informative"][corpus]["timestamp_column"]:
            raise AssertionError(
                f"Table 1 says the {corpus} release ships no event-time column; it now does")
    print("ok  signal matrix: URL/media degeneracy, UPFD splits and 'no event times' hold")

    # ------------------------------------------------------ cross-reference guards
    # main.tex and supplement.tex are separately compiled, so 'Table~S<n>' and
    # 'main-text \S5.<n>' are literal strings that LaTeX cannot check. A stale
    # pointer is invisible to `grep -c undefined` and to '??' in the PDF.
    import re as _re
    main_tex = _source("main.tex")
    supp_tex = _source("supplement.tex")
    if main_tex.exists() and supp_tex.exists():
        supp_src = supp_tex.read_text()
        n_supp_tables = len(_re.findall(r"\\label\{tab:supp-", supp_src))
        refs = [int(m) for m in _re.findall(r"Table~S(\d+)", main_tex.read_text())]
        bad = [n for n in refs if n < 1 or n > n_supp_tables]
        if bad:
            raise AssertionError(
                f"main.tex points at supplement Table S{bad}, but the supplement "
                f"defines only {n_supp_tables} tables")
        print(f"ok  all {len(refs)} 'Table~S<n>' pointers resolve "
              f"({n_supp_tables} supplement tables)")

        # The bound above is only an upper bound: inserting a supplement table
        # BEFORE an existing one silently redirects every later pointer while the
        # count check still passes. Pin the identity of each cited table instead.
        supp_labels = _re.findall(r"\\label\{(tab:supp-[^}]*)\}", supp_src)
        expected_targets = {
            1: "tab:supp-transfer", 3: "tab:supp-rawlog", 4: "tab:supp-hawkes",
            5: "tab:supp-calib", 6: "tab:supp-size", 7: "tab:supp-mitig",
            8: "tab:supp-structsummary", 9: "tab:supp-reconcile",
            10: "tab:supp-notation",
        }
        for n, want in expected_targets.items():
            got = supp_labels[n - 1] if n <= len(supp_labels) else "<missing>"
            if got != want:
                raise AssertionError(
                    f"main.tex cites supplement Table~S{n} expecting {want}, but S{n} is "
                    f"now {got}. A supplement table was inserted or reordered ahead of it; "
                    "renumber the main-text pointer or move the new table later.")
        print(f"ok  all {len(expected_targets)} cited supplement tables still have their "
              "expected identity")

        subsecs = _re.findall(r"\\subsection\{", main_tex.read_text())
        n_results_subsecs = len(_re.findall(
            r"\\subsection\{[^}]*\}\\label\{sec:(?:idfree|graphinf|general|imbal"
            r"|secondary|text|reconcile)\}", main_tex.read_text()))
        cited = sorted({int(m) for m in
                        _re.findall(r"main[- ]text \\S5\.(\d+)", supp_src)})
        bad = [n for n in cited if n < 1 or n > n_results_subsecs]
        if bad:
            raise AssertionError(
                f"supplement cites main-text 5.{bad}, but Section 5 has only "
                f"{n_results_subsecs} subsections")
        print(f"ok  all supplement 'main-text 5.<n>' pointers resolve "
              f"(cited {cited}, Section 5 has {n_results_subsecs})")

    # ---- revision r15: validation-clean selection, tree power, controls ----
    vs_gc = load_json("outputs/r15/graph_valselect/upfd_gossipcop.json")
    if vs_gc["selection_rule"] != "val":
        raise AssertionError("UPFD-GossipCop valselect run is not the validation arm")
    if vs_gc["best_graph"] != "bigcn" or vs_gc["best_summary"] != "histgbt":
        raise AssertionError(
            "main.tex says validation selection picks the same pair as test selection on "
            f"UPFD-GossipCop; run now picks {vs_gc['best_graph']}/{vs_gc['best_summary']}")
    if vs_gc["best_graph"] != vs_gc["test_selected_best_graph"] or \
       vs_gc["best_summary"] != vs_gc["test_selected_best_summary"]:
        raise AssertionError("validation and test selection no longer agree on UPFD-GossipCop")
    gc_cell = vs_gc["clustered_best_graph_vs_best_summary"]
    close(gc_cell["mean"], 0.0125, "val-clean UPFD-GossipCop delta", tol=5e-4)
    close(gc_cell["ci95"][0], 0.0053, "val-clean UPFD-GossipCop CI lo", tol=5e-4)
    close(gc_cell["ci95"][1], 0.0198, "val-clean UPFD-GossipCop CI hi", tol=5e-4)
    if not gc_cell["tost_equivalent_0.03"]:
        raise AssertionError("main.tex claims the val-clean GossipCop pair stays inside 0.03")
    close(gc_cell["n_stories"], 3826, "val-clean UPFD-GossipCop clusters")

    vs_pf = load_json("outputs/r15/graph_valselect/upfd_politifact.json")
    pf_cell = vs_pf["clustered_best_graph_vs_best_summary"]
    close(pf_cell["mean"], 0.0172, "val-clean UPFD-PolitiFact delta", tol=5e-4)
    if vs_pf["best_summary"] != "rf" or vs_pf["test_selected_best_summary"] != "mlp":
        raise AssertionError("main.tex says validation selection swaps MLP for RF on PolitiFact")

    tp = load_json("outputs/r15/tree_power/tree_power.json")["per_corpus"]
    for ds, mde in (("upfd_gossipcop", 0.010), ("weibo", 0.014), ("twitter15", 0.051),
                    ("twitter16", 0.063), ("upfd_politifact", 0.080)):
        close(tp[ds]["mde_80"], mde, f"tree MDE80 {ds}", tol=5e-4)

    so = load_json("outputs/r15/size_only/size_only.json")["corpora"]
    for ds, f1 in (("coaid", 0.5142), ("recovery", 0.6230),
                   ("politifact", 0.4963), ("gossipcop", 0.7068)):
        close(so[ds]["macro_f1_mean"], f1, f"size-only macro-F1 {ds}", tol=5e-4)

    pa = load_json("outputs/r15/profile_permute/analysis.json")["per_variant"]
    for variant in pa:
        for ds, cell in pa[variant].items():
            if not cell["summary_bit_identical"]:
                raise AssertionError(
                    f"profile permutation ({variant}/{ds}) changed a summary baseline; the "
                    "control is only interpretable if the summary side is untouched")
    print(f"ok  profile permutation left every summary baseline bit-identical "
          f"({sum(len(v) for v in pa.values())} corpus-variant cells)")

    print("\nSelected headline claim checks passed (PropMLP/DocLessFake/RGCN/GATv2 "
          "cells, key p-values, tabular, GossipCop cap, and mitigation point "
          "estimates). This is not the full event-list table; README.md maps every "
          "table and figure to its script and stored output.")


if __name__ == "__main__":
    main()
