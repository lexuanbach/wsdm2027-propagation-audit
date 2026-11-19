"""Apply the frozen val-fold selection rule and emit (a) the combiner's test
predictions and (b) selection diagnostics.

Selection reads ONLY val_* fields (see SELECTION_RULE.json, frozen first).
Test fields are read afterwards and only for reporting.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np
from collections import Counter
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PRED, SEEDS, DATASETS
from docless.eval.stats import macro_f1

ORDER = ["rf", "propmlp", "propmlp15", "histgbt15", "ebgcn", "gatv2", "dlf", "dlf15"]
PRETTY = {"rf": "RandomForest (7d)", "propmlp": "PropMLP (7d)", "propmlp15": "PropMLP+ (15d)",
          "histgbt15": "HistGBT (15d)", "ebgcn": "EBGCN-style", "gatv2": "GATv2",
          "dlf": "DLF-Full (7d)", "dlf15": "DLF-Full+ (15d)"}
# components whose regenerated test macro-F1 reproduces the published cell exactly
UNMODIFIED = {"rf", "propmlp", "propmlp15", "dlf", "dlf15"}
# published main-table test means, for the fidelity check
PUBLISHED = {
    "coaid":      {"propmlp": .618, "propmlp15": .610, "rf": .595, "histgbt15": .629,
                   "ebgcn": .611, "gatv2": .598, "dlf": .654, "dlf15": .621},
    "recovery":   {"propmlp": .659, "propmlp15": .699, "rf": .651, "histgbt15": .646,
                   "ebgcn": .675, "gatv2": .597, "dlf": .663, "dlf15": .689},
    "politifact": {"propmlp": .657, "propmlp15": .717, "rf": .667, "histgbt15": .706,
                   "ebgcn": .660, "gatv2": .627, "dlf": .678, "dlf15": .724},
    "gossipcop":  {"propmlp": .836, "propmlp15": .851, "rf": .865, "histgbt15": .877,
                   "ebgcn": .789, "gatv2": .797, "dlf": .842, "dlf15": None},
}


def load():
    R = {}
    for c in DATASETS:
        for f in sorted(glob.glob(f"{PRED}/{c}/*__seed*.json")):
            d = json.load(open(f))
            R.setdefault(c, {}).setdefault(d["model"], {})[d["seed"]] = d
    return {c: {m: v for m, v in ms.items() if set(v) == set(SEEDS)} for c, ms in R.items()}


def f1s(R, c, m, fold):
    return [float(macro_f1(np.asarray(R[c][m][s][f"{fold}_preds"]),
                           np.asarray(R[c][m][s][f"{fold}_labels"]))) for s in SEEDS]


def select(val, R, c, pool):
    """VAL ONLY. Returns (variant-A component, {seed: variant-B component})."""
    mv = val[c]
    A = min(pool, key=lambda m: (-float(np.mean(mv[m])), float(np.std(mv[m])), ORDER.index(m)))
    B = {}
    for i, s in enumerate(SEEDS):
        acc = {m: float((np.asarray(R[c][m][s]["val_preds"]) ==
                         np.asarray(R[c][m][s]["val_labels"])).mean()) for m in pool}
        B[s] = min(pool, key=lambda m: (-mv[m][i], -acc[m], ORDER.index(m)))
    return A, B


def analyse(val, test, R, pools, tag):
    rep = {"pool": pools, "selection": {}, "diagnostics": {}, "beats_best_single_on_val": {}}
    combiner = {}
    for c, pool in pools.items():
        if len(pool) < 2:
            continue
        A, Bm = select(val, R, c, pool)
        B = [Bm[s] for s in SEEDS]
        tA = [test[c][A][i] for i in range(5)]
        tB = [test[c][B[i]][i] for i in range(5)]
        oracle = max(pool, key=lambda m: np.mean(test[c][m]))
        rep["selection"][c] = dict(
            variant_A_component=A, variant_A_pretty=PRETTY[A],
            variant_A_val_mean=round(float(np.mean(val[c][A])), 4),
            variant_A_val_per_seed=[round(x, 4) for x in val[c][A]],
            variant_A_test_per_seed=[round(x, 4) for x in tA],
            variant_A_test_mean=round(float(np.mean(tA)), 4),
            variant_A_test_std=round(float(np.std(tA)), 4),
            variant_B_components=B,
            variant_B_test_per_seed=[round(x, 4) for x in tB],
            variant_B_test_mean=round(float(np.mean(tB)), 4),
            variant_B_test_std=round(float(np.std(tB)), 4),
        )
        rv = [-np.mean(val[c][m]) for m in pool]
        rt = [-np.mean(test[c][m]) for m in pool]
        loo = []
        for k in range(5):
            keep = [i for i in range(5) if i != k]
            loo.append(min(pool, key=lambda m: (-float(np.mean([val[c][m][i] for i in keep])),
                                                float(np.std([val[c][m][i] for i in keep])),
                                                ORDER.index(m))))
        per_seed_test_leader = [max(pool, key=lambda m: test[c][m][i]) for i in range(5)]
        rep["diagnostics"][c] = dict(
            oracle_component_on_test_mean=oracle,
            oracle_test_mean=round(float(np.mean(test[c][oracle])), 4),
            variant_A_picked_test_leader=bool(A == oracle),
            variant_A_regret_vs_oracle=round(float(np.mean(test[c][oracle]) - np.mean(tA)), 4),
            variant_B_seedwise_hits=int(sum(
                test[c][B[i]][i] >= max(test[c][m][i] for m in pool) - 1e-12 for i in range(5))),
            variant_B_regret_vs_seed_oracle=round(float(
                np.mean([max(test[c][m][i] for m in pool) for i in range(5)]) - np.mean(tB)), 4),
            spearman_val_vs_test_ranking=(round(float(spearmanr(rv, rt).statistic), 3)
                                          if len(pool) > 2 else None),
            kendall_val_vs_test_ranking=(round(float(kendalltau(rv, rt).statistic), 3)
                                         if len(pool) > 2 else None),
            val_leader_margin=round(float(sorted(-np.array(rv))[-1] - sorted(-np.array(rv))[-2]), 4),
            test_leader_margin=round(float(sorted(-np.array(rt))[-1] - sorted(-np.array(rt))[-2]), 4),
            test_leader_by_seed=per_seed_test_leader,
            n_distinct_test_leaders_across_seeds=len(set(per_seed_test_leader)),
            variant_A_leave_one_seed_out_picks=loo,
            variant_A_loo_stable=bool(len(set(loo)) == 1),
            variant_A_test_minus_component={
                m: round(float(np.mean([tA[i] - test[c][m][i] for i in range(5)])), 4) for m in pool},
        )
        bsv = max(np.mean(val[c][m]) for m in pool)
        vA = float(np.mean(val[c][A]))
        vB = float(np.mean([val[c][B[i]][i] for i in range(5)]))
        rep["beats_best_single_on_val"][c] = dict(
            best_single_val_mean=round(bsv, 4), variant_A_val_mean=round(vA, 4),
            variant_B_val_mean=round(vB, 4),
            A_strictly_beats=bool(vA > bsv + 1e-9), B_strictly_beats=bool(vB > bsv + 1e-9))
        combiner[c] = {}
        for i, s in enumerate(SEEDS):
            for v, m in (("A", A), ("B", B[i])):
                d = R[c][m][s]
                combiner[c].setdefault(v, {})[str(s)] = dict(
                    selected_component=m, story_ids=d["test_story_ids"],
                    labels=d["test_labels"], preds=d["test_preds"], p1=d["test_p1"])
    # ---- cross-corpus: does per-corpus selection beat the best UNIFORM single model? ----
    cs0 = sorted(rep["selection"])
    common = [m for m in ORDER if all(m in pools[c] for c in cs0)]
    if cs0 and common:
        uni_val = {m: float(np.mean([np.mean(val[c][m]) for c in cs0])) for m in common}
        uni_test = {m: float(np.mean([np.mean(test[c][m]) for c in cs0])) for m in common}
        best_uni = min(common, key=lambda m: (-uni_val[m], ORDER.index(m)))   # chosen on VAL
        rep["uniform_single_baseline"] = dict(
            components_available_on_all_corpora=common,
            val_means={m: round(uni_val[m], 4) for m in common},
            test_means={m: round(uni_test[m], 4) for m in common},
            best_uniform_chosen_on_val=best_uni,
            best_uniform_val_mean=round(uni_val[best_uni], 4),
            best_uniform_test_mean=round(uni_test[best_uni], 4),
            per_corpus_selection_val_mean=round(float(np.mean(
                [rep["selection"][c]["variant_A_val_mean"] for c in cs0])), 4),
            per_corpus_selection_test_mean=round(float(np.mean(
                [rep["selection"][c]["variant_A_test_mean"] for c in cs0])), 4),
            per_corpus_selection_beats_uniform_on_val=bool(
                np.mean([rep["selection"][c]["variant_A_val_mean"] for c in cs0]) > uni_val[best_uni] + 1e-9),
            per_corpus_selection_beats_uniform_on_test=bool(
                np.mean([rep["selection"][c]["variant_A_test_mean"] for c in cs0]) > uni_test[best_uni] + 1e-9),
            note="best_uniform is picked by mean VAL macro-F1 across corpora, never by test.",
        )

    cs = sorted(rep["selection"])
    rep["headline"] = dict(
        pool_tag=tag, corpora=cs,
        variant_A_components={c: rep["selection"][c]["variant_A_component"] for c in cs},
        variant_A_hits_on_test_leader=sum(rep["diagnostics"][c]["variant_A_picked_test_leader"] for c in cs),
        variant_A_n_corpora=len(cs),
        variant_B_hits_on_test_leader=sum(rep["diagnostics"][c]["variant_B_seedwise_hits"] for c in cs),
        variant_B_cells=5 * len(cs),
        variant_A_test_means={c: rep["selection"][c]["variant_A_test_mean"] for c in cs},
        variant_B_test_means={c: rep["selection"][c]["variant_B_test_mean"] for c in cs},
        mean_regret_A=round(float(np.mean([rep["diagnostics"][c]["variant_A_regret_vs_oracle"] for c in cs])), 4),
        mean_regret_B=round(float(np.mean([rep["diagnostics"][c]["variant_B_regret_vs_seed_oracle"] for c in cs])), 4),
        A_beats_best_single_on_val_anywhere=any(
            rep["beats_best_single_on_val"][c]["A_strictly_beats"] for c in cs),
    )
    return rep, combiner


def main():
    R = load()
    val = {c: {m: f1s(R, c, m, "val") for m in R[c]} for c in R}
    test = {c: {m: f1s(R, c, m, "test") for m in R[c]} for c in R}

    full = {c: [m for m in ORDER if m in R[c]] for c in R}
    strict = {c: [m for m in full[c] if m in UNMODIFIED] for c in R}

    common = [m for m in ORDER if all(m in full[c] for c in full)]
    commonp = {c: list(common) for c in full}
    rep_full, comb_full = analyse(val, test, R, full, "full_pool")
    rep_common, comb_common = analyse(val, test, R, commonp, "common_pool")
    rep_strict, comb_strict = analyse(val, test, R, strict, "unmodified_protocol_pool")

    fid = {}
    for c in R:
        fid[c] = {m: dict(regen_test_mean=round(float(np.mean(test[c][m])), 4),
                          published_test_mean=PUBLISHED[c].get(m),
                          matches=bool(PUBLISHED[c].get(m) is not None and
                                       abs(np.mean(test[c][m]) - PUBLISHED[c][m]) < 0.0015))
                  for m in full[c]}

    out = dict(rule="outputs/r14/ensemble/valselect/SELECTION_RULE.json",
               fidelity_vs_published=fid,
               val_table={c: {m: dict(per_seed=[round(x, 4) for x in val[c][m]],
                                      mean=round(float(np.mean(val[c][m])), 4),
                                      std=round(float(np.std(val[c][m])), 4)) for m in full[c]} for c in R},
               test_table={c: {m: dict(per_seed=[round(x, 4) for x in test[c][m]],
                                       mean=round(float(np.mean(test[c][m])), 4),
                                       std=round(float(np.std(test[c][m])), 4)) for m in full[c]} for c in R},
               full_pool=rep_full, unmodified_protocol_pool=rep_strict,
               common_pool=rep_common)
    json.dump(out, open(f"{ROOT}/report.json", "w"), indent=2)
    os.makedirs(f"{ROOT}/combiner", exist_ok=True)
    json.dump(comb_full, open(f"{ROOT}/combiner/test_preds_full_pool.json", "w"))
    json.dump(comb_strict, open(f"{ROOT}/combiner/test_preds_unmodified_pool.json", "w"))
    json.dump(comb_common, open(f"{ROOT}/combiner/test_preds_common_pool.json", "w"))

    for tag, rep in (("FULL POOL", rep_full), ("UNMODIFIED-PROTOCOL POOL", rep_strict),
                     ("COMMON POOL", rep_common)):
        print(f"\n############ {tag} ############")
        for c in sorted(rep["selection"]):
            pool = rep["pool"][c]
            print(f"\n=== {c}  (pool: {', '.join(pool)}) ===")
            print(f"  {'component':12s} {'val mean':>9s} {'test mean':>10s}  {'pub':>6s}")
            for m in sorted(pool, key=lambda m: -np.mean(val[c][m])):
                mk = " <-VAL PICK" if m == rep["selection"][c]["variant_A_component"] else ""
                mk += " *TEST LEADER*" if m == rep["diagnostics"][c]["oracle_component_on_test_mean"] else ""
                pub = PUBLISHED[c].get(m)
                print(f"  {m:12s} {np.mean(val[c][m]):9.4f} {np.mean(test[c][m]):10.4f}  "
                      f"{'--' if pub is None else f'{pub:.3f}':>6s}{mk}")
            s, d = rep["selection"][c], rep["diagnostics"][c]
            print(f"  A: {s['variant_A_component']} -> test {s['variant_A_test_mean']:.4f}"
                  f" (oracle {d['oracle_component_on_test_mean']} {d['oracle_test_mean']:.4f},"
                  f" regret {d['variant_A_regret_vs_oracle']:+.4f}, hit={d['variant_A_picked_test_leader']},"
                  f" LOO-stable={d['variant_A_loo_stable']})")
            print(f"  B: {s['variant_B_components']} -> test {s['variant_B_test_mean']:.4f}"
                  f" (seedwise hits {d['variant_B_seedwise_hits']}/5)")
            print(f"  val-vs-test rank corr: spearman={d['spearman_val_vs_test_ranking']}"
                  f"  test leader by seed: {d['test_leader_by_seed']}")
        print("\nHEADLINE:", json.dumps(rep["headline"], indent=2))
        print("BEATS BEST SINGLE ON VAL:", json.dumps(rep["beats_best_single_on_val"], indent=2))
        if "uniform_single_baseline" in rep:
            print("UNIFORM-SINGLE BASELINE:", json.dumps(rep["uniform_single_baseline"], indent=2))
    print("\nFIDELITY vs published:", json.dumps(fid, indent=2))


if __name__ == "__main__":
    main()
