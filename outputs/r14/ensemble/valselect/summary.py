"""One-screen summary of the val-selection control."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT
r = json.load(open(f"{ROOT}/report.json"))
vr = json.load(open(f"{ROOT}/val_resolution.json"))
ls = json.load(open(f"{ROOT}/leader_stability.json"))
p = r["full_pool"]
NAMES = {"rf": "RandomForest(7d)", "propmlp": "PropMLP(7d)", "propmlp15": "PropMLP+(15d)",
         "histgbt15": "HistGBT(15d)*", "ebgcn": "EBGCN*", "gatv2": "GATv2*",
         "dlf": "DLF-Full(7d)", "dlf15": "DLF-Full+(15d)"}
print(f"{'corpus':11s} {'|pool|':>6s} {'n_val':>5s} {'VAL PICK':>16s} {'valF1':>6s} "
      f"{'testF1':>7s} {'TEST LEADER':>16s} {'oracle':>7s} {'regret':>7s} {'hit':>5s} {'P(pick leads test)':>19s}")
for c in sorted(p["selection"]):
    s, d = p["selection"][c], p["diagnostics"][c]
    a = s["variant_A_component"]
    pl = ls.get(c, {}).get("p_leads_on_test", {}).get(a)
    print(f"{c:11s} {len(p['pool'][c]):6d} {vr[c]['n_val_per_seed']:5d} {NAMES[a]:>16s} "
          f"{s['variant_A_val_mean']:6.3f} {s['variant_A_test_mean']:7.3f} "
          f"{NAMES[d['oracle_component_on_test_mean']]:>16s} {d['oracle_test_mean']:7.3f} "
          f"{d['variant_A_regret_vs_oracle']:+7.3f} {str(d['variant_A_picked_test_leader']):>5s} "
          f"{'--' if pl is None else f'{pl:.3f}':>19s}")
print()
h = p["headline"]
print(f"val selection recovers the test leader on {h['variant_A_hits_on_test_leader']}/"
      f"{h['variant_A_n_corpora']} corpora (variant A, per-corpus)")
print(f"per-seed selection (variant B) recovers the per-seed test leader on "
      f"{h['variant_B_hits_on_test_leader']}/{h['variant_B_cells']} corpus-seed cells")
print(f"mean regret vs the test-fold oracle: A {h['mean_regret_A']:+.4f}   B {h['mean_regret_B']:+.4f}")
print()
print("VAL FOLD RESOLVING POWER (val only):")
for c in sorted(vr):
    v = vr[c]
    print(f"  {c:11s} n_val={v['n_val_per_seed']:4d}  leader margin {v['val_leader_margin']:.4f}  "
          f"bootstrap SD {v['mean_bootstrap_sd']:.4f}  margin/SD={v['margin_over_sd']:.2f}  "
          f"P(same val leader under resampling)={v['p_same_val_leader_under_resampling']:.2f}")
print()
print("BEATS BEST SINGLE ON VAL (the fold selection saw):")
for c in sorted(p["beats_best_single_on_val"]):
    b = p["beats_best_single_on_val"][c]
    print(f"  {c:11s} A {b['variant_A_val_mean']:.4f} vs best single {b['best_single_val_mean']:.4f}"
          f"  -> strictly beats: {b['A_strictly_beats']}   (B {b['variant_B_val_mean']:.4f}, "
          f"beats: {b['B_strictly_beats']})")
u = p.get("uniform_single_baseline")
if u:
    print()
    print(f"vs BEST UNIFORM SINGLE MODEL (chosen on val = {u['best_uniform_chosen_on_val']}):")
    print(f"  val : per-corpus selection {u['per_corpus_selection_val_mean']:.4f} vs uniform "
          f"{u['best_uniform_val_mean']:.4f}  -> beats: {u['per_corpus_selection_beats_uniform_on_val']}")
    print(f"  test: per-corpus selection {u['per_corpus_selection_test_mean']:.4f} vs uniform "
          f"{u['best_uniform_test_mean']:.4f}  -> beats: {u['per_corpus_selection_beats_uniform_on_test']}")
print()
print("* = component whose protocol had to be changed to make its val score honest; "
      "all others reproduce the published main-table cell exactly.")
