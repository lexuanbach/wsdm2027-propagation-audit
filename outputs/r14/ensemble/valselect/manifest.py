"""Emit the judge-facing manifest: what this combiner is, what it selected, and
where its frozen test predictions live."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT

rep = json.load(open(f"{ROOT}/report.json"))
rule = json.load(open(f"{ROOT}/SELECTION_RULE.json"))
prim = rep["full_pool"]
man = dict(
    key="valselect",
    name="Per-corpus model selection on the validation fold",
    primary=dict(pool="full_pool", variant="A (per-corpus, mean val macro-F1 over seeds 0-4)",
                 selected_component_per_corpus=prim["headline"]["variant_A_components"],
                 pool_per_corpus=prim["pool"],
                 val_mean_per_corpus={c: prim["selection"][c]["variant_A_val_mean"] for c in prim["selection"]},
                 test_predictions_file=f"{ROOT}/combiner/test_preds_full_pool.json",
                 test_predictions_layout="{corpus: {'A'|'B': {seed: {selected_component, story_ids, labels, preds, p1}}}}"),
    robustness=dict(
        unmodified_protocol_pool=dict(
            selected=rep["unmodified_protocol_pool"]["headline"]["variant_A_components"],
            file=f"{ROOT}/combiner/test_preds_unmodified_pool.json"),
        common_pool=dict(
            selected=rep["common_pool"]["headline"]["variant_A_components"],
            file=f"{ROOT}/combiner/test_preds_common_pool.json")),
    rule_file=f"{ROOT}/SELECTION_RULE.json",
    report_file=f"{ROOT}/report.json",
    leader_stability_file=f"{ROOT}/leader_stability.json",
    val_resolution_file=f"{ROOT}/val_resolution.json",
    per_story_dumps=f"{ROOT}/preds/<corpus>/<model>__seed<0-4>.json",
    invariants=[
        "every component fitted on the TRAIN fold only",
        "selection reads val_* fields only; the rule was frozen in SELECTION_RULE.json first",
        "test fold read only to emit predictions and post-hoc diagnostics",
        "seeds 0-4, paired: every component's fold membership verified story-for-story "
        "against outputs/r14/ensemble/inventory/fold_manifest.json",
    ],
)
json.dump(man, open(f"{ROOT}/MANIFEST.json", "w"), indent=2)
print(json.dumps(man, indent=2))
