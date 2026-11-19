"""Shared fold/dump helpers for the val-fold model-selection control.

EVERY fit here is TRAIN-ONLY. Val is scored once per (corpus, model, seed) and is
used for selection; test predictions are dumped but never inspected during any
selection decision.
"""
from __future__ import annotations
import json, os
import numpy as np
from docless.train import DataConfig, _load_cascades, _set_seed, _stratified_split

ROOT = "outputs/r14/ensemble/valselect"
PRED = f"{ROOT}/preds"
SEEDS = [0, 1, 2, 3, 4]
DATASETS = {
    "coaid":      dict(source="coaid",      balanced=True, data_root="data/raw/coaid"),
    "recovery":   dict(source="recovery",   balanced=True, data_root="data/raw/recovery"),
    "politifact": dict(source="politifact", balanced=True, data_root="data/raw/fakenewsnet"),
    "gossipcop":  dict(source="gossipcop",  balanced=True, data_root="data/raw/fakenewsnet",
                       max_per_class=1500),
}


def folds(name, seed):
    """The one recipe every main-table script uses."""
    dcfg = DataConfig(**DATASETS[name])
    _set_seed(seed)
    casc = _load_cascades(dcfg, seed)
    tr, va, te = _stratified_split(casc, dcfg.split_ratios, seed)
    return dcfg, tr, va, te


_MANIFEST = None


def check_alignment(name, seed, va, te):
    """Assert the reconstructed folds match the independently-built manifest."""
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = json.load(open("outputs/r14/ensemble/inventory/fold_manifest.json"))
    m = _MANIFEST[name][str(seed)]
    assert [c.story_id for c in va] == m["val_ids"], f"val fold mismatch {name} {seed}"
    assert [c.story_id for c in te] == m["test_ids"], f"test fold mismatch {name} {seed}"
    assert [c.label for c in te] == m["test_labels"], f"test label mismatch {name} {seed}"


def dump(model, name, seed, va, te, val_pred, val_p1, test_pred, test_p1, extra=None):
    os.makedirs(f"{PRED}/{name}", exist_ok=True)
    rec = dict(
        model=model, corpus=name, seed=seed,
        val_story_ids=[c.story_id for c in va],
        val_labels=[int(c.label) for c in va],
        val_preds=[int(x) for x in val_pred],
        val_p1=None if val_p1 is None else [float(x) for x in val_p1],
        test_story_ids=[c.story_id for c in te],
        test_labels=[int(c.label) for c in te],
        test_preds=[int(x) for x in test_pred],
        test_p1=None if test_p1 is None else [float(x) for x in test_p1],
    )
    if extra:
        rec.update(extra)
    with open(f"{PRED}/{name}/{model}__seed{seed}.json", "w") as f:
        json.dump(rec, f)
    return rec
