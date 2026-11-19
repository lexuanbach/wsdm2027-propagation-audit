"""Auditor's check: prove the selection is reproducible from VAL fields alone.

Loads every per-story dump, DELETES every test_* field in memory, re-runs the
selection rule on what is left, and asserts it reproduces MANIFEST.json's
selected component for every corpus and both variants. Also re-verifies fold
membership against the independently-built fold manifest.
"""
from __future__ import annotations
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, PRED, SEEDS, DATASETS
from docless.eval.stats import macro_f1
from analyze import ORDER, select, load as _load

man = json.load(open(f"{ROOT}/MANIFEST.json"))
fold = json.load(open("outputs/r14/ensemble/inventory/fold_manifest.json"))

R = _load()
# 1. fold membership must match the independent manifest
n_checked = 0
for c in R:
    for m in R[c]:
        for s in SEEDS:
            d = R[c][m][s]
            assert d["val_story_ids"] == fold[c][str(s)]["val_ids"], (c, m, s, "val ids")
            assert d["test_story_ids"] == fold[c][str(s)]["test_ids"], (c, m, s, "test ids")
            assert d["test_labels"] == fold[c][str(s)]["test_labels"], (c, m, s, "test labels")
            n_checked += 1
print(f"[ok] fold membership verified for {n_checked} (corpus, model, seed) dumps")

# 2. blind the test fold entirely, then re-run selection
for c in R:
    for m in R[c]:
        for s in SEEDS:
            for k in [k for k in R[c][m][s] if k.startswith("test_")]:
                R[c][m][s][k] = None
val = {c: {m: [float(macro_f1(np.asarray(R[c][m][s]["val_preds"]),
                              np.asarray(R[c][m][s]["val_labels"]))) for s in SEEDS]
           for m in R[c]} for c in R}
pool = {c: [m for m in ORDER if m in R[c]] for c in R}
blind = {}
for c in pool:
    A, Bm = select(val, R, c, pool[c])
    blind[c] = A
print("[ok] selection recomputed with every test_* field set to None:", blind)

exp = man["primary"]["selected_component_per_corpus"]
assert blind == exp, f"MISMATCH blind={blind} manifest={exp}"
print("[ok] blind selection reproduces MANIFEST.json exactly -- selection uses no test data")
