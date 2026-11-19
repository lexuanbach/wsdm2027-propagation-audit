"""Materialise the deterministically regenerated split IDs.

The audit regenerates every split from its seed rather than shipping index
files, which is reproducible but leaves nothing to diff.  Required-checklist
item 7 asks for the exact split IDs, so this writes them out: for each
event-list corpus, seed, and fold, the ordered story-id list, plus a SHA-256
over that list so a re-run can be checked in one comparison.

Corpus configurations and the seed list are imported from the canonical ones in
scripts/rev2_power.py so the manifest cannot drift from the evaluated splits.
The UPFD corpora are not included: they ship official train/val/test index
files, which the loader uses verbatim (external/upfd/*/raw/{train,val,test}_idx.npy).

Usage: PYTHONPATH=src python3 scripts/rev9_split_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rev2_power import DATASETS, SEEDS                      # noqa: E402
from docless.train import (DataConfig, _load_cascades,      # noqa: E402
                           _set_seed, _stratified_split, _temporal_split)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "r15" / "split_manifest"


def digest(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": SEEDS, "splitters": ["stratified", "temporal"],
                "note": ("UPFD is excluded: it ships official index files that the "
                         "loader consumes verbatim"),
                "corpora": {}}
    for name, kw in DATASETS.items():
        dcfg = DataConfig(**kw)
        entry = {"config": kw, "ratios": list(dcfg.split_ratios), "folds": {}}
        for splitter, fn in (("stratified", _stratified_split), ("temporal", _temporal_split)):
            for s in SEEDS:
                _set_seed(s)
                casc = _load_cascades(dcfg, s)
                tr, va, te = fn(casc, dcfg.split_ratios, s)
                rec = {}
                for fold, part in (("train", tr), ("val", va), ("test", te)):
                    ids = [c.story_id for c in part]
                    rec[fold] = {"n": len(ids), "sha256_16": digest(ids), "story_ids": ids}
                entry["folds"][f"{splitter}__seed{s}"] = rec
                if splitter == "temporal":
                    break          # temporal split is seed-independent by construction
        manifest["corpora"][name] = entry
        st = entry["folds"]["stratified__seed0"]
        print(f"{name:11s} train/val/test = {st['train']['n']}/{st['val']['n']}/{st['test']['n']} "
              f"test-sha={st['test']['sha256_16']}")
    (OUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {OUT / 'split_manifest.json'}")


if __name__ == "__main__":
    main()
