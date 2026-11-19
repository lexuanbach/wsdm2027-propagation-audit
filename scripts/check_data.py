#!/usr/bin/env python3
"""Check downloaded inputs against the SHA-256 values recorded for the paper.

The reference list is outputs/r15/dataset_card/dataset_card.json, written by
scripts/rev9_dataset_card.py on the machine that produced the paper's results.
Run after `make data` (or any subset of it):

    python scripts/check_data.py            # all three input trees
    python scripts/check_data.py --tree raw # only data/raw (event-list corpora)

A tree whose folder is absent is reported as "not downloaded" and is not an
error. Exit status is 1 if any present file differs from the recorded hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = ROOT / "outputs" / "r15" / "dataset_card" / "dataset_card.json"
# recorded tree root -> local folder
LOCAL = {
    "artifact/data/raw": ROOT / "data" / "raw",
    "external/BiGCN/data": ROOT / "external" / "BiGCN" / "data",
    "external/upfd": ROOT / "external" / "upfd",
}
SHORT = {"raw": "artifact/data/raw", "bigcn": "external/BiGCN/data", "upfd": "external/upfd"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", choices=sorted(SHORT), action="append",
                    help="limit the check to one tree (repeatable)")
    args = ap.parse_args()
    wanted = {SHORT[t] for t in args.tree} if args.tree else set(LOCAL)

    card = json.loads(CARD.read_text())
    bad = 0
    for tree in card["trees"]:
        name = tree["root"]
        if name not in wanted:
            continue
        local = LOCAL[name]
        if not local.exists():
            print(f"[skip] {name}: not downloaded ({local.relative_to(ROOT)}/ missing)")
            continue
        ok = missing = diff = 0
        for rel, rec in sorted(tree["files"].items()):
            path = local / rel
            if not path.exists():
                missing += 1
                print(f"  missing   {name}/{rel}")
            elif sha256(path) != rec["sha256"]:
                diff += 1
                print(f"  DIFFERENT {name}/{rel}")
            else:
                ok += 1
        bad += diff
        print(f"[{'ok' if not diff and not missing else 'check'}] {name}: "
              f"{ok} match, {diff} differ, {missing} missing of {len(tree['files'])}")
    if bad:
        print("Some inputs differ from the versions used in the paper; results may not match.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
