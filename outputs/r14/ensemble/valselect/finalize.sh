#!/bin/bash
# Re-derive every product of the val-selection control from the per-story dumps.
# Order matters: analyze -> manifest -> audit -> diagnostics -> report.
set -e
cd <artifact-root>
export PYTHONPATH=src
D=outputs/r14/ensemble/valselect
run(){ echo "=== $* ==="; nice -n 15 python3 "$@"; }
run $D/analyze.py                                  # report.json + combiner test preds
run $D/manifest.py > /dev/null                     # MANIFEST.json
run $D/verify_no_leakage.py                        # THE AUDIT: blind test, re-select
run $D/published_leader_check.py
B=1000 run $D/valnoise.py
B=4000 run $D/stability.py
POOL=full   run $D/val_loso.py
POOL=common run $D/val_loso.py
B=3000 run $D/val_loso_ci.py
B=3000 run $D/val_contrast.py > /dev/null
run $D/make_report.py
echo "=== DONE ==="
