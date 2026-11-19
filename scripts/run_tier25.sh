#!/usr/bin/env bash
# Tier 2.5 — bigger-budget local sweep.
#
# Settings chosen so baselines (GCNFN, UPFD-prop) get enough capacity and
# epochs to escape the majority-class floor, while keeping total CPU wall
# time under ~4 hours on the local Apple Silicon machine.
#
#   hidden=64, num_layers=2
#   docless:  4 SSL pretrain + 25 finetune epochs
#   baselines: 30 epochs
#   seeds:    3
#   datasets: CoAID + ReCOVery (main_table, coverage)
#             CoAID only       (ablation, confshift)

set -uo pipefail

HIDDEN=64
PRE=4
FT=25
SHORT=30
SEEDS="0 1 2"
ROOT=outputs/tier25

mkdir -p "$ROOT"
LOG="$ROOT/run.log"
: > "$LOG"

stamp() { date '+%H:%M:%S'; }
echo "[$(stamp)] Tier 2.5 start" | tee -a "$LOG"

echo "[$(stamp)] >>> main_table" | tee -a "$LOG"
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m docless.eval.main_table \
    --datasets coaid recovery \
    --methods majority gcnfn upfd_prop docless \
    --seeds $SEEDS \
    --hidden-dim $HIDDEN \
    --epochs-short $SHORT \
    --epochs-docless-pre $PRE \
    --epochs-docless-ft $FT \
    --out "$ROOT/main_table" 2>&1 | tee -a "$LOG"

echo "[$(stamp)] >>> ablation" | tee -a "$LOG"
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m docless.eval.ablation \
    --datasets coaid \
    --seeds $SEEDS \
    --hidden-dim $HIDDEN \
    --pretrain-epochs $PRE \
    --finetune-epochs $FT \
    --out "$ROOT/ablation" 2>&1 | tee -a "$LOG"

echo "[$(stamp)] >>> coverage" | tee -a "$LOG"
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m docless.eval.coverage \
    --datasets coaid recovery \
    --seeds $SEEDS \
    --budgets 0.5 0.7 0.8 0.9 0.95 1.0 \
    --hidden-dim $HIDDEN \
    --pretrain-epochs $PRE \
    --finetune-epochs $FT \
    --out "$ROOT/coverage" 2>&1 | tee -a "$LOG"

echo "[$(stamp)] >>> confshift" | tee -a "$LOG"
PYTHONUNBUFFERED=1 PYTHONPATH=src python3 -m docless.eval.confshift \
    --datasets coaid \
    --seeds $SEEDS \
    --grid 2 4 8 \
    --hidden-dim $HIDDEN \
    --pretrain-epochs $PRE \
    --finetune-epochs $FT \
    --out "$ROOT/confshift" 2>&1 | tee -a "$LOG"

echo "[$(stamp)] >>> figures + report" | tee -a "$LOG"
PYTHONPATH=src python3 -m docless.eval.make_figures \
    --results-dir "$ROOT" --out results/figures_tier25 2>&1 | tee -a "$LOG"
PYTHONPATH=src python3 -m docless.eval.aggregate \
    --results-dir "$ROOT" --out results/tier25.md 2>&1 | tee -a "$LOG"

echo "[$(stamp)] Tier 2.5 done" | tee -a "$LOG"
