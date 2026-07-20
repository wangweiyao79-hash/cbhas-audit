#!/usr/bin/env bash
# Phase 6 — CB-HAS+FL(fixed alpha=0.25) multi-seed completion.
#
# Reuses the SAME cached train_aug_tvae_seed{S}.npz files that Phase 1 used
# for tvae+CE and tvae+FL(sqrt_inverse), verified byte-for-byte identical to
# the class distribution Phase 1 logged at generation time (BENIGN=1,257,033,
# all 7 routed classes == 5,000). This guarantees a clean same-seed, same-
# augmented-set paired comparison against CB-HAS+CE (phase1_multiseed.csv)
# that differs ONLY in the loss function.
#
# seed=42 already exists in main_results.csv (77.65%) — not rerun.
# Writes ONLY to a brand-new file; no existing results file is touched.

set +e
cd "$(dirname "$0")/.."

PY="./venv/Scripts/python.exe"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
CSV="results/tables/phase6_cbhas_flfixed.csv"

JOB_IDX=0
run_job() {
    local desc="$1"; shift
    local stem="$1"; shift
    JOB_IDX=$((JOB_IDX + 1))
    local log="$LOG_DIR/${STAMP}_${JOB_IDX}_${stem}.log"
    echo "[job $JOB_IDX START] $desc  ->  $log"
    local t0=$(date +%s)
    "$PY" "$@" > "$log" 2>&1
    local rc=$?
    local dt=$(( $(date +%s) - t0 ))
    if [ $rc -ne 0 ]; then
        echo "[job $JOB_IDX FAIL ] ${dt}s  exit=$rc  $desc"
    else
        local f1=$(grep -oE "Macro-F1[ ]+:[ ]+[0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$" || echo "n/a")
        echo "[job $JOB_IDX OK   ] ${dt}s  F1=$f1  $desc"
    fi
}

for seed in 7 123 1337 2024; do
    run_job "[seed=$seed] CB-HAS + FL(fixed alpha=0.25)" "p6_tvae_fl_fixed_s${seed}" \
            src/train.py --data "train_aug_tvae_seed${seed}" --loss focal \
            --alpha_mode fixed --seed $seed --tag tvae --results-csv "$CSV"
done

echo "[ALL DONE] Results in: $CSV"
