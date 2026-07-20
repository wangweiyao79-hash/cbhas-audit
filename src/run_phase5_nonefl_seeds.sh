#!/usr/bin/env bash
# Phase 5 — multi-seed for the 2x2 interaction's "none" arm (no augmentation).
#
# none+FL(fixed, alpha=0.25) and none+FL(sqrt_inverse) each get 4 new seeds
# (7, 123, 1337, 2024) to match the 5-seed set already used for none+CE and
# CB-HAS+CE/CB-HAS+FL(sqrt_inv) in Phase 1. seed=42 for both configs already
# exists in phase2_results.csv (read-only source, never modified here).
#
# Writes ONLY to a brand-new file (phase5_nonefl_new_seeds.csv) — no existing
# results file is touched. sampler=none never reaches the CAP/downsample code
# path (see resample()'s self-check), so no contamination risk here.

set +e
cd "$(dirname "$0")/.."

PY="./venv/Scripts/python.exe"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
CSV="results/tables/phase5_nonefl_new_seeds.csv"

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
        local chk=$(grep -oE "BENIGN count is|class 0: [0-9,]+" "$log" | tail -1)
        echo "[job $JOB_IDX OK   ] ${dt}s  F1=$f1  [$chk]  $desc"
    fi
}

for seed in 7 123 1337 2024; do
    run_job "[seed=$seed] none+FL(fixed alpha=0.25)" "p5_none_fl_fixed_s${seed}" \
            src/train.py --sampler none --loss focal --alpha_mode fixed \
            --seed $seed --results-csv "$CSV"

    run_job "[seed=$seed] none+FL(sqrt_inverse)" "p5_none_fl_sqrtinv_s${seed}" \
            src/train.py --sampler none --loss focal --alpha_mode sqrt_inverse \
            --seed $seed --results-csv "$CSV"
done

echo "[ALL DONE] Results in: $CSV"
