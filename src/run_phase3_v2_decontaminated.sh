#!/usr/bin/env bash
# Phase 3 v2 — decontaminated matched-budget rerun (M5 fairness re-audit).
#
# The original ros_matched/smote_matched/adasyn_matched runs (phase3_results.csv,
# archived as phase3_results_contaminated.csv) silently capped BENIGN and three
# other majority/near-majority classes to 50,000 samples via RandomUnderSampler
# before oversampling — a memory-safety measure that was never applied to
# CB-HAS, confounding the M5 "matched budget" comparison. resample() in
# train.py has been fixed to skip this cap entirely on the matched-budget path
# and to assert the untouched classes are byte-for-byte preserved. This script
# reruns all 9 matched-budget configs (3 samplers x 3 seeds) with the fix.

set +e
cd "$(dirname "$0")/.."

PY="./venv/Scripts/python.exe"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
CSV="results/tables/phase3_results_v2.csv"

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
        local dcap=$(grep -oE "non-augmented classes byte-for-byte preserved: [A-Za-z]+" "$log" | tail -1)
        echo "[job $JOB_IDX OK   ] ${dt}s  F1=$f1  [$dcap]  $desc"
    fi
}

for seed in 42 123 2024; do
    run_job "[seed=$seed] ROS matched (decontaminated)" "p3v2_ros_matched_s${seed}" \
            src/train.py --sampler ros --budget matched --tau 2000 --n_target 5000 \
            --tag ros_matched --seed $seed --results-csv "$CSV"

    run_job "[seed=$seed] SMOTE matched (decontaminated)" "p3v2_smote_matched_s${seed}" \
            src/train.py --sampler smote --budget matched --tau 2000 --n_target 5000 \
            --tag smote_matched --seed $seed --results-csv "$CSV"

    run_job "[seed=$seed] ADASYN matched (decontaminated)" "p3v2_adasyn_matched_s${seed}" \
            src/train.py --sampler adasyn --budget matched --tau 2000 --n_target 5000 \
            --tag adasyn_matched --seed $seed --results-csv "$CSV"
done

echo "[ALL DONE] Results in: $CSV"
