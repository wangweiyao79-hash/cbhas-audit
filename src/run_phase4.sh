#!/usr/bin/env bash
# Phase 4 orchestrator — M7 threshold sensitivity, Bash version.
#
# Grid (6 configs; the paper's origin tau_s=50/N=5000 is already covered by
# main_results.csv):
#   tau_s in {20, 100}   with N_target=5000
#   N_target in {2000, 10000, 20000}  with tau_s=50
#   Interaction:  tau_s=20  N_target=2000
#
# 6 TVAE augmentations + 6 CE trainings => ~1 h at 2 min/train + 5 min/aug.

set +e
cd "$(dirname "$0")/.."

PY="./venv/Scripts/python.exe"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
CSV="results/tables/phase4_sensitivity.csv"

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

sweep() {
    local tau_s=$1
    local n_target=$2
    local stem="tvae_taus${tau_s}_N${n_target}"
    local tag="tvae_taus${tau_s}_N${n_target}"

    run_job "P4 tau_s=$tau_s N=$n_target - augment" "p4_aug_${stem}" \
            src/augment_tvae.py --seed 42 --tau_s $tau_s --n_target $n_target \
            --out "train_aug_${stem}"

    run_job "P4 tau_s=$tau_s N=$n_target - train" "p4_train_${stem}" \
            src/train.py --data "train_aug_${stem}" --loss ce --seed 42 \
            --tag "$tag" --results-csv "$CSV"
}

# ── tau_s sweep (N_target fixed at 5000) ────────────────────────────────────
sweep 20  5000
sweep 100 5000

# ── N_target sweep (tau_s fixed at 50) ──────────────────────────────────────
sweep 50  2000
sweep 50  10000
sweep 50  20000

# ── Interaction ─────────────────────────────────────────────────────────────
sweep 20  2000

echo "[ALL DONE] Results in: $CSV"
