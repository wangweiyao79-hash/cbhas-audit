#!/usr/bin/env bash
# Sequential Phase 2 (remaining) + Phase 3 runbook, portable Bash version.
#
# Emits one progress line per job so a Monitor tool can pick up events:
#   [job N START] desc
#   [job N OK   ] {sec}s  F1={value}  desc
#   [job N FAIL ] {sec}s  exit=K  desc
#
# Full per-job logs go to results/logs/{stamp}_{N}_{stem}.log.

set +e   # continue past a single failed job — capture exit code, don't abort
cd "$(dirname "$0")/.."     # anchor at repo root

PY="./venv/Scripts/python.exe"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

CSV_P2="results/tables/phase2_results.csv"
CSV_P3="results/tables/phase3_results.csv"

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
        # Extract final Macro-F1 from log tail
        local f1=$(grep -oE "Macro-F1[ ]+:[ ]+[0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$" || echo "n/a")
        echo "[job $JOB_IDX OK   ] ${dt}s  F1=$f1  $desc"
    fi
}

# ── Phase 2.2a — none + FL(sqrt_inverse) ─────────────────────────────────────
run_job "P2.2a none+FL(sqrt_inverse)" "p2_none_fl_sqrtinv" \
        src/train.py --sampler none --loss focal --alpha_mode sqrt_inverse \
        --seed 42 --results-csv "$CSV_P2"

# ── Phase 2.1 — TVAE-all ─────────────────────────────────────────────────────
run_job "P2.1a TVAE-all augmentation" "p2_aug_tvae_all" \
        src/augment_tvae.py --seed 42 --route tvae_all

run_job "P2.1b train on TVAE-all" "p2_train_tvae_all" \
        src/train.py --data train_aug_tvae_tvae_all --loss ce --seed 42 \
        --tag tvae_all --results-csv "$CSV_P2"

# ── Phase 2.1 — Interp-all ───────────────────────────────────────────────────
run_job "P2.1c Interp-all augmentation" "p2_aug_interp_all" \
        src/augment_tvae.py --seed 42 --route interp_all

run_job "P2.1d train on Interp-all" "p2_train_interp_all" \
        src/train.py --data train_aug_tvae_interp_all --loss ce --seed 42 \
        --tag interp_all --results-csv "$CSV_P2"

# ── Phase 3.1 — matched-budget ROS/SMOTE/ADASYN ──────────────────────────────
run_job "P3.1a ROS (matched budget)" "p3_ros_matched" \
        src/train.py --sampler ros --budget matched --tau 2000 --n_target 5000 \
        --tag ros_matched --seed 42 --results-csv "$CSV_P3"

run_job "P3.1b SMOTE (matched budget)" "p3_smote_matched" \
        src/train.py --sampler smote --budget matched --tau 2000 --n_target 5000 \
        --tag smote_matched --seed 42 --results-csv "$CSV_P3"

run_job "P3.1c ADASYN (matched budget)" "p3_adasyn_matched" \
        src/train.py --sampler adasyn --budget matched --tau 2000 --n_target 5000 \
        --tag adasyn_matched --seed 42 --results-csv "$CSV_P3"

# ── Phase 3.2 — strict CTGAN (no FORCE_SMOTE) ────────────────────────────────
run_job "P3.2a strict CTGAN augmentation" "p3_aug_ctgan_strict" \
        src/augment_ctgan.py --no-force-smote --seed 42 \
        --out train_aug_ctgan_strict

run_job "P3.2b train on strict CTGAN" "p3_train_ctgan_strict" \
        src/train.py --data train_aug_ctgan_strict --loss ce --seed 42 \
        --tag ctgan_strict --results-csv "$CSV_P3"

echo "[ALL DONE] Phase 2 CSV: $CSV_P2"
echo "[ALL DONE] Phase 3 CSV: $CSV_P3"
