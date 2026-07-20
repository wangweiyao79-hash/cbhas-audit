#!/usr/bin/env bash
# Extra-seed variance check for M3/M5/M6 (currently single-seed=42 point estimates).
# Adds seeds 123 and 2024 for: tvae_all, interp_all, ros_matched, smote_matched,
# adasyn_matched, ctgan_strict. Appends into the existing phase2/phase3 CSVs
# (train.py dedupes on sampler/loss/alpha_mode/seed, so this is safe to re-run).

set +e
cd "$(dirname "$0")/.."

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
        local f1=$(grep -oE "Macro-F1[ ]+:[ ]+[0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+$" || echo "n/a")
        echo "[job $JOB_IDX OK   ] ${dt}s  F1=$f1  $desc"
    fi
}

for seed in 123 2024; do
    echo "=== seed=$seed ==="

    run_job "[seed=$seed] TVAE-all augmentation" "p2x_aug_tvae_all_s${seed}" \
            src/augment_tvae.py --seed $seed --route tvae_all --out train_aug_tvae_all_s${seed}

    run_job "[seed=$seed] train on TVAE-all" "p2x_train_tvae_all_s${seed}" \
            src/train.py --data train_aug_tvae_all_s${seed} --loss ce --seed $seed \
            --tag tvae_all --results-csv "$CSV_P2"

    run_job "[seed=$seed] Interp-all augmentation" "p2x_aug_interp_all_s${seed}" \
            src/augment_tvae.py --seed $seed --route interp_all --out train_aug_interp_all_s${seed}

    run_job "[seed=$seed] train on Interp-all" "p2x_train_interp_all_s${seed}" \
            src/train.py --data train_aug_interp_all_s${seed} --loss ce --seed $seed \
            --tag interp_all --results-csv "$CSV_P2"

    run_job "[seed=$seed] ROS (matched budget)" "p3x_ros_matched_s${seed}" \
            src/train.py --sampler ros --budget matched --tau 2000 --n_target 5000 \
            --tag ros_matched --seed $seed --results-csv "$CSV_P3"

    run_job "[seed=$seed] SMOTE (matched budget)" "p3x_smote_matched_s${seed}" \
            src/train.py --sampler smote --budget matched --tau 2000 --n_target 5000 \
            --tag smote_matched --seed $seed --results-csv "$CSV_P3"

    run_job "[seed=$seed] ADASYN (matched budget)" "p3x_adasyn_matched_s${seed}" \
            src/train.py --sampler adasyn --budget matched --tau 2000 --n_target 5000 \
            --tag adasyn_matched --seed $seed --results-csv "$CSV_P3"

    run_job "[seed=$seed] strict CTGAN augmentation" "p3x_aug_ctgan_strict_s${seed}" \
            src/augment_ctgan.py --no-force-smote --seed $seed --out train_aug_ctgan_strict_s${seed}

    run_job "[seed=$seed] train on strict CTGAN" "p3x_train_ctgan_strict_s${seed}" \
            src/train.py --data train_aug_ctgan_strict_s${seed} --loss ce --seed $seed \
            --tag ctgan_strict --results-csv "$CSV_P3"
done

echo "[ALL DONE] Phase 2 CSV: $CSV_P2"
echo "[ALL DONE] Phase 3 CSV: $CSV_P3"
