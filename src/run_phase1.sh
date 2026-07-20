#!/usr/bin/env bash
# Phase 1 orchestrator — M1 multi-seed reproducibility, Bash version.
#
# 4 configurations x 5 seeds = 20 trainings.  seed=42 already in
# main_results.csv, so we need 16 new trainings + 4 new TVAE augmentations
# + 4 new CTGAN augmentations.
#
# Configurations:
#   A) none  + CE
#   B) ctgan + CE            (strongest data-level baseline)
#   C) tvae  + CE            (proposed CB-HAS)
#   D) tvae  + FL(sqrt_inverse)
#
# Estimated GPU time: ~4 h on RTX 4060.  Designed to run headless.
#
# Emits per-job progress events to stdout so Monitor can pick them up.

set +e
cd "$(dirname "$0")/.."

PY="./venv/Scripts/python.exe"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

CSV="results/tables/phase1_multiseed.csv"
SEEDS_NEW=(123 2024 7 1337)

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

# ── Step 1: seed the phase1 CSV with the seed=42 rows from main_results.csv ──
"$PY" - <<'PYEOF'
import os, pandas as pd
src = 'results/tables/main_results.csv'
dst = 'results/tables/phase1_multiseed.csv'
df = pd.read_csv(src)
df['alpha_mode'] = df['alpha_mode'].fillna('N/A')
mask = (
    ((df.sampler == 'none')  & (df.loss == 'ce'))                                   |
    ((df.sampler == 'ctgan') & (df.loss == 'ce'))                                   |
    ((df.sampler == 'tvae')  & (df.loss == 'ce'))                                   |
    ((df.sampler == 'tvae')  & (df.loss == 'focal') & (df.alpha_mode == 'sqrt_inverse'))
)
keep = df[(df.seed == 42) & mask]
if os.path.exists(dst):
    existing = pd.read_csv(dst)
    keep = pd.concat([existing, keep], ignore_index=True).drop_duplicates(
        subset=['sampler','loss','alpha_mode','seed'], keep='last')
keep.to_csv(dst, index=False)
print(f'Wrote {len(keep)} rows -> {dst}')
PYEOF

# ── Step 2: for each new seed, run augmentation + 4 trainings ────────────────
for seed in "${SEEDS_NEW[@]}"; do
    echo "[phase1] === seed=$seed ==="

    # 2a. TVAE augmentation for this seed
    run_job "[seed=$seed] TVAE augmentation" "p1_aug_tvae_s${seed}" \
            src/augment_tvae.py --seed $seed

    # 2b. CTGAN augmentation for this seed
    run_job "[seed=$seed] CTGAN augmentation" "p1_aug_ctgan_s${seed}" \
            src/augment_ctgan.py --seed $seed

    # 2c. A) none + CE
    run_job "[seed=$seed] A) none + CE" "p1_none_ce_s${seed}" \
            src/train.py --sampler none --loss ce --seed $seed --results-csv "$CSV"

    # 2d. B) ctgan + CE  (uses train_aug_ctgan_seed{S}.npz)
    run_job "[seed=$seed] B) ctgan + CE" "p1_ctgan_ce_s${seed}" \
            src/train.py --data "train_aug_ctgan_seed${seed}" --loss ce --seed $seed \
            --tag ctgan --results-csv "$CSV"

    # 2e. C) tvae + CE
    run_job "[seed=$seed] C) tvae + CE (CB-HAS)" "p1_tvae_ce_s${seed}" \
            src/train.py --data "train_aug_tvae_seed${seed}" --loss ce --seed $seed \
            --tag tvae --results-csv "$CSV"

    # 2f. D) tvae + FL(sqrt_inverse)
    run_job "[seed=$seed] D) tvae + FL(sqrt_inverse)" "p1_tvae_fl_s${seed}" \
            src/train.py --data "train_aug_tvae_seed${seed}" --loss focal \
            --alpha_mode sqrt_inverse --seed $seed --tag tvae --results-csv "$CSV"
done

echo "[ALL DONE] Results in: $CSV"
echo "[ALL DONE] Next: run  ./venv/Scripts/python.exe src/analyze_phase1.py"
