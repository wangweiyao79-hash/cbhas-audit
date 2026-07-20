"""
run_ciciot_grid.py — Unattended orchestrator for the CICIoT2023 external
validation of RQ2 (no-augmentation-arm CE vs Focal Loss) and RQ3 (does the
seed noise floor track test-set support size).

Scope (confirmed 2026-07): sampler="none" only. No ROS/SMOTE/ADASYN/CB-HAS —
those answer RQ1 (augmentation comparison), which is out of scope for this
run. See CB-HAS_..._v6_final.md Section 6 discussion for why.

For each seed: preprocess (per-seed split, cheap if the merge/dedup cache
already exists) -> train {CE, Focal-fixed, Focal-sqrt_inverse}.

Usage:
    python run_ciciot_grid.py                          # default 5 seeds
    python run_ciciot_grid.py --seeds 7 42 123 1337 2024
    python run_ciciot_grid.py --dry-run                # print the plan, don't run
"""
import os
import csv
import sys
import time
import argparse
import subprocess
from datetime import datetime

SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
PROC_DIR = os.path.join(ROOT_DIR, "data", "ciciot_processed")
LOG_DIR  = os.path.join(ROOT_DIR, "results", "logs")
RESULTS_CSV = os.path.join(ROOT_DIR, "results", "tables", "ciciot_results.csv")
PY       = sys.executable

os.makedirs(LOG_DIR, exist_ok=True)

# Same 5 core seeds as the CIC-IDS2017 audit protocol (Section 4.2), so the
# two datasets' seed sets line up for the cross-dataset comparison.
DEFAULT_SEEDS = [7, 42, 123, 1337, 2024]

CONFIGS = [
    # (sampler, loss, alpha_mode, tag)
    ("none", "ce",    None,           "none_ce"),
    ("none", "focal", "fixed",        "none_focal_fixed"),
    ("none", "focal", "sqrt_inverse", "none_focal_sqrtinv"),
]


def completed_tags():
    """Tags already present in the results CSV — used to resume after an
    interruption (e.g. shutdown mid-grid) without redoing finished runs."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    with open(RESULTS_CSV, encoding="utf-8") as f:
        return {row.get("tag", "") for row in csv.DictReader(f)}


def run(cmd, log_path, desc):
    t0 = time.time()
    print(f"[job START] {desc}  ->  {os.path.basename(log_path)}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    status = "OK  " if proc.returncode == 0 else "FAIL"
    print(f"[job {status}] {elapsed:7.1f}s  exit={proc.returncode}  {desc}", flush=True)
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_jobs = len(args.seeds) * (1 + len(CONFIGS))  # 1 preprocess + N configs per seed
    job_i = 0

    print(f"CICIoT2023 grid: {len(args.seeds)} seeds x {len(CONFIGS)} configs "
          f"= {len(args.seeds) * len(CONFIGS)} training runs "
          f"(+ {len(args.seeds)} preprocessing splits)")
    print(f"Seeds: {args.seeds}")
    if args.dry_run:
        for seed in args.seeds:
            print(f"  [preprocess] seed={seed}")
            for sampler, loss, alpha_mode, tag in CONFIGS:
                print(f"  [train]      seed={seed}  tag={tag}")
        return

    failures = []
    for seed in args.seeds:
        job_i += 1
        seed_dir = os.path.join(PROC_DIR, f"seed{seed}")
        if os.path.isdir(seed_dir):
            print(f"[job SKIP ] preprocessing seed={seed} — split already exists at {seed_dir}")
        else:
            log_path = os.path.join(LOG_DIR, f"{ts}_{job_i:02d}_preprocess_seed{seed}.log")
            ok = run([PY, os.path.join(SRC_DIR, "preprocess_ciciot.py"), "--seed", str(seed)],
                     log_path, f"preprocess seed={seed}")
            if not ok:
                failures.append(f"preprocess seed={seed}")
                print(f"  !! preprocessing failed for seed={seed}, skipping its training runs")
                continue

        done = completed_tags()
        for sampler, loss, alpha_mode, tag in CONFIGS:
            job_i += 1
            run_tag = f"{tag}_{seed}"
            if run_tag in done:
                print(f"[job SKIP ] {run_tag} — already in {os.path.basename(RESULTS_CSV)}")
                continue
            cmd = [PY, os.path.join(SRC_DIR, "train_ciciot.py"),
                   "--sampler", sampler, "--loss", loss, "--seed", str(seed),
                   "--tag", run_tag]
            if alpha_mode:
                cmd += ["--alpha_mode", alpha_mode]
            log_path = os.path.join(LOG_DIR, f"{ts}_{job_i:02d}_{run_tag}.log")
            ok = run(cmd, log_path, f"seed={seed}  tag={tag}")
            if not ok:
                failures.append(run_tag)

    print(f"\n{'='*65}")
    if failures:
        print(f"DONE with {len(failures)} failure(s): {failures}")
    else:
        print("DONE — all jobs succeeded.")
    print(f"Results: results/tables/ciciot_results.csv")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
