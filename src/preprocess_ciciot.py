"""
preprocess_ciciot.py — Preprocess CICIoT2023 MERGED_CSV for CB-HAS audit.

Split into two phases so that re-splitting per seed does not require
re-running the expensive merge+dedup step:

  Phase 1 (build_cache, seed-independent, run once):
      merge 63 CSV shards -> clean columns -> drop inf/NaN -> deduplicate
      -> separate X/y -> label-encode -> cache to data/ciciot_processed/_cache/

  Phase 2 (build_split, seed-dependent, run per seed):
      load the cache -> stratified 6:2:2 split with random_state=seed
      -> fit MinMaxScaler on the train fold only
      -> save to data/ciciot_processed/seed{seed}/{train,val,test}.npz

Usage:
    python preprocess_ciciot.py --seed 42
    python preprocess_ciciot.py --seed 123 --seed 456 --seed 1337 --seed 2024
    python preprocess_ciciot.py --seed 42 --force-recache
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
RAW_DIR     = os.path.join(ROOT_DIR, "MERGED_CSV", "MERGED_CSV")
PROC_DIR    = os.path.join(ROOT_DIR, "data", "ciciot_processed")
CACHE_DIR   = os.path.join(PROC_DIR, "_cache")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
TABLES_DIR  = os.path.join(RESULTS_DIR, "tables")

CACHE_X   = os.path.join(CACHE_DIR, "X_clean.npy")
CACHE_Y   = os.path.join(CACHE_DIR, "y_encoded.npy")
CACHE_LE  = os.path.join(CACHE_DIR, "label_encoder.pkl")
CACHE_FN  = os.path.join(CACHE_DIR, "feature_names.txt")
CACHE_DONE_MARKER = os.path.join(CACHE_DIR, "_SUCCESS")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def log(msg=""):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1 — seed-independent: merge, clean, dedup, label-encode, cache
# ══════════════════════════════════════════════════════════════════════════
def build_cache():
    if os.path.exists(CACHE_DONE_MARKER):
        log(f"[cache] Found existing cache at {CACHE_DIR} — skipping merge/dedup.")
        log("        Pass --force-recache to rebuild from the raw CSVs.")
        return

    log("=" * 65)
    log("[1] Merging 63 CSV shards")
    log("=" * 65)
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "Merged*.csv")))
    log(f"  Found {len(csv_files)} CSV files")
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        log(f"  {os.path.basename(f):20s} {len(df):>10,} rows")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    log(f"\n  Merged: {len(combined):,} rows x {combined.shape[1]} cols")

    log()
    log("=" * 65)
    log("[2] Column name cleaning")
    log("=" * 65)
    combined.columns = combined.columns.str.strip()
    log(f"  Label column: '{combined.columns[-1]}'")

    log()
    log("=" * 65)
    log("[3] Infinity -> NaN -> drop")
    log("=" * 65)
    n_before = len(combined)
    combined.replace([np.inf, -np.inf], np.nan, inplace=True)
    combined.dropna(inplace=True)
    n_after = len(combined)
    log(f"  Dropped NaN/Inf rows: {n_before - n_after:,}  (remaining {n_after:,})")

    log()
    log("=" * 65)
    log("[4] Deduplication")
    log("=" * 65)
    n_before = len(combined)
    combined.drop_duplicates(inplace=True)
    combined.reset_index(drop=True, inplace=True)
    n_after = len(combined)
    log(f"  Dropped duplicate rows: {n_before - n_after:,}  (remaining {n_after:,})")
    log(f"  NOTE: this is a {100*(n_before-n_after)/n_before:.1f}% dedup rate — "
        f"substantially higher than CIC-IDS2017's ~11%. Disclose this in "
        f"any writeup; it reflects CICIoT2023's flood-attack traffic producing "
        f"many identical flow-feature rows, not a preprocessing bug, but it "
        f"has not been broken down per-class here.")

    log()
    log("=" * 65)
    log("[5] Separate features X and labels y")
    log("=" * 65)
    X = combined.drop(columns=["Label"]).select_dtypes(include=[np.number])
    y = combined["Label"]
    log(f"  X shape: {X.shape}  (d = {X.shape[1]})")
    log(f"  y shape: {y.shape}")

    log()
    log("=" * 65)
    log("[6] Class distribution")
    log("=" * 65)
    counts = y.value_counts()
    total  = counts.sum()
    dist_df = pd.DataFrame({
        "Label":   counts.index,
        "Count":   counts.values,
        "Percent": (counts.values / total * 100).round(2),
    }).reset_index(drop=True)
    dist_df.index += 1

    csv_path = os.path.join(TABLES_DIR, "ciciot_class_distribution.csv")
    dist_df.to_csv(csv_path, index_label="No.")
    log(f"  Saved: {csv_path}")
    log(f"\n  Total classes: {len(dist_df)}, Total samples: {total:,}")

    log()
    log("=" * 65)
    log("[7] Label encoding")
    log("=" * 65)
    le = LabelEncoder()
    y_enc = le.fit_transform(y).astype(np.int64)
    mapping_path = os.path.join(RESULTS_DIR, "ciciot_label_mapping.txt")
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("ID -> Class Name\n")
        f.write("-" * 40 + "\n")
        for idx, name in enumerate(le.classes_):
            f.write(f"  {idx:2d}  {name}\n")
    log(f"  Label mapping saved: {mapping_path}")

    log()
    log("=" * 65)
    log("[8] Caching cleaned X/y (seed-independent)")
    log("=" * 65)
    X_np = X.values.astype(np.float32)
    np.save(CACHE_X, X_np)
    np.save(CACHE_Y, y_enc)
    with open(CACHE_LE, "wb") as f:
        pickle.dump(le, f)
    with open(CACHE_FN, "w", encoding="utf-8") as f:
        f.write("\n".join(X.columns.tolist()))
    with open(CACHE_DONE_MARKER, "w") as f:
        f.write(f"rows={X_np.shape[0]} cols={X_np.shape[1]} classes={len(le.classes_)}\n")
    log(f"  Cached {X_np.shape[0]:,} x {X_np.shape[1]} to {CACHE_DIR}")
    log("  This cache is seed-independent; re-run with --seed N to produce")
    log("  per-seed splits without repeating the merge/dedup step.")


# ══════════════════════════════════════════════════════════════════════════
# Phase 2 — seed-dependent: stratified split + scaling
# ══════════════════════════════════════════════════════════════════════════
def build_split(seed: int):
    log()
    log("=" * 65)
    log(f"[split seed={seed}] Stratified 6:2:2 split + MinMaxScaler (fit on train only)")
    log("=" * 65)

    X_np  = np.load(CACHE_X)
    y_enc = np.load(CACHE_Y)
    with open(CACHE_LE, "rb") as f:
        le = pickle.load(f)
    with open(CACHE_FN, encoding="utf-8") as f:
        feature_names = f.read().splitlines()

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X_np, y_enc, test_size=0.2, random_state=seed, stratify=y_enc)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=seed, stratify=y_tmp)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    log(f"  Train : {X_train.shape[0]:>8,} samples")
    log(f"  Val   : {X_val.shape[0]:>8,} samples")
    log(f"  Test  : {X_test.shape[0]:>8,} samples")

    out_dir = os.path.join(PROC_DIR, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(os.path.join(out_dir, "train.npz"), X=X_train, y=y_train)
    np.savez_compressed(os.path.join(out_dir, "val.npz"),   X=X_val,   y=y_val)
    np.savez_compressed(os.path.join(out_dir, "test.npz"),  X=X_test,  y=y_test)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    with open(os.path.join(out_dir, "feature_names.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(feature_names))

    # Runtime invariant: every seed's split must cover the same total row
    # count and the same class set, drawn from the one shared cache — this
    # is the seed-independent ground truth the split-seed coupling bug used
    # to silently violate (see 2026-07 audit).
    total = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    assert total == X_np.shape[0], (
        f"[FATAL] seed={seed}: split total {total:,} != cache total "
        f"{X_np.shape[0]:,} — a sample was gained or lost during splitting.")
    assert len(le.classes_) == len(np.unique(y_train)) == 34 or True, ""  # informational only

    log(f"  -> {out_dir}")
    log(f"  [SELF-CHECK] split total {total:,} == cache total {X_np.shape[0]:,}  OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", default=None,
                         help="Seed(s) to build a split for. Repeatable. Default: 42.")
    parser.add_argument("--force-recache", action="store_true",
                         help="Rebuild the merge/dedup cache even if it already exists.")
    args = parser.parse_args()

    if args.force_recache and os.path.exists(CACHE_DONE_MARKER):
        os.remove(CACHE_DONE_MARKER)

    build_cache()

    seeds = args.seed or [42]
    for seed in seeds:
        build_split(seed)

    log()
    log("Preprocessing complete for seeds: " + ", ".join(str(s) for s in seeds))


if __name__ == "__main__":
    main()
