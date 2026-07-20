"""
augment_tvae.py — TVAE-based targeted augmentation (stable, reproducible variant).

Replaces CTGAN with TVAE (VAE-based, single-loss objective → more stable convergence
and bit-exact reproducibility when global seeds are fixed before each fit call).

Pipeline:
  1. Identify rare classes (count < tau=2000)
  2. For each rare class c_k:
       |c_k| >= SMOTE_THRESH (50) → train TVAE (seed-fixed), sample Δn_k rows
       |c_k| <  SMOTE_THRESH      → SMOTE-style linear interpolation
  3. Clip synthetic samples to [min, max] of real samples (validity filter)
  4. Merge → data/processed/train_aug_tvae.npz
  5. Write results/tables/aug_report_tvae.csv
  6. PCA quality plots → results/figures/tvae_quality/
"""

import os
import sys
import time
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch

sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
PROC_DIR    = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_PATH = os.path.join(SRC_DIR, "config.yaml")
TABLES_DIR  = os.path.join(ROOT_DIR, "results", "tables")
FIG_DIR     = os.path.join(ROOT_DIR, "results", "figures", "tvae_quality")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)


def log(msg=""):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SMOTE-style linear interpolation fallback
# ══════════════════════════════════════════════════════════════════════════════
def smote_interpolate(X_real: np.ndarray, n_needed: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n   = len(X_real)
    syn = []
    for _ in range(n_needed):
        i, j  = rng.choice(n, size=2, replace=(n < 2))
        alpha = rng.uniform(0.0, 1.0)
        syn.append(X_real[i] + alpha * (X_real[j] - X_real[i]))
    return np.array(syn, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# TVAE training + sampling (reproducible via fixed seed)
# ══════════════════════════════════════════════════════════════════════════════
def run_tvae(X_real: np.ndarray, n_needed: int,
             feature_names: list, epochs: int, batch_size: int,
             cuda: bool, cls_name: str, seed: int) -> np.ndarray:
    """Train a dedicated TVAE on one class's real samples with a fixed seed."""
    from ctgan import TVAE

    df = pd.DataFrame(X_real.astype(np.float64), columns=feature_names)

    log(f"    epochs={epochs}  batch_size={batch_size}  "
        f"n_real={len(X_real)}  n_needed={n_needed}  gpu={cuda}  seed={seed}")

    # Fix all RNG sources before construction + fit for full reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    synthesizer = TVAE(
        epochs=epochs,
        batch_size=batch_size,
        cuda=cuda,
    )
    synthesizer.fit(df, discrete_columns=[])
    log(f"    Training done in {time.time()-t0:.1f}s  →  sampling {n_needed} rows ...")

    t1 = time.time()
    df_syn = synthesizer.sample(n_needed)
    log(f"    Sampling done in {time.time()-t1:.1f}s")

    return df_syn.values.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Validity filter
# ══════════════════════════════════════════════════════════════════════════════
def clip_to_real_range(X_syn: np.ndarray, X_real: np.ndarray) -> np.ndarray:
    return np.clip(X_syn,
                   X_real.min(axis=0),
                   X_real.max(axis=0)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# PCA quality plot
# ══════════════════════════════════════════════════════════════════════════════
def save_pca_plot(X_real, X_syn, cls_name, save_path):
    pca   = PCA(n_components=2, random_state=42)
    X_2d  = pca.fit_transform(np.vstack([X_real, X_syn]))
    nr    = len(X_real)
    var   = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_2d[:nr, 0], X_2d[:nr, 1],
               c="steelblue", s=30, alpha=0.7,
               label=f"Real  (n={nr:,})", zorder=3)
    ax.scatter(X_2d[nr:, 0], X_2d[nr:, 1],
               c="tomato", s=8, alpha=0.25, marker="x",
               label=f"TVAE  (n={len(X_syn):,})", zorder=2)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(f"Real vs TVAE synthetic — {cls_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    log(f"    Saved: {os.path.basename(save_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed for both TVAE and SMOTE fallback.")
    parser.add_argument("--route", default="default",
                        choices=["default", "tvae_all", "interp_all"],
                        help="default = paper policy (TVAE if n>=tau_s else SMOTE); "
                             "tvae_all = force TVAE for every rare class (M3 ablation); "
                             "interp_all = force SMOTE interpolation for every rare class.")
    parser.add_argument("--tau_s", type=int, default=50,
                        help="Routing threshold: below this, fall back to SMOTE.")
    parser.add_argument("--tau", type=int, default=None,
                        help="Rare-class threshold (default: from config.yaml).")
    parser.add_argument("--n_target", type=int, default=None,
                        help="Target sample count per rare class (default: from config.yaml).")
    parser.add_argument("--out", default=None,
                        help="Output NPZ basename (without .npz) under data/processed/. "
                             "Default: train_aug_tvae[_seed{S}][_{route}].")
    parser.add_argument("--pca", action="store_true",
                        help="Emit PCA quality plots (skipped by default for speed).")
    args = parser.parse_args()

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    tau          = args.tau if args.tau is not None else cfg["tau"]
    n_target     = args.n_target if args.n_target is not None else cfg["n_target"]
    tvae_epochs  = cfg["ctgan_epochs"]   # reuse ctgan_epochs key for TVAE
    tvae_batch   = cfg["ctgan_batch"]
    seed         = args.seed if args.seed is not None else cfg.get("seed", 42)
    smote_thresh = args.tau_s
    route        = args.route
    cuda         = torch.cuda.is_available()

    # Route override: tvae_all → force every rare class into TVAE branch (tau_s=1);
    # interp_all → force every rare class into SMOTE branch (tau_s=inf).
    if route == "tvae_all":
        smote_thresh = 1
    elif route == "interp_all":
        smote_thresh = 10**9

    # Output NPZ name embeds seed and route when they diverge from the paper default.
    if args.out is not None:
        out_stem = args.out
    else:
        parts = ["train_aug_tvae"]
        if route != "default":
            parts.append(route)
        if seed != 42:
            parts.append(f"seed{seed}")
        if args.tau_s != 50 and route == "default":
            parts.append(f"taus{args.tau_s}")
        if args.n_target is not None and args.n_target != cfg["n_target"]:
            parts.append(f"N{args.n_target}")
        out_stem = "_".join(parts)

    log("=" * 65)
    log("[Config]  TVAE augmentation")
    log(f"  tau={tau}  n_target={n_target}  "
        f"tvae_epochs={tvae_epochs}  tvae_batch={tvae_batch}")
    log(f"  Route : {route}  (effective SMOTE fallback threshold : < {smote_thresh})")
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU only"
    log(f"  GPU  : {cuda}  ({gpu_name})")
    log(f"  Seed : {seed}  (fixed before every TVAE fit call)")
    log(f"  Out  : data/processed/{out_stem}.npz")

    log("\n[Load] data/processed/train.npz")
    d       = np.load(os.path.join(PROC_DIR, "train.npz"))
    X_train = d["X"].astype(np.float32)
    y_train = d["y"].astype(np.int64)
    log(f"  X={X_train.shape}  y={y_train.shape}")

    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    feat_path = os.path.join(PROC_DIR, "feature_names.txt")
    with open(feat_path, encoding="utf-8") as f:
        feature_names = [ln.strip() for ln in f]
    log(f"  Feature names loaded: {len(feature_names)} columns")

    counts = Counter(y_train.tolist())
    rare   = {c: n for c, n in counts.items() if n < tau}

    log(f"\n[Rare Classes]  tau={tau}  →  {len(rare)} classes to augment")
    log(f"  {'#':<3} {'Class':<40} {'n_real':>7}  {'Method':<16}  {'Δn':>6}")
    log(f"  {'-'*3} {'-'*40} {'-'*7}  {'-'*16}  {'-'*6}")
    for i, cls_id in enumerate(sorted(rare), 1):
        n   = rare[cls_id]
        met = "TVAE" if n >= smote_thresh else "SMOTE_fallback"
        log(f"  {i:<3} {le.classes_[cls_id]:<40} {n:>7,}  {met:<16}  {n_target-n:>6,}")

    X_syn_list, y_syn_list, report_rows = [], [], []
    tvae_store = []

    for cls_id in sorted(rare):
        cls_name = le.classes_[cls_id]
        n_real   = rare[cls_id]
        n_needed = n_target - n_real
        X_cls    = X_train[y_train == cls_id]

        log(f"\n{'─'*65}")
        log(f"[{cls_name}]  n_real={n_real}  n_needed={n_needed}")

        if n_real >= smote_thresh:
            method = "TVAE"
            X_syn  = run_tvae(X_cls, n_needed, feature_names,
                              tvae_epochs, tvae_batch, cuda, cls_name, seed)
            tvae_store.append((cls_id, cls_name, X_cls.copy(), X_syn.copy()))
        else:
            method = "SMOTE_fallback"
            log(f"  Only {n_real} samples (< {smote_thresh}) → SMOTE interpolation fallback")
            X_syn = smote_interpolate(X_cls, n_needed, seed=seed)

        X_syn = clip_to_real_range(X_syn, X_cls)
        log(f"  Clipped to real range → synthetic shape {X_syn.shape}")

        X_syn_list.append(X_syn)
        y_syn_list.append(np.full(n_needed, cls_id, dtype=np.int64))
        report_rows.append({
            "class_id"  : int(cls_id),
            "class_name": cls_name,
            "n_before"  : int(n_real),
            "n_added"   : int(n_needed),
            "n_after"   : int(n_real + n_needed),
            "method"    : method,
        })

    log(f"\n{'─'*65}")
    log("[Merge]")
    X_aug = np.vstack([X_train] + X_syn_list)
    y_aug = np.concatenate([y_train] + y_syn_list)
    n_added_total = sum(len(x) for x in X_syn_list)
    log(f"  Original  : {X_train.shape[0]:>10,}")
    log(f"  Synthetic : {n_added_total:>10,}")
    log(f"  Augmented : {X_aug.shape[0]:>10,}  ×  {X_aug.shape[1]}")

    out_path = os.path.join(PROC_DIR, f"{out_stem}.npz")
    np.savez_compressed(out_path, X=X_aug, y=y_aug)
    log(f"  Saved → {out_path}")

    df_rep   = pd.DataFrame(report_rows)
    rep_path = os.path.join(TABLES_DIR, f"aug_report_{out_stem.replace('train_aug_', '')}.csv")
    df_rep.to_csv(rep_path, index=False)
    log(f"\n[Report] {rep_path}")
    log(f"  {'Class':<40} {'before':>7}  {'added':>7}  {'after':>7}  {'method'}")
    log(f"  {'-'*40} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*16}")
    for row in report_rows:
        log(f"  {row['class_name']:<40} {row['n_before']:>7,}  "
            f"{row['n_added']:>7,}  {row['n_after']:>7,}  {row['method']}")

    if args.pca and tvae_store:
        log(f"\n[PCA Quality Plots]  {len(tvae_store)} TVAE-augmented classes")
        for cls_id, cls_name, X_real, X_syn in tvae_store:
            safe  = cls_name.replace(" ", "_").replace("-", "_")
            fpath = os.path.join(FIG_DIR, f"pca_{safe}.png")
            save_pca_plot(X_real, X_syn, cls_name, fpath)

    log(f"\n[Final Distribution]  train_aug_tvae.npz")
    counts_aug = Counter(y_aug.tolist())
    log(f"  {'Class':<40} {'Count':>10}")
    log(f"  {'-'*40} {'-'*10}")
    for cls_id in sorted(counts_aug):
        marker = " ← augmented" if cls_id in rare else ""
        log(f"  {le.classes_[cls_id]:<40} {counts_aug[cls_id]:>10,}{marker}")
    log(f"\n  Total: {len(y_aug):,}")
    log("\nDone.")


if __name__ == "__main__":
    main()
