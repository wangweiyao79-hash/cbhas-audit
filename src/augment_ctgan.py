"""
augment_ctgan.py — Algorithm 1: CTGAN-based targeted augmentation for rare classes.

Pipeline:
  1. Identify rare classes (count < tau)
  2. For each rare class c_k:
       |c_k| >= SMOTE_THRESH (50) → train CTGAN, sample Δn_k rows
       |c_k| <  SMOTE_THRESH      → SMOTE-style linear interpolation (fallback)
  3. Clip synthetic samples to [min, max] of real samples (validity filter)
  4. Merge → data/processed/train_aug.npz
  5. Write results/tables/aug_report.csv
  6. PCA quality plots → results/figures/ctgan_quality/
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
matplotlib.use("Agg")          # non-interactive, safe on Windows servers
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch

sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
PROC_DIR    = os.path.join(ROOT_DIR, "data", "processed")
CONFIG_PATH = os.path.join(SRC_DIR, "config.yaml")
TABLES_DIR  = os.path.join(ROOT_DIR, "results", "tables")
FIG_DIR     = os.path.join(ROOT_DIR, "results", "figures", "ctgan_quality")

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIG_DIR,    exist_ok=True)

# CTGAN is unreliable on extremely few samples; fall back to interpolation below this.
SMOTE_THRESH = 50

# Classes forced to SMOTE regardless of sample count (poor CTGAN quality confirmed by PCA).
FORCE_SMOTE_NAMES = ["Bot", "XSS"]


def log(msg=""):
    print(msg, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# SMOTE-style linear interpolation fallback
# ══════════════════════════════════════════════════════════════════════════════
def smote_interpolate(X_real: np.ndarray, n_needed: int, seed: int) -> np.ndarray:
    """Random convex interpolation between pairs of real samples.
    Safe for very small classes (even 2 samples works).
    """
    rng = np.random.RandomState(seed)
    n   = len(X_real)
    syn = []
    for _ in range(n_needed):
        i, j  = rng.choice(n, size=2, replace=(n < 2))
        alpha = rng.uniform(0.0, 1.0)
        syn.append(X_real[i] + alpha * (X_real[j] - X_real[i]))
    return np.array(syn, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# CTGAN training + sampling
# ══════════════════════════════════════════════════════════════════════════════
def run_ctgan(X_real: np.ndarray, n_needed: int,
              feature_names: list, epochs: int, batch_size: int,
              cuda: bool, cls_name: str, seed: int = 42) -> np.ndarray:
    """Train a dedicated CTGAN on one class's real samples, then sample."""
    from ctgan import CTGAN   # ctgan 0.12.x exports CTGAN (not CTGANSynthesizer)

    df = pd.DataFrame(X_real.astype(np.float64), columns=feature_names)

    log(f"    epochs={epochs}  batch_size={batch_size}  "
        f"n_real={len(X_real)}  n_needed={n_needed}  gpu={cuda}  seed={seed}")

    # Fix RNGs before construction + fit for per-seed reproducibility.
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    t0 = time.time()
    synthesizer = CTGAN(
        epochs=epochs,
        batch_size=batch_size,
        verbose=True,       # per-epoch loss lines confirm training is live
        enable_gpu=cuda,    # 'cuda' param is deprecated in 0.12.x → use enable_gpu
    )
    # All 78 features are continuous — discrete_columns must be empty
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
    """Clip each feature of synthetic samples to [min, max] of the real class."""
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
               label=f"CTGAN (n={len(X_syn):,})", zorder=2)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title(f"Real vs CTGAN synthetic — {cls_name}")
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
                        help="Override config seed for CTGAN and SMOTE fallback.")
    parser.add_argument("--no-force-smote", action="store_true",
                        help="Disable the FORCE_SMOTE override on Bot/XSS "
                             "(strict CTGAN-only comparison, addresses M6).")
    parser.add_argument("--out", default=None,
                        help="Output NPZ basename (without .npz) under data/processed/. "
                             "Default: train_aug[_seed{S}][_strict].")
    parser.add_argument("--pca", action="store_true",
                        help="Emit PCA quality plots (skipped by default).")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    tau          = cfg["tau"]           # rare-class threshold
    n_target     = cfg["n_target"]      # target samples per rare class
    ctgan_epochs = cfg["ctgan_epochs"]
    ctgan_batch  = cfg["ctgan_batch"]
    seed         = args.seed if args.seed is not None else cfg.get("seed", 42)
    cuda         = torch.cuda.is_available()

    force_smote_names = [] if args.no_force_smote else FORCE_SMOTE_NAMES

    if args.out is not None:
        out_stem = args.out
    else:
        parts = ["train_aug"]
        if args.no_force_smote:
            parts.append("ctgan_strict")
        else:
            parts.append("ctgan")
        if seed != 42:
            parts.append(f"seed{seed}")
        # Backward compat: the original paper output is "train_aug.npz"; only
        # emit the plain name when this run matches the paper configuration.
        if not args.no_force_smote and seed == 42 and args.out is None:
            out_stem = "train_aug"
        else:
            out_stem = "_".join(parts)

    log("=" * 65)
    log("[Config]")
    log(f"  tau={tau}  n_target={n_target}  "
        f"ctgan_epochs={ctgan_epochs}  ctgan_batch={ctgan_batch}")
    log(f"  SMOTE fallback threshold : < {SMOTE_THRESH} samples")
    log(f"  FORCE_SMOTE overrides    : {force_smote_names}")
    gpu_name = torch.cuda.get_device_name(0) if cuda else "CPU only"
    log(f"  GPU  : {cuda}  ({gpu_name})")
    log(f"  Seed : {seed}")
    log(f"  Out  : data/processed/{out_stem}.npz")

    # ── Load training data (only — val/test untouched) ────────────────────────
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

    # ── Identify rare classes ─────────────────────────────────────────────────
    counts = Counter(y_train.tolist())
    rare   = {c: n for c, n in counts.items() if n < tau}

    log(f"\n[Rare Classes]  tau={tau}  →  {len(rare)} classes to augment")
    log(f"  {'#':<3} {'Class':<40} {'n_real':>7}  {'Method':<16}  {'Δn':>6}")
    log(f"  {'-'*3} {'-'*40} {'-'*7}  {'-'*16}  {'-'*6}")
    for i, cls_id in enumerate(sorted(rare), 1):
        n   = rare[cls_id]
        met = "CTGAN" if n >= SMOTE_THRESH else "SMOTE_fallback"
        log(f"  {i:<3} {le.classes_[cls_id]:<40} {n:>7,}  {met:<16}  {n_target-n:>6,}")

    # ── Augment each rare class ───────────────────────────────────────────────
    X_syn_list, y_syn_list, report_rows = [], [], []
    ctgan_store = []     # (cls_id, cls_name, X_real, X_syn) for PCA plots

    for cls_id in sorted(rare):
        cls_name = le.classes_[cls_id]
        n_real   = rare[cls_id]
        n_needed = n_target - n_real
        X_cls    = X_train[y_train == cls_id]

        log(f"\n{'─'*65}")
        log(f"[{cls_name}]  n_real={n_real}  n_needed={n_needed}")

        force_smote = any(pat in cls_name for pat in force_smote_names)
        if n_real >= SMOTE_THRESH and not force_smote:
            method = "CTGAN"
            X_syn  = run_ctgan(X_cls, n_needed, feature_names,
                               ctgan_epochs, ctgan_batch, cuda, cls_name, seed=seed)
            ctgan_store.append((cls_id, cls_name, X_cls.copy(), X_syn.copy()))
        else:
            reason = f"< {SMOTE_THRESH} samples" if n_real < SMOTE_THRESH else "FORCE_SMOTE override"
            method = "SMOTE_fallback"
            log(f"  {reason} → SMOTE interpolation fallback")
            X_syn = smote_interpolate(X_cls, n_needed, seed=seed)

        # Validity filter
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

    # ── Merge with original training data ────────────────────────────────────
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

    # ── Augmentation report ───────────────────────────────────────────────────
    df_rep = pd.DataFrame(report_rows)
    rep_stem = out_stem.replace("train_aug_", "").replace("train_aug", "ctgan_paper")
    rep_path = os.path.join(TABLES_DIR, f"aug_report_{rep_stem}.csv")
    df_rep.to_csv(rep_path, index=False)
    log(f"\n[Report] {rep_path}")
    log(f"  {'Class':<40} {'before':>7}  {'added':>7}  {'after':>7}  {'method'}")
    log(f"  {'-'*40} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*16}")
    for row in report_rows:
        log(f"  {row['class_name']:<40} {row['n_before']:>7,}  "
            f"{row['n_added']:>7,}  {row['n_after']:>7,}  {row['method']}")

    # ── PCA quality plots (CTGAN classes only) ────────────────────────────────
    if args.pca and ctgan_store:
        log(f"\n[PCA Quality Plots]  {len(ctgan_store)} CTGAN-augmented classes")
        for cls_id, cls_name, X_real, X_syn in ctgan_store:
            safe  = cls_name.replace(" ", "_").replace("-", "_")
            fpath = os.path.join(FIG_DIR, f"pca_{safe}.png")
            save_pca_plot(X_real, X_syn, cls_name, fpath)

    # ── Final distribution ────────────────────────────────────────────────────
    log(f"\n[Final Distribution]  train_aug.npz")
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
