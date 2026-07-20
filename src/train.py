"""
train.py — universal training script for CIC-IDS2017 baselines.

Usage:
    python src/train.py --sampler {none,ros,smote,adasyn,ctgan} \
                        --loss    {ce,focal}                     \
                        --seed    42
"""
import os
import sys
import copy
import time
import pickle
import argparse
import yaml
import numpy as np
import pandas as pd
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, precision_score, recall_score)
from imblearn.over_sampling  import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
PROC_DIR    = os.path.join(ROOT_DIR, "data", "processed")
TABLES_DIR  = os.path.join(ROOT_DIR, "results", "tables")
CKPT_DIR    = os.path.join(ROOT_DIR, "checkpoints")
CONFIG_PATH = os.path.join(SRC_DIR, "config.yaml")
RESULTS_CSV = os.path.join(TABLES_DIR, "main_results.csv")

sys.path.insert(0, SRC_DIR)
from model import CNNBiLSTM, count_params

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Focal Loss  FL = -(1-p_t)^γ · log(p_t)
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    alpha: (C,) inverse-frequency weights, normalized to sum=1; None → no weighting.
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # registered as plain attribute (moved to device by caller)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, reduction="none")   # -log(p_t), shape (B,)
        pt    = torch.exp(-ce)                                  # p_t
        w     = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            w = w * self.alpha[targets]
        return (w * ce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# G-mean (multiclass) = geometric mean of per-class recall
# ══════════════════════════════════════════════════════════════════════════════
def gmean_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    per_cls = recall_score(y_true, y_pred, average=None, zero_division=0)
    # Additive smoothing (+1e-6) prevents a single zero-recall class from
    # collapsing the geometric mean to 0 while barely affecting non-zero values.
    return float(np.exp(np.mean(np.log(per_cls + 1e-6))))


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════
def load_data():
    tr  = np.load(os.path.join(PROC_DIR, "train.npz"))
    val = np.load(os.path.join(PROC_DIR, "val.npz"))
    te  = np.load(os.path.join(PROC_DIR, "test.npz"))
    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return (tr["X"],  tr["y"].astype(np.int64),
            val["X"], val["y"].astype(np.int64),
            te["X"],  te["y"].astype(np.int64),
            le)


def make_loader(X: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X).float(),
                       torch.from_numpy(y).long())
    # num_workers=0: avoids Windows multiprocessing issues in DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


# ══════════════════════════════════════════════════════════════════════════════
# Resampling
#
# MEMORY NOTE (SMOTE / ADASYN):
#   The training set contains ~1.5 M samples. SMOTE and ADASYN build a KNN
#   graph over the entire feature matrix, which requires O(n·d) memory and
#   can easily exceed available RAM.
#
#   Mitigation: before any over-sampler we cap every class that exceeds
#   CAP=50,000 samples using RandomUnderSampler.  This reduces the corpus to
#   ≤ 15 × 50,000 = 750,000 rows before KNN is computed.
#
#   The same cap is applied to ROS for a fair head-to-head comparison.
#   The 'downsampled' flag is recorded in main_results.csv.
# ══════════════════════════════════════════════════════════════════════════════
CAP = 50_000


def _apply_majority_cap(X: np.ndarray, y: np.ndarray, seed: int):
    """Cap every class that has more than CAP samples down to CAP."""
    counts   = Counter(y)
    strategy = {c: min(n, CAP) for c, n in counts.items()}
    rus      = RandomUnderSampler(sampling_strategy=strategy, random_state=seed)
    Xr, yr   = rus.fit_resample(X, y)
    print(f"    → after majority cap (≤{CAP:,}/class): {len(yr):,} samples")
    return Xr, yr.astype(np.int64)


def resample(X: np.ndarray, y: np.ndarray,
             sampler_name: str, seed: int,
             budget: str = "default", n_target: int = 5000, tau: int = 2000):
    """
    Apply resampling to the training set only.

    Parameters
    ----------
    budget    : "default"  — imbalanced-learn default (balance all classes to majority);
                "matched"  — mirror CB-HAS budget: only classes with n_k < tau are
                             lifted, each to exactly n_target samples (addresses M5).

    Returns
    -------
    X_res, y_res  : resampled feature/label arrays
    downsampled   : True when RandomUnderSampler was used to cap the majority
    """
    if sampler_name == "none":
        # No resampling at all — used for the none+CE / none+FL(...) arms.
        # Self-check anyway: proves this path never touches the CAP/downsample
        # logic below, since it returns before that code is reached.
        counts = Counter(y)
        print(f"    [SELF-CHECK] sampler=none: no resampling applied. "
              f"Training class distribution ({len(y):,} rows):")
        for c in sorted(counts):
            print(f"      class {c}: {counts[c]:,}")
        assert counts[0] == 1_257_033, (
            f"[FATAL] BENIGN count is {counts[0]:,}, expected 1,257,033 — "
            f"unexpected mutation of the raw training set.")
        return X, y, False

    if sampler_name == "ctgan":
        raise NotImplementedError(
            "CTGAN sampler: run generate_ctgan.py first to create synthetic "
            "data, then load the augmented training set here.")

    counts_before = Counter(y)

    # ── Majority-class cap: ONLY for the legacy "default" budget path ──────────
    # Skipped entirely for budget="matched". Empirically verified (2026-07-10)
    # that ROS/SMOTE/ADASYN with a sampling_strategy dict restricted to C_aug
    # only touch the targeted classes — imbalanced-learn's SMOTE/ADASYN fit a
    # neighbor structure per target class (ADASYN's density estimate is the
    # only step that looks at the full array, and it completed in ~10s on the
    # full 1.51M-row training set with no memory issue). Capping the majority
    # class here would silently shrink BENIGN/DoS Hulk/DDoS/PortScan by up to
    # 87%, confounding any comparison against CB-HAS (which never downsamples
    # the majority class) — see M5 fairness re-audit, Section 4.5.
    need_cap = False
    if budget != "matched":
        need_cap = any(n > CAP for n in counts_before.values())
        if need_cap:
            print(f"    Majority class > {CAP:,}. "
                  f"Downsampling before {sampler_name.upper()} (memory safety) ...")
            X, y = _apply_majority_cap(X, y, seed)

    # Compute k so that SMOTE/ADASYN never exceeds the smallest class size
    min_count = min(Counter(y).values())
    k         = min(5, min_count - 1)

    # Build sampling_strategy dict for --budget matched (M5):
    # only classes with n_k < tau are lifted, each to exactly n_target samples.
    # imbalanced-learn requires target counts >= current counts.
    if budget == "matched":
        counts = Counter(y)
        strategy = {}
        for cid, n in counts.items():
            if n < tau:
                strategy[cid] = max(n, n_target)   # lift to n_target (never shrink)
        if not strategy:
            print(f"    [WARN] --budget matched but no class below tau={tau}; "
                  "falling back to default balancing.")
            budget = "default"
        else:
            print(f"    Budget = matched, no majority cap  (tau={tau}, N_target={n_target})")
            print(f"    Sampling strategy: {len(strategy)} classes to be lifted")
            for cid, tgt in sorted(strategy.items()):
                print(f"      class {cid}: {counts[cid]:,} → {tgt:,}")
    else:
        strategy = "auto"      # imbalanced-learn default: balance to majority
        print(f"    Budget = default (imbalanced-learn 'auto' → balance to majority)")

    if sampler_name == "ros":
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)

    elif sampler_name == "smote":
        print(f"    SMOTE  k_neighbors={k}  (smallest class: {min_count} samples)")
        sampler = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)

    elif sampler_name == "adasyn":
        print(f"    ADASYN n_neighbors={k}  (smallest class: {min_count} samples)")
        sampler = ADASYN(sampling_strategy=strategy, n_neighbors=k, random_state=seed)

    else:
        raise ValueError(f"Unknown sampler: {sampler_name!r}")

    try:
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as exc:
        # ADASYN can fail when a class is already in a clean region; fall back
        print(f"    [WARN] {sampler_name.upper()} failed ({exc}). "
              "Falling back to RandomOverSampler.")
        X_res, y_res = RandomOverSampler(sampling_strategy=strategy,
                                          random_state=seed).fit_resample(X, y)

    print(f"    → after {sampler_name.upper()}: {len(y_res):,} samples")

    # ── Self-check for the matched-budget path: prove no non-augmented class ───
    # was altered, and log the full post-resample distribution for the paper's
    # reproducibility statement (Section 4.9).
    if budget == "matched":
        counts_after = Counter(y_res)
        untouched_ok = all(counts_after[c] == counts_before[c]
                            for c in counts_before if c not in strategy)
        # ADASYN's synthetic count per class is density-adaptive and only
        # approximates n_target (typically within a few percent); ROS/SMOTE
        # hit it exactly. Allow a tolerance band so the assertion is meaningful
        # for all three samplers without being ADASYN-brittle.
        aug_ok = all(abs(counts_after[c] - n_target) <= max(50, 0.05 * n_target)
                     for c in strategy)
        print(f"    [SELF-CHECK] full post-resample class distribution:")
        for c in sorted(counts_after):
            tag = "AUG" if c in strategy else "unchanged"
            before_n = counts_before.get(c, 0)
            print(f"      class {c}: {before_n:,} -> {counts_after[c]:,}  [{tag}]")
        print(f"    [SELF-CHECK] non-augmented classes byte-for-byte preserved: {untouched_ok}")
        print(f"    [SELF-CHECK] augmented classes within tolerance of N_target={n_target}: {aug_ok}")
        assert untouched_ok, (
            f"[FATAL] {sampler_name.upper()} matched-budget resample altered a "
            f"class outside C_aug — majority-class contamination regression.")
        assert aug_ok, (
            f"[FATAL] {sampler_name.upper()} matched-budget resample missed "
            f"N_target={n_target:,} for an augmented class beyond tolerance.")
        assert counts_after[0] == counts_before[0], (
            f"[FATAL] BENIGN count changed: {counts_before[0]:,} -> {counts_after[0]:,}")

    return X_res, y_res.astype(np.int64), need_cap


# ══════════════════════════════════════════════════════════════════════════════
# Training / evaluation loops
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total, n = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
        n     += len(yb)
    return total / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for Xb, yb in loader:
        preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(preds)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, class_names) -> dict:
    # ── Primary metrics (5) ───────────────────────────────────────────────────
    m = {
        "accuracy"          : float(accuracy_score(y_true, y_pred)),
        "macro_precision"   : float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall"      : float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "macro_f1"          : float(f1_score(y_true, y_pred, average="macro",        zero_division=0)),
        "balanced_accuracy" : float(balanced_accuracy_score(y_true, y_pred)),
    }
    # ── Auxiliary ─────────────────────────────────────────────────────────────
    # G-mean: geometric mean of per-class recall with ε=1e-6 smoothing to
    # prevent a single zero-recall class from collapsing the whole metric to 0.
    m["gmean"]       = gmean_score(y_true, y_pred)
    m["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # ── Per-class Recall and Precision (for Table 5 analysis) ─────────────────
    per_recall    = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    for name, r, p in zip(class_names, per_recall, per_precision):
        safe = name.replace(" ", "_").replace("-", "_")
        m[f"recall_{safe}"]    = round(float(r), 6)
        m[f"precision_{safe}"] = round(float(p), 6)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler",    default="none",
                        choices=["none", "ros", "smote", "adasyn", "ctgan"])
    parser.add_argument("--data",       default=None,
                        help="NPZ basename in data/processed/ to use as training set "
                             "(e.g. 'train_aug_tvae'). Overrides --sampler for data loading.")
    parser.add_argument("--loss",       default="ce",
                        choices=["ce", "focal"])
    parser.add_argument("--alpha_mode", default="inverse",
                        choices=["inverse", "sqrt_inverse", "fixed"],
                        help="Alpha-weighting for Focal Loss: "
                             "inverse=1/n (original), "
                             "sqrt_inverse=1/sqrt(n) (milder), "
                             "fixed=0.25 uniform (original paper default).")
    parser.add_argument("--budget",     default="default",
                        choices=["default", "matched"],
                        help="Sampling budget for ROS/SMOTE/ADASYN. "
                             "default = imbalanced-learn 'auto' (balance to majority); "
                             "matched = only lift classes below tau to N_target "
                             "(mirrors CB-HAS budget for the M5 fairness check).")
    parser.add_argument("--tau",        type=int, default=2000,
                        help="Rare-class threshold used when --budget matched.")
    parser.add_argument("--n_target",   type=int, default=5000,
                        help="Target sample count when --budget matched.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--tag",        default=None,
                        help="Optional override for the (sampler) key in results CSV; "
                             "use to keep matched/default budget runs from clobbering each "
                             "other, e.g. --tag ros_matched.")
    parser.add_argument("--results-csv", default=None,
                        help="Absolute or relative path to CSV file for row insertion "
                             "(default: results/tables/main_results.csv).")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; load saved checkpoint and re-evaluate.")
    args = parser.parse_args()

    # Derive effective sampler name used in checkpoints and CSV.
    # Priority: explicit --tag > --data-derived name > --sampler.
    if args.tag is not None:
        effective_sampler = args.tag
    elif args.data is not None:
        effective_sampler = args.data.replace("train_aug_", "")
    else:
        effective_sampler = args.sampler

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_tag = (f"{args.loss}/{args.alpha_mode}" if args.loss == "focal"
                else args.loss)
    tag = f"sampler={effective_sampler}  loss={loss_tag}  seed={args.seed}"

    print(f"\n{'='*65}")
    print(f"  {tag}")
    print(f"  device={device}")
    print(f"{'='*65}")

    # ── 1. Load preprocessed data ─────────────────────────────────────────────
    X_tr, y_tr, X_val, y_val, X_te, y_te, le = load_data()
    print(f"\n[Data]  train={len(y_tr):,}  val={len(y_val):,}  test={len(y_te):,}")

    num_classes = len(le.classes_)
    ckpt_path   = os.path.join(CKPT_DIR,
                      f"{effective_sampler}_{loss_tag.replace('/', '_')}_seed{args.seed}.pt")
    bs          = cfg["batch_size"]

    # ── eval-only: load checkpoint and skip straight to test evaluation ───────
    if args.eval_only:
        print(f"\n[Eval-only]  loading {ckpt_path}")
        model = CNNBiLSTM(input_dim=78, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        test_loader    = make_loader(X_te, y_te, bs, shuffle=False)
        y_true, y_pred = evaluate(model, test_loader, device)
        metrics        = compute_metrics(y_true, y_pred, le.classes_)
        was_downsampled = False
    else:
        # ── 2. Prepare training set (resample or load pre-augmented) ──────────
        print(f"\n[Resample]  method={effective_sampler}")

        if args.data is not None:
            # Load explicitly specified augmented dataset (e.g. train_aug_tvae.npz).
            aug_path = os.path.join(PROC_DIR, f"{args.data}.npz")
            aug      = np.load(aug_path)
            X_tr     = aug["X"].astype(np.float32)
            y_tr     = aug["y"].astype(np.int64)
            was_downsampled = False
            print(f"  Loaded {aug_path}")
            print(f"  Augmented train size: {len(y_tr):,}")
        elif args.sampler == "ctgan":
            # Load train_aug.npz produced by augment_ctgan.py.
            aug_path = os.path.join(PROC_DIR, "train_aug.npz")
            aug      = np.load(aug_path)
            X_tr     = aug["X"].astype(np.float32)
            y_tr     = aug["y"].astype(np.int64)
            was_downsampled = False
            print(f"  Loaded {aug_path}")
            print(f"  Augmented train size: {len(y_tr):,}")
        else:
            X_tr, y_tr, was_downsampled = resample(
                X_tr, y_tr, args.sampler, args.seed,
                budget=args.budget, n_target=args.n_target, tau=args.tau)

        dist = sorted(Counter(y_tr).items())
        print(f"  Class dist after resample:")
        for cid, cnt in dist:
            print(f"    {le.classes_[cid]:<42} {cnt:>8,}")

        # ── 3. DataLoaders ────────────────────────────────────────────────────
        train_loader = make_loader(X_tr,  y_tr,  bs, shuffle=True)
        val_loader   = make_loader(X_val, y_val, bs, shuffle=False)
        test_loader  = make_loader(X_te,  y_te,  bs, shuffle=False)

        # ── 4. Model / loss / optimiser ───────────────────────────────────────
        print(f"\n[Model]  CNN-BiLSTM  num_classes={num_classes}")
        model = CNNBiLSTM(input_dim=78, num_classes=num_classes).to(device)
        count_params(model)

        if args.loss == "focal":
            cnt = Counter(y_tr.tolist())
            if args.alpha_mode == "inverse":
                # Original: alpha_k = (1/n_k) / sum(1/n_j)  — can cause precision collapse
                raw = np.array([1.0 / max(cnt.get(i, 1), 1)
                                for i in range(num_classes)], dtype=np.float32)
            elif args.alpha_mode == "sqrt_inverse":
                # Milder: alpha_k = (1/sqrt(n_k)) / sum(1/sqrt(n_j))
                raw = np.array([1.0 / np.sqrt(max(cnt.get(i, 1), 1))
                                for i in range(num_classes)], dtype=np.float32)
            else:  # "fixed"
                # Uniform alpha=0.25 (original Focal Loss paper default, class-agnostic)
                raw = np.full(num_classes, 0.25, dtype=np.float32)
            alpha_w   = torch.tensor(raw / raw.sum(), device=device)
            criterion = FocalLoss(gamma=cfg["gamma"], alpha=alpha_w)
            print(f"  FocalLoss: gamma={cfg['gamma']}  alpha_mode={args.alpha_mode}  "
                  f"alpha range=[{alpha_w.min():.5f}, {alpha_w.max():.5f}]")
        else:
            criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

        # ── 5. Training loop (early stopping on val Macro-F1) ─────────────────
        patience   = cfg["patience"]
        max_epochs = cfg["epochs"]
        print(f"\n[Train]  max_epochs={max_epochs}  patience={patience}")

        best_f1    = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(1, max_epochs + 1):
            t0       = time.time()
            tr_loss  = train_epoch(model, train_loader, optimizer, criterion, device)
            y_v, p_v = evaluate(model, val_loader, device)
            val_f1   = float(f1_score(y_v, p_v, average="macro", zero_division=0))
            elapsed  = time.time() - t0

            mark = " *" if val_f1 > best_f1 else ""
            print(f"  Epoch {epoch:3d}/{max_epochs}  "
                  f"loss={tr_loss:.4f}  val_macro_F1={val_f1:.4f}  "
                  f"({elapsed:.1f}s){mark}", flush=True)

            if val_f1 > best_f1:
                best_f1    = val_f1
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
                torch.save(best_state, ckpt_path)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"  >> Early stop at epoch {epoch}  "
                          f"(best val Macro-F1={best_f1:.4f})")
                    break

        # ── 6. Test-set evaluation ────────────────────────────────────────────
        model.load_state_dict(best_state)
        y_true, y_pred = evaluate(model, test_loader, device)
        metrics        = compute_metrics(y_true, y_pred, le.classes_)

    print(f"\n{'='*65}")
    print(f"  TEST RESULTS — {tag}")
    print(f"{'='*65}")
    print(f"  [Primary]")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Macro-Precision   : {metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall      : {metrics['macro_recall']:.4f}")
    print(f"  Macro-F1          : {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  [Auxiliary]")
    print(f"  G-mean (ε=1e-6)   : {metrics['gmean']:.4f}")
    print(f"  Weighted-F1       : {metrics['weighted_f1']:.4f}")
    print()
    print(f"  {'Class':<42} {'Recall':>7}  {'Precision':>9}")
    print(f"  {'-'*42} {'-'*7}  {'-'*9}")
    for name in le.classes_:
        safe = name.replace(" ", "_").replace("-", "_")
        print(f"  {name:<42} {metrics[f'recall_{safe}']:>7.4f}  "
              f"{metrics[f'precision_{safe}']:>9.4f}")

    # ── 7. Upsert results to CSV
    # Key: (sampler, loss, alpha_mode, seed) — alpha_mode only matters for focal loss.
    alpha_mode_val = args.alpha_mode if args.loss == "focal" else "N/A"
    row = {"sampler": effective_sampler, "loss": args.loss,
           "alpha_mode": alpha_mode_val,
           "seed": args.seed, "downsampled": was_downsampled,
           **metrics}

    results_csv = args.results_csv or RESULTS_CSV
    if not os.path.isabs(results_csv):
        # Interpret non-absolute paths relative to the repo root, not CWD.
        results_csv = os.path.join(ROOT_DIR, results_csv)
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)

    if os.path.exists(results_csv):
        df_all = pd.read_csv(results_csv)
        if "alpha_mode" not in df_all.columns:
            df_all["alpha_mode"] = "N/A"   # backfill for rows written before this column existed
        mask = ~((df_all["sampler"]    == effective_sampler) &
                 (df_all["loss"]       == args.loss)         &
                 (df_all["alpha_mode"] == alpha_mode_val)    &
                 (df_all["seed"]       == args.seed))
        df_all = pd.concat([df_all[mask], pd.DataFrame([row])], ignore_index=True)
    else:
        df_all = pd.DataFrame([row])
    df_all.to_csv(results_csv, index=False)
    print(f"\n  Results saved → {results_csv}")


if __name__ == "__main__":
    main()
