"""
train_ciciot.py — Adapted training script for CICIoT2023.
Based on train.py but uses ciciot_processed/ data and 39-dim features.
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

SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SRC_DIR)
PROC_DIR    = os.path.join(ROOT_DIR, "data", "ciciot_processed")
TABLES_DIR  = os.path.join(ROOT_DIR, "results", "tables")
CKPT_DIR    = os.path.join(ROOT_DIR, "checkpoints", "ciciot")
CONFIG_PATH = os.path.join(SRC_DIR, "config.yaml")
RESULTS_CSV = os.path.join(TABLES_DIR, "ciciot_results.csv")

sys.path.insert(0, SRC_DIR)
from model import CNNBiLSTM, count_params

os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# ── Focal Loss ──────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)
        ce    = F.nll_loss(log_p, targets, reduction="none")
        pt    = torch.exp(-ce)
        w     = (1.0 - pt) ** self.gamma
        if self.alpha is not None:
            w = w * self.alpha[targets]
        return (w * ce).mean()

# ── Data loading ────────────────────────────────────────────────────────────────
def load_data(seed: int):
    """Load the per-seed split produced by `preprocess_ciciot.py --seed N`.

    Each seed has its own independently re-drawn 6:2:2 stratified split and
    its own MinMaxScaler (fit on that seed's train fold only) — this is what
    makes seeds statistically independent replicates rather than the same
    split retrained with a different init (the 2026-07 split-seed coupling
    bug this replaces).
    """
    seed_dir = os.path.join(PROC_DIR, f"seed{seed}")
    if not os.path.isdir(seed_dir):
        raise FileNotFoundError(
            f"No split found for seed={seed} at {seed_dir}.\n"
            f"Run:  python preprocess_ciciot.py --seed {seed}   first.")
    tr  = np.load(os.path.join(seed_dir, "train.npz"))
    val = np.load(os.path.join(seed_dir, "val.npz"))
    te  = np.load(os.path.join(seed_dir, "test.npz"))
    with open(os.path.join(seed_dir, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return (tr["X"],  tr["y"].astype(np.int64),
            val["X"], val["y"].astype(np.int64),
            te["X"],  te["y"].astype(np.int64),
            le)

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)

# ── Resampling (budget-matched only, no majority cap) ───────────────────────────
CAP = 50_000

def resample(X, y, sampler_name, seed, n_target=5000, tau=2000):
    """Budget-matched resampling: only lift classes below tau to n_target."""
    if sampler_name == "none":
        counts = Counter(y)
        print(f"    [SELF-CHECK] sampler=none: {len(y):,} rows, {len(counts)} classes")
        assert X.shape[0] == len(y), "[FATAL] X/y length mismatch on the no-op path."
        return X, y, False

    counts_before = Counter(y)
    min_count = min(counts_before.values())
    k = min(5, min_count - 1)

    # Build matched-budget strategy
    strategy = {}
    for cid, n in counts_before.items():
        if n < tau:
            strategy[cid] = max(n, n_target)
    if not strategy:
        print(f"    [WARN] No class below tau={tau}; falling back to default")
        strategy = "auto"
    else:
        print(f"    Budget=matched, tau={tau}, N_target={n_target}")
        print(f"    Lifting {len(strategy)} classes")

    if sampler_name == "ros":
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    elif sampler_name == "smote":
        sampler = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    elif sampler_name == "adasyn":
        sampler = ADASYN(sampling_strategy=strategy, n_neighbors=k, random_state=seed)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    try:
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as exc:
        print(f"    [WARN] {sampler_name.upper()} failed ({exc}). Falling back to ROS.")
        X_res, y_res = RandomOverSampler(sampling_strategy=strategy, random_state=seed).fit_resample(X, y)

    # Runtime invariant (T3): classes outside the matched-budget strategy must
    # be byte-for-byte preserved — this is the check that caught the silent
    # majority-cap contamination on CIC-IDS2017 (train.py); ported here so the
    # same class of bug cannot recur silently on CICIoT2023.
    counts_after = Counter(y_res.tolist())
    untouched = set(counts_before) - set(strategy if isinstance(strategy, dict) else {})
    for cid in untouched:
        assert counts_after.get(cid, 0) == counts_before[cid], (
            f"[FATAL] class {cid} count changed from {counts_before[cid]:,} to "
            f"{counts_after.get(cid, 0):,} outside the matched-budget strategy "
            f"— {sampler_name.upper()} touched a class it should not have.")

    print(f"    After {sampler_name.upper()}: {len(y_res):,} samples")
    print(f"    [SELF-CHECK] {len(untouched)} untouched classes verified byte-for-byte preserved")
    return X_res, y_res.astype(np.int64), False

# ── Training / evaluation ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, n = 0.0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / n

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for Xb, yb in loader:
        preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(preds)

def compute_metrics(y_true, y_pred, class_names):
    m = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    for name, r, p in zip(class_names, per_recall, per_precision):
        safe = name.replace(" ", "_").replace("-", "_")
        m[f"recall_{safe}"] = round(float(r), 6)
        m[f"precision_{safe}"] = round(float(p), 6)
    return m

# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--sampler", default="none", choices=["none", "ros", "smote", "adasyn"])
    parser.add_argument("--loss", default="ce", choices=["ce", "focal"])
    parser.add_argument("--alpha_mode", default="sqrt_inverse",
                        choices=["inverse", "sqrt_inverse", "fixed"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--tau", type=int, default=2000)
    parser.add_argument("--n_target", type=int, default=5000)
    args = parser.parse_args()

    effective_sampler = args.tag or args.sampler

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_tag = f"{args.loss}/{args.alpha_mode}" if args.loss == "focal" else args.loss
    tag = f"sampler={effective_sampler}  loss={loss_tag}  seed={args.seed}"

    print(f"\n{'='*65}")
    print(f"  CICIoT2023 — {tag}")
    print(f"  device={device}")
    print(f"{'='*65}")

    X_tr, y_tr, X_val, y_val, X_te, y_te, le = load_data(args.seed)
    print(f"\n[Data]  train={len(y_tr):,}  val={len(y_val):,}  test={len(y_te):,}")

    num_classes = len(le.classes_)
    input_dim = X_tr.shape[1]
    ckpt_path = os.path.join(CKPT_DIR, f"{effective_sampler}_{loss_tag.replace('/', '_')}_seed{args.seed}.pt")
    bs = cfg["batch_size"]

    # Resample
    print(f"\n[Resample]  method={effective_sampler}")
    X_tr, y_tr, was_downsampled = resample(X_tr, y_tr, args.sampler, args.seed,
                                             n_target=args.n_target, tau=args.tau)

    dist = sorted(Counter(y_tr).items())
    print(f"  Class dist after resample:")
    for cid, cnt in dist[:5]:
        print(f"    {le.classes_[cid]:<42} {cnt:>8,}")
    if len(dist) > 5:
        print(f"    ... ({len(dist)} classes total)")

    train_loader = make_loader(X_tr, y_tr, bs, shuffle=True)
    val_loader = make_loader(X_val, y_val, bs, shuffle=False)
    test_loader = make_loader(X_te, y_te, bs, shuffle=False)

    # Model
    print(f"\n[Model]  CNN-BiLSTM  input_dim={input_dim}  num_classes={num_classes}")
    model = CNNBiLSTM(input_dim=input_dim, num_classes=num_classes).to(device)
    count_params(model)

    # Loss
    if args.loss == "focal":
        cnt = Counter(y_tr.tolist())
        if args.alpha_mode == "inverse":
            raw = np.array([1.0 / max(cnt.get(i, 1), 1) for i in range(num_classes)], dtype=np.float32)
        elif args.alpha_mode == "sqrt_inverse":
            raw = np.array([1.0 / np.sqrt(max(cnt.get(i, 1), 1)) for i in range(num_classes)], dtype=np.float32)
        else:
            raw = np.full(num_classes, 0.25, dtype=np.float32)
        alpha_w = torch.tensor(raw / raw.sum(), device=device)
        criterion = FocalLoss(gamma=cfg["gamma"], alpha=alpha_w)
        print(f"  FocalLoss: gamma={cfg['gamma']}  alpha_mode={args.alpha_mode}")
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    # Training
    patience = cfg["patience"]
    max_epochs = cfg["epochs"]
    print(f"\n[Train]  max_epochs={max_epochs}  patience={patience}")

    best_f1 = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        y_v, p_v = evaluate(model, val_loader, device)
        val_f1 = float(f1_score(y_v, p_v, average="macro", zero_division=0))
        elapsed = time.time() - t0

        mark = " *" if val_f1 > best_f1 else ""
        print(f"  Epoch {epoch:3d}/{max_epochs}  loss={tr_loss:.4f}  val_macro_F1={val_f1:.4f}  ({elapsed:.1f}s){mark}", flush=True)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            torch.save(best_state, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  >> Early stop at epoch {epoch}  (best val Macro-F1={best_f1:.4f})")
                break

    # Test evaluation
    model.load_state_dict(best_state)
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, le.classes_)

    print(f"\n{'='*65}")
    print(f"  TEST RESULTS — {tag}")
    print(f"{'='*65}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Macro-Precision   : {metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall      : {metrics['macro_recall']:.4f}")
    print(f"  Macro-F1          : {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Weighted-F1       : {metrics['weighted_f1']:.4f}")

    # Save results.
    # `sampler` stays a clean categorical key (matches main_results.csv's
    # convention) so groupby("sampler") gives real per-method groups; `tag`
    # carries the CLI --tag string (which previously overwrote `sampler` and
    # silently broke groupby into one-row groups — 2026-07 audit finding).
    alpha_mode_val = args.alpha_mode if args.loss == "focal" else "N/A"
    row = {"sampler": args.sampler, "tag": effective_sampler, "loss": args.loss,
           "alpha_mode": alpha_mode_val, "seed": args.seed,
           "downsampled": was_downsampled, **metrics}

    if os.path.exists(RESULTS_CSV):
        df_all = pd.read_csv(RESULTS_CSV)
        if "alpha_mode" not in df_all.columns:
            df_all["alpha_mode"] = "N/A"
        if "tag" not in df_all.columns:
            df_all["tag"] = df_all["sampler"]
        mask = ~((df_all["tag"] == effective_sampler) &
                 (df_all["loss"] == args.loss) &
                 (df_all["alpha_mode"] == alpha_mode_val) &
                 (df_all["seed"] == args.seed))
        df_all = pd.concat([df_all[mask], pd.DataFrame([row])], ignore_index=True)
    else:
        df_all = pd.DataFrame([row])
    df_all.to_csv(RESULTS_CSV, index=False)
    print(f"\n  Results saved -> {RESULTS_CSV}")

if __name__ == "__main__":
    main()