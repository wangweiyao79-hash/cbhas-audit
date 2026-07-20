"""
train_ablation.py — Architecture ablation experiment.

Three variants trained on identical TVAE-augmented data (train_aug_tvae.npz),
CE loss, seed=42, same hyperparams as the main experiment:

  cnn_bilstm  — eval-only from checkpoints/tvae_ce_seed42.pt
  cnn_only    — Conv×2 + AdaptiveAvgPool, no BiLSTM
  bilstm_only — BiLSTM(input=1, 78-step), no CNN

Outputs
-------
  results/tables/ABLATION_architecture.csv
  results/tables/ABLATION_arch_perclass_{arch}.csv  (3 files)
  checkpoints/ablation_cnn_only_seed42.pt
  checkpoints/ablation_bilstm_only_seed42.pt
"""
import os
import sys
import copy
import time
import pickle

import numpy as np
import pandas as pd
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, f1_score,
    precision_score, recall_score,
)

sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(SRC_DIR)
PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")
CKPT_DIR  = os.path.join(ROOT_DIR, "checkpoints")
TABLE_DIR = os.path.join(ROOT_DIR, "results", "tables")
CFG_PATH  = os.path.join(SRC_DIR, "config.yaml")

FULL_CKPT    = os.path.join(CKPT_DIR, "tvae_ce_seed42.pt")
ABLATION_CSV = os.path.join(TABLE_DIR, "ABLATION_architecture.csv")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,  exist_ok=True)

sys.path.insert(0, SRC_DIR)
from model import CNNBiLSTM, CNNOnly, BiLSTMOnly, count_params


# ── helpers ──────────────────────────────────────────────────────────────────

def make_loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True)


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


def compute_metrics(y_true, y_pred):
    return {
        "accuracy"          : float(accuracy_score(y_true, y_pred)),
        "macro_precision"   : float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall"      : float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
        "macro_f1"          : float(f1_score(y_true, y_pred, average="macro",        zero_division=0)),
        "balanced_accuracy" : float(balanced_accuracy_score(y_true, y_pred)),
    }


def save_perclass_csv(y_true, y_pred, class_names, arch):
    rpt = classification_report(y_true, y_pred, target_names=class_names,
                                output_dict=True, zero_division=0)
    rows = []
    for name in class_names:
        d = rpt[name]
        rows.append({"class": name,
                     "precision": round(d["precision"], 6),
                     "recall":    round(d["recall"],    6),
                     "f1_score":  round(d["f1-score"],  6),
                     "support":   int(d["support"])})
    for avg in ("macro avg", "weighted avg"):
        d = rpt[avg]
        rows.append({"class": avg,
                     "precision": round(d["precision"], 6),
                     "recall":    round(d["recall"],    6),
                     "f1_score":  round(d["f1-score"],  6),
                     "support":   int(d["support"])})
    path = os.path.join(TABLE_DIR, f"ABLATION_arch_perclass_{arch}.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  Saved per-class → {path}")


def upsert_ablation_csv(arch, seed, metrics):
    row = {"arch": arch, "seed": seed, **metrics}
    if os.path.exists(ABLATION_CSV):
        df = pd.read_csv(ABLATION_CSV)
        mask = ~((df["arch"] == arch) & (df["seed"] == seed))
        df   = pd.concat([df[mask], pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(ABLATION_CSV, index=False)
    print(f"  Upserted ABLATION_architecture.csv  ({arch}, seed={seed})")


# ── per-architecture runner ───────────────────────────────────────────────────

def run_arch(arch, cfg, device, X_tr, y_tr, X_val, y_val, X_te, y_te,
             class_names, seed):
    num_classes = len(class_names)
    bs          = cfg["batch_size"]

    print(f"\n{'='*65}")
    print(f"  ARCH: {arch}   seed={seed}   device={device}")
    print(f"{'='*65}")

    # build model
    if arch == "cnn_bilstm":
        model = CNNBiLSTM(input_dim=78, num_classes=num_classes).to(device)
    elif arch == "cnn_only":
        model = CNNOnly(input_dim=78, num_classes=num_classes).to(device)
    else:
        model = BiLSTMOnly(input_dim=78, num_classes=num_classes).to(device)
    count_params(model)

    test_loader = make_loader(X_te, y_te, bs, shuffle=False)

    if arch == "cnn_bilstm":
        # eval-only: load best checkpoint from main experiment
        print(f"\n[Eval-only]  {FULL_CKPT}")
        model.load_state_dict(torch.load(FULL_CKPT, map_location=device))
    else:
        # train from scratch with same hyperparams
        ckpt_path    = os.path.join(CKPT_DIR, f"ablation_{arch}_seed{seed}.pt")
        train_loader = make_loader(X_tr, y_tr, bs, shuffle=True)
        val_loader   = make_loader(X_val, y_val, bs, shuffle=False)
        criterion    = nn.CrossEntropyLoss()
        optimizer    = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        patience     = cfg["patience"]
        max_epochs   = cfg["epochs"]

        print(f"\n[Train]  max_epochs={max_epochs}  patience={patience}  "
              f"train_size={len(y_tr):,}")

        best_f1, best_state, no_improve = -1.0, None, 0
        for epoch in range(1, max_epochs + 1):
            t0      = time.time()
            tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            y_v, p_v = evaluate(model, val_loader, device)
            val_f1  = float(f1_score(y_v, p_v, average="macro", zero_division=0))
            elapsed = time.time() - t0
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
                    print(f"  >> Early stop  (best val Macro-F1={best_f1:.4f})")
                    break
        model.load_state_dict(best_state)
        print(f"  Checkpoint saved → {ckpt_path}")

    # test evaluation
    y_true, y_pred = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred)

    print(f"\n[Test Results]")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Macro-Precision   : {metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall      : {metrics['macro_recall']:.4f}")
    print(f"  Macro-F1          : {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")

    # write outputs
    save_perclass_csv(y_true, y_pred, class_names, arch)
    upsert_ablation_csv(arch, seed, metrics)

    return metrics


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    seed = cfg.get("seed", 42)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    print("\n[Load data]")
    aug = np.load(os.path.join(PROC_DIR, "train_aug_tvae.npz"))
    X_tr = aug["X"].astype(np.float32)
    y_tr = aug["y"].astype(np.int64)

    val = np.load(os.path.join(PROC_DIR, "val.npz"))
    X_val = val["X"].astype(np.float32)
    y_val = val["y"].astype(np.int64)

    te = np.load(os.path.join(PROC_DIR, "test.npz"))
    X_te = te["X"].astype(np.float32)
    y_te = te["y"].astype(np.int64)

    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    class_names = list(le.classes_)

    print(f"  train={len(y_tr):,}  val={len(y_val):,}  test={len(y_te):,}  "
          f"classes={len(class_names)}")

    # run ablations in order: full model first (eval-only), then two variants
    archs   = ["cnn_bilstm", "cnn_only", "bilstm_only"]
    results = {}
    for arch in archs:
        results[arch] = run_arch(
            arch, cfg, device,
            X_tr, y_tr, X_val, y_val, X_te, y_te,
            class_names, seed,
        )

    # final comparison summary
    full_f1  = results["cnn_bilstm"]["macro_f1"]
    cnn_f1   = results["cnn_only"]["macro_f1"]
    lstm_f1  = results["bilstm_only"]["macro_f1"]
    delta_cnn  = full_f1 - cnn_f1
    delta_lstm = full_f1 - lstm_f1

    print(f"\n{'='*65}")
    print(f"  ARCHITECTURE ABLATION SUMMARY (seed={seed})")
    print(f"{'='*65}")
    print(f"  {'Variant':<18} {'Acc':>7} {'MacroP':>8} {'MacroR':>8} "
          f"{'MacroF1':>9} {'BalAcc':>8}")
    print(f"  {'-'*18} {'-'*7} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")
    for arch in archs:
        m = results[arch]
        print(f"  {arch:<18} {m['accuracy']:>7.4f} {m['macro_precision']:>8.4f} "
              f"{m['macro_recall']:>8.4f} {m['macro_f1']:>9.4f} "
              f"{m['balanced_accuracy']:>8.4f}")
    print()
    print(f"  [Paper sentence]")
    print(f"  Removing BiLSTM (CNN-only) reduces Macro-F1 by "
          f"{delta_cnn:.4f} ({delta_cnn*100:.2f} pp) relative to the full model "
          f"({full_f1:.4f} → {cnn_f1:.4f}).")
    print(f"  Removing CNN (BiLSTM-only) reduces Macro-F1 by "
          f"{delta_lstm:.4f} ({delta_lstm*100:.2f} pp) relative to the full model "
          f"({full_f1:.4f} → {lstm_f1:.4f}).")
    print(f"\n  All outputs in results/tables/ABLATION_*.csv")


if __name__ == "__main__":
    main()
