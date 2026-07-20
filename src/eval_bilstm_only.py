"""eval_bilstm_only.py — eval-only recovery for interrupted bilstm_only ablation.

Loads ablation_bilstm_only_seed42.pt, evaluates on test set, writes:
  results/tables/ABLATION_arch_perclass_bilstm_only.csv
  upserts bilstm_only row into results/tables/ABLATION_architecture.csv
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, f1_score, precision_score, recall_score,
)

sys.stdout.reconfigure(encoding="utf-8")
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(SRC_DIR)
PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")
CKPT_DIR  = os.path.join(ROOT_DIR, "checkpoints")
TABLE_DIR = os.path.join(ROOT_DIR, "results", "tables")

CKPT         = os.path.join(CKPT_DIR,  "ablation_bilstm_only_seed42.pt")
ABLATION_CSV = os.path.join(TABLE_DIR, "ABLATION_architecture.csv")
PERCLASS_CSV = os.path.join(TABLE_DIR, "ABLATION_arch_perclass_bilstm_only.csv")

sys.path.insert(0, SRC_DIR)
from model import BiLSTMOnly

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device={device}")

te = np.load(os.path.join(PROC_DIR, "test.npz"))
X_te = te["X"].astype(np.float32)
y_te = te["y"].astype(np.int64)
with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)
class_names = list(le.classes_)
num_classes = len(class_names)

model = BiLSTMOnly(input_dim=78, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()
print(f"Loaded {CKPT}")

loader = DataLoader(
    TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
    batch_size=256, shuffle=False, num_workers=0)

preds, labels = [], []
with torch.no_grad():
    for Xb, yb in loader:
        preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
        labels.append(yb.numpy())
y_true = np.concatenate(labels)
y_pred = np.concatenate(preds)

metrics = {
    "accuracy"          : float(accuracy_score(y_true, y_pred)),
    "macro_precision"   : float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
    "macro_recall"      : float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
    "macro_f1"          : float(f1_score(y_true, y_pred, average="macro",        zero_division=0)),
    "balanced_accuracy" : float(balanced_accuracy_score(y_true, y_pred)),
}

print(f"\n[Test Results — bilstm_only]")
for k, v in metrics.items():
    print(f"  {k:<22}: {v:.6f}")

# per-class CSV
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
pd.DataFrame(rows).to_csv(PERCLASS_CSV, index=False, encoding="utf-8-sig")
print(f"\nSaved per-class → {PERCLASS_CSV}")

# upsert architecture CSV
row = {"arch": "bilstm_only", "seed": 42, **metrics}
if os.path.exists(ABLATION_CSV):
    df = pd.read_csv(ABLATION_CSV)
    mask = ~((df["arch"] == "bilstm_only") & (df["seed"] == 42))
    df   = pd.concat([df[mask], pd.DataFrame([row])], ignore_index=True)
else:
    df = pd.DataFrame([row])
df.to_csv(ABLATION_CSV, index=False)
print(f"Upserted → {ABLATION_CSV}")
print(f"\n[ABLATION_architecture.csv — complete]")
print(df.to_string(index=False))
