"""One-shot: evaluate tvae_ce_seed42.pt on test set, export per-class CSV."""
import sys, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
from model import CNNBiLSTM

PROC = "data/processed"
CKPT = "checkpoints/tvae_ce_seed42.pt"
OUT  = "results/tables/FINAL_tvae_ce_perclass.csv"

te = np.load(f"{PROC}/test.npz")
X_te = te["X"].astype(np.float32)
y_te = te["y"].astype(np.int64)

with open(f"{PROC}/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)
names = list(le.classes_)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CNNBiLSTM(input_dim=78, num_classes=len(names)).to(device)
model.load_state_dict(torch.load(CKPT, map_location=device))
model.eval()

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

rpt = classification_report(y_true, y_pred, target_names=names,
                             output_dict=True, zero_division=0)
rows = []
for name in names:
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

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False, encoding="utf-8-sig")
print(df.to_string(index=False))
print(f"\nSaved -> {OUT}")
