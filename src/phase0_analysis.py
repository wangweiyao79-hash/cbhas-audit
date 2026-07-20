"""
phase0_analysis.py — no-training diagnostics for reviewer comments M2, S2, S4, S5.

Emits four artifacts to results/tables/:
  1. augmentation_budget.csv     — per-class train/val/test counts, augmentation route,
                                   n_added, synthetic-to-real ratio, post-aug fraction
                                   (addresses S4 threshold semantics + S5 aug ratios)
  2. abstract_numbers.md         — recomputed abstract claims vs. published values
                                   (addresses M2 fact-check: "20.53 vs oversampling")
  3. rare_class_wilson.csv       — Wilson 95% CI for per-class recall & precision,
                                   with TP/FP/FN counts (addresses S2)
  4. macro_delta_source.md       — where every "improvement" number in the abstract
                                   comes from, so reviewers can retrace

Reads existing artifacts only; no training.
"""
import os
import sys
import pickle
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(SRC_DIR)
PROC_DIR  = os.path.join(ROOT_DIR, "data", "processed")
TABLE_DIR = os.path.join(ROOT_DIR, "results", "tables")
CKPT_DIR  = os.path.join(ROOT_DIR, "checkpoints")

sys.path.insert(0, SRC_DIR)
from model import CNNBiLSTM


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Two-sided Wilson score CI at 95% for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p     = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw  = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


# ══════════════════════════════════════════════════════════════════════════════
# 1. Augmentation budget table  (M5, S4, S5)
# ══════════════════════════════════════════════════════════════════════════════
def augmentation_budget():
    print("\n[1/4] Augmentation budget table")
    tr  = np.load(os.path.join(PROC_DIR, "train.npz"))["y"]
    val = np.load(os.path.join(PROC_DIR, "val.npz"))["y"]
    te  = np.load(os.path.join(PROC_DIR, "test.npz"))["y"]
    aug = np.load(os.path.join(PROC_DIR, "train_aug_tvae.npz"))["y"]

    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    ct_tr, ct_val, ct_te, ct_aug = map(Counter, [tr, val, te, aug])

    # Read the aug_report to attribute each rare class to its route
    route_map = {}
    rep = pd.read_csv(os.path.join(TABLE_DIR, "aug_report_tvae.csv"))
    for _, r in rep.iterrows():
        route_map[int(r["class_id"])] = r["method"]

    rows = []
    for cid, cname in enumerate(le.classes_):
        n_tr    = ct_tr.get(cid, 0)
        n_val   = ct_val.get(cid, 0)
        n_te    = ct_te.get(cid, 0)
        n_aug   = ct_aug.get(cid, 0)
        n_added = n_aug - n_tr
        rows.append({
            "class_id"          : cid,
            "class_name"        : cname,
            "n_train"           : n_tr,
            "n_val"             : n_val,
            "n_test"            : n_te,
            "route"             : route_map.get(cid, "none"),
            "n_added"           : n_added,
            "n_after_aug"       : n_aug,
            "syn_to_real_ratio" : round(n_added / max(n_tr, 1), 2),
            "train_frac_before" : round(100 * n_tr / len(tr), 4),
            "train_frac_after"  : round(100 * n_aug / len(aug), 4),
        })
    df = pd.DataFrame(rows).sort_values("n_train", ascending=False)
    out = os.path.join(TABLE_DIR, "augmentation_budget.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")

    # Console summary of the imbalance ratios
    n_maj = ct_tr[df.iloc[0].class_id]
    print(f"\n  Imbalance ratio  (majority : rarest class)")
    print(f"    Before aug: {n_maj:,} : {ct_tr[df.iloc[-1].class_id]}"
          f"  =  {n_maj / max(ct_tr[df.iloc[-1].class_id], 1):.0f} : 1")
    print(f"    After  aug: {ct_aug[df.iloc[0].class_id]:,} : {ct_aug[df.iloc[-1].class_id]:,}"
          f"  =  {ct_aug[df.iloc[0].class_id] / ct_aug[df.iloc[-1].class_id]:.1f} : 1")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Abstract-numbers fact check  (M2)
# ══════════════════════════════════════════════════════════════════════════════
def abstract_numbers():
    print("\n[2/4] Abstract-numbers fact check (M2)")
    df = pd.read_csv(os.path.join(TABLE_DIR, "main_results.csv"))
    df["alpha_mode"] = df["alpha_mode"].fillna("N/A")

    def pick(sampler, loss, alpha="N/A", seed=42):
        rows = df[(df.sampler == sampler) & (df.loss == loss) &
                  (df.alpha_mode == alpha) & (df.seed == seed)]
        if len(rows) == 0:
            return None
        return float(rows.iloc[0].macro_f1) * 100, float(rows.iloc[0].accuracy) * 100

    none_ce = pick("none", "ce")
    ros_ce  = pick("ros",  "ce")
    smote_ce = pick("smote", "ce")
    adasyn_ce = pick("adasyn", "ce")
    ctgan_ce = pick("ctgan", "ce")
    tvae_ce  = pick("tvae", "ce")

    if tvae_ce is None:
        print("  [ERR] tvae/ce/seed=42 row missing from main_results.csv")
        return

    tvae_f1 = tvae_ce[0]
    lines = [
        "# Abstract number reconciliation (M2)",
        "",
        f"CB-HAS+CE Macro-F1: **{tvae_f1:.2f}**  (Table 4 value)",
        "",
        "| Claim                                              | Baseline F1 | Δ (pp) |",
        "|----------------------------------------------------|-------------|--------|",
    ]
    def row(label, base):
        if base is None:
            return f"| {label:<52s} | (missing)   |        |"
        return f"| {label:<52s} | {base[0]:>10.2f}  | {tvae_f1 - base[0]:+7.2f} |"

    lines.extend([
        row("vs none+CE (paper claims +13.77)", none_ce),
        row("vs ROS+CE (best oversampling by F1?)", ros_ce),
        row("vs SMOTE+CE",  smote_ce),
        row("vs ADASYN+CE (paper text implies 'best oversampling')", adasyn_ce),
        row("vs CTGAN+CE (strongest data-level baseline)", ctgan_ce),
        "",
        "**Interpretation for M2**:",
        "- ROS, SMOTE, ADASYN are the three 'traditional oversampling' methods.",
        "- The maximum among their Macro-F1 values is the true 'best oversampling'.",
        "- The abstract's `20.53 pp over best oversampling` matches ADASYN (63.38%), not the maximum.",
        "- Reviewers M2 asks for one of two fixes:",
        "  (a) Rewrite abstract to say 'best interpolation-based method' and cite ADASYN, or",
        "  (b) Rewrite abstract to say 'strongest data-level baseline (CTGAN)' and cite +4.12 pp.",
        "",
        f"- Accuracy sanity check: 99.76 - 98.43 = **{tvae_ce[1] - none_ce[1]:.2f} pp**"
        f"  (paper text says 1.32; correct value is above).",
    ])

    out = os.path.join(TABLE_DIR, "abstract_numbers.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {out}")
    print("  ─── Preview ───")
    for ln in lines[:14]:
        print(f"  {ln}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Rare-class Wilson intervals + TP/FP/FN  (S2)
# ══════════════════════════════════════════════════════════════════════════════
def rare_class_wilson():
    print("\n[3/4] Rare-class Wilson CI + TP/FP/FN table (S2)")
    ckpt_path = os.path.join(CKPT_DIR, "tvae_ce_seed42.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] Missing checkpoint: {ckpt_path}")
        return

    te = np.load(os.path.join(PROC_DIR, "test.npz"))
    X_te = te["X"].astype(np.float32)
    y_te = te["y"].astype(np.int64)
    with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNBiLSTM(input_dim=78, num_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ds = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)

    cm = confusion_matrix(y_te, y_pred, labels=range(len(le.classes_)))

    rows = []
    for cid, cname in enumerate(le.classes_):
        support = int(cm[cid, :].sum())
        tp = int(cm[cid, cid])
        fn = support - tp
        # FP = predicted-as-this-class but true class was different
        fp = int(cm[:, cid].sum() - tp)
        recall    = tp / support if support > 0 else float("nan")
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        r_lo, r_hi = wilson_ci(tp, support)
        p_lo, p_hi = wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (float("nan"), float("nan"))
        rows.append({
            "class"           : cname,
            "test_support"    : support,
            "TP"              : tp,
            "FP"              : fp,
            "FN"              : fn,
            "recall"          : round(recall, 4),
            "recall_CI95_lo"  : round(r_lo, 4),
            "recall_CI95_hi"  : round(r_hi, 4),
            "precision"       : round(precision, 4),
            "prec_CI95_lo"    : round(p_lo, 4),
            "prec_CI95_hi"    : round(p_hi, 4),
        })
    df = pd.DataFrame(rows).sort_values("test_support")
    out = os.path.join(TABLE_DIR, "rare_class_wilson.csv")
    df.to_csv(out, index=False)
    print(f"  Saved → {out}")

    # Print the rows the reviewers care most about
    print(f"\n  Rare-class rows (test_support < 100):")
    hdr = f"    {'class':<40} {'sup':>4}  {'TP':>3}  {'FP':>3}  {'recall (95% CI)':>22}"
    print(hdr)
    print(f"    {'-'*40} {'-'*4}  {'-'*3}  {'-'*3}  {'-'*22}")
    for _, r in df[df.test_support < 100].iterrows():
        ci = f"[{r.recall_CI95_lo:.2f}, {r.recall_CI95_hi:.2f}]"
        print(f"    {r['class']:<40} {r.test_support:>4}  {int(r.TP):>3}  "
              f"{int(r.FP):>3}  {r.recall:.3f} {ci:>15}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. Delta source table  (M2 traceability)
# ══════════════════════════════════════════════════════════════════════════════
def macro_delta_source():
    print("\n[4/4] Where each abstract number comes from")
    lines = [
        "# Traceability of numeric claims in the abstract",
        "",
        "Every number the abstract cites is reproduced from these CSV rows.",
        "Reviewers can grep `main_results.csv` with the (sampler, loss, alpha_mode, seed)",
        "keys below to verify each figure.",
        "",
        "| Abstract claim                          | Source row key                              |",
        "|-----------------------------------------|---------------------------------------------|",
        "| Macro-F1 83.91 (proposed)               | sampler=tvae, loss=ce, alpha_mode=N/A, seed=42 |",
        "| Accuracy 99.76                          | same row, `accuracy` field                  |",
        "| Macro-Precision 90.22                   | same row, `macro_precision` field           |",
        "| +13.77 pp vs no aug                     | none/ce/N/A vs tvae/ce/N/A                  |",
        "| +20.53 pp vs 'best oversampling'        | **ADASYN, not ROS** — see abstract_numbers.md |",
        "| Focal Loss −6.26 to −14.50 pp           | tvae/focal/fixed and tvae/focal/sqrt_inverse |",
    ]
    out = os.path.join(TABLE_DIR, "macro_delta_source.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {out}")


if __name__ == "__main__":
    print("=" * 65)
    print("Phase 0 diagnostics (no training)")
    print("=" * 65)
    augmentation_budget()
    abstract_numbers()
    rare_class_wilson()
    macro_delta_source()
    print("\nDone.")
