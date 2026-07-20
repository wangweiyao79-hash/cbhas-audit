"""
plot_figures.py  —  Generate 4 paper figures for CIC-IDS2017 NIDS paper.

Outputs (PDF + PNG 300 dpi) in results/figures/:
  fig_confusion_matrix.*
  fig_config_comparison.*
  fig_perclass_f1.*
  fig_ablation.*
"""
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

sys.stdout.reconfigure(encoding="utf-8")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC  = os.path.join(ROOT, "data", "processed")
CKPT  = os.path.join(ROOT, "checkpoints", "tvae_ce_seed42.pt")
TABLE = os.path.join(ROOT, "results", "tables")
FIG   = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "src"))
from model import CNNBiLSTM

# ── global matplotlib style ───────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"     : "Times New Roman",
    "font.size"       : 11,
    "axes.labelsize"  : 11,
    "xtick.labelsize" : 10,
    "ytick.labelsize" : 10,
    "legend.fontsize" : 10,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "pdf.fonttype"    : 42,   # TrueType embed (avoids Type-3 fonts)
    "ps.fonttype"     : 42,
})

# Wong (2011) colour-blind-safe palette
CB = dict(
    blue   = "#0072B2",
    orange = "#E69F00",
    green  = "#009E73",
    red    = "#D55E00",
    purple = "#CC79A7",
    yellow = "#F0E442",
    lblue  = "#56B4E9",
    black  = "#000000",
    gray   = "#999999",
)

# 15 abbreviated class labels in label_encoder order (alphabetical)
SHORT = [
    "BENIGN", "Bot", "DDoS", "DoS-GE", "DoS-Hulk",
    "DoS-SHT", "DoS-SL", "FTP-Pat.", "Heartbleed", "Infilt.",
    "PortScan", "SSH-Pat.", "WA-BF", "WA-SQLi", "WA-XSS",
]


def save_fig(fig, name):
    for ext, kw in [("pdf", {}), ("png", {"dpi": 300})]:
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)
    print(f"  → {name}.pdf  +  {name}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Shared: load CB-HAS+CE model and run test-set inference
# ══════════════════════════════════════════════════════════════════════════════
def run_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(PROC, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    data = np.load(os.path.join(PROC, "test.npz"))
    X_te = data["X"].astype(np.float32)
    y_te = data["y"].astype(np.int64)

    model = CNNBiLSTM(input_dim=78, num_classes=len(le.classes_)).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=512, shuffle=False, num_workers=0)

    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
            labels.append(yb.numpy())
    return np.concatenate(labels), np.concatenate(preds), le.classes_


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Confusion Matrix (CB-HAS+CE, row-normalised)
# ══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(y_true, y_pred):
    print("\n[Fig 1] Confusion matrix ...")
    nc = len(SHORT)
    cm = confusion_matrix(y_true, y_pred)
    # Row-normalise: proportion of true class predicted as each class
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_n = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.03)
    cbar.set_label("Proportion of true class", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(range(nc))
    ax.set_xticklabels(SHORT, rotation=50, ha="right", fontsize=9)
    ax.set_yticks(range(nc))
    ax.set_yticklabels(SHORT, fontsize=9)
    ax.set_xlabel("Predicted label", labelpad=8)
    ax.set_ylabel("True label", labelpad=8)

    # Annotate: always show diagonal; off-diagonal only if ≥ 0.04
    for i in range(nc):
        for j in range(nc):
            v = cm_n[i, j]
            if i == j or v >= 0.04:
                txt_color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7.5, color=txt_color, fontweight="bold" if i == j else "normal")

    fig.tight_layout()
    save_fig(fig, "fig_confusion_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — 9-config Macro-F1 & Balanced Accuracy grouped bar chart
# ══════════════════════════════════════════════════════════════════════════════
def plot_config_comparison():
    print("\n[Fig 2] Config comparison ...")
    df = pd.read_csv(os.path.join(TABLE, "FINAL_results.csv"))

    # ── Decontaminated matched-budget swap-in (2026-07-11) ──────────────────
    # The original ROS/SMOTE/ADASYN rows in FINAL_results.csv used a default
    # budget whose provenance could not be verified (see
    # main_results_default_retracted.csv). Table 4's traditional-oversampling
    # baselines are unified to the matched-budget convention; substitute the
    # verified decontaminated seed=42 values here (source of truth:
    # phase3_results_v2.csv). FINAL_results.csv itself is left untouched.
    _matched_f1 = {"ros": 0.7934404928757388,
                   "smote": 0.7966925312792353,
                   "adasyn": 0.7949014397361988}
    for _s, _f1 in _matched_f1.items():
        df.loc[(df.sampler == _s) & (df.loss == "ce"), "macro_f1"] = _f1

    def make_label(row):
        s = str(row["sampler"])
        l = str(row["loss"])
        a = "" if pd.isna(row.get("alpha_mode", np.nan)) else str(row.get("alpha_mode", ""))
        if l == "ce":
            if s == "none": return "No Aug.+CE"
            if s == "tvae": return "CB-HAS+CE"
            if s in ("ros", "smote", "adasyn"): return f"{s.upper()}+CE (matched)"
            return f"{s.upper()}+CE"
        # focal
        if a == "sqrt_inverse": return f"{s.upper()}+Focal(√inv)"
        if a == "fixed":        return f"{s.upper()}+Focal(α=.25)"
        return f"{s.upper()}+Focal"

    df["label"] = df.apply(make_label, axis=1)
    df = df.drop_duplicates(subset="label", keep="first")

    ORDER = [
        "No Aug.+CE", "ROS+CE (matched)", "SMOTE+CE (matched)", "ADASYN+CE (matched)",
        "CTGAN+CE",   "CTGAN+Focal",
        "CB-HAS+CE",    "TVAE+Focal(√inv)", "TVAE+Focal(α=.25)",
    ]
    df = df.set_index("label").reindex(ORDER).reset_index()

    # ── print Focal vs CB-HAS+CE for user verification ──────────────────────
    tvae_ce_f1 = df.loc[df["label"] == "CB-HAS+CE", "macro_f1"].values[0]
    for lbl in ["TVAE+Focal(√inv)", "TVAE+Focal(α=.25)"]:
        v = df.loc[df["label"] == lbl, "macro_f1"].values[0]
        below = "✓ below CB-HAS+CE" if v < tvae_ce_f1 else "✗ NOT below CB-HAS+CE"
        print(f"  {lbl:<26}  macro_f1={v:.6f}  ({below})")
    print(f"  {'CB-HAS+CE':<26}  macro_f1={tvae_ce_f1:.6f}  (reference)")

    n  = len(df)
    x  = np.arange(n)
    w  = 0.55
    hi = list(df["label"]).index("CB-HAS+CE")   # index of best config

    # CB-HAS+CE bar highlighted in orange; all others in steel blue
    f1_col = [CB["orange"] if i == hi else CB["blue"] for i in range(n)]
    edge   = ["black"] * n

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.bar(x, df["macro_f1"], width=w, color=f1_col,
           edgecolor=edge, linewidth=0.6, zorder=3)

    # Dashed reference at CB-HAS+CE Macro-F1
    ax.axhline(tvae_ce_f1, color="black", linestyle="--", linewidth=0.8,
               alpha=0.55, zorder=2)

    # Best annotation on CB-HAS+CE bar
    ax.text(hi, df["macro_f1"].iloc[hi] + 0.017, "[Best]",
            ha="center", va="bottom", fontsize=9.5, color="black", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=28, ha="right", fontsize=10)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=0)

    legend_patches = [
        Patch(facecolor=CB["blue"],   edgecolor="black", label="Macro-F1"),
        Patch(facecolor=CB["orange"], edgecolor="black", label="Macro-F1 (CB-HAS+CE [Best])"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9.5,
              framealpha=0.92, edgecolor="gray")
    fig.tight_layout()
    save_fig(fig, "fig_config_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Per-class F1: no-augmentation baseline vs CB-HAS+CE
# ══════════════════════════════════════════════════════════════════════════════
def plot_perclass_f1():
    print("\n[Fig 3] Per-class F1 comparison ...")

    # ── CB-HAS+CE F1 (from pre-computed perclass CSV) ───────────────────────
    df_tv = pd.read_csv(os.path.join(TABLE, "FINAL_tvae_ce_perclass.csv"))
    df_tv = df_tv[~df_tv["class"].isin(["macro avg", "weighted avg"])].reset_index(drop=True)
    tvae_f1 = df_tv["f1_score"].values.copy()   # shape (15,)

    # ── none+CE baseline F1: extract P and R by column position ──────────
    df_main = pd.read_csv(os.path.join(TABLE, "FINAL_results.csv"))
    none_row = df_main[(df_main["sampler"] == "none") & (df_main["loss"] == "ce")].iloc[0]

    # All columns starting with "recall_"  = 15 per-class recall values (in class order)
    # All columns starting with "precision_" (excluding "macro_precision") = 15 per-class
    rec_cols  = [c for c in df_main.columns if c.startswith("recall_")]
    prec_cols = [c for c in df_main.columns if c.startswith("precision_")
                 and not c.startswith("macro")]

    assert len(rec_cols)  == 15, f"Expected 15 recall cols, got {len(rec_cols)}"
    assert len(prec_cols) == 15, f"Expected 15 prec cols,   got {len(prec_cols)}"

    P = none_row[prec_cols].values.astype(float)
    R = none_row[rec_cols].values.astype(float)
    denom = P + R
    with np.errstate(invalid="ignore", divide="ignore"):
        base_f1 = np.where(denom > 0, 2 * P * R / denom, 0.0)

    # ── sort by baseline F1 ascending ─────────────────────────────────────
    idx    = np.argsort(base_f1)
    b_sort = base_f1[idx]
    t_sort = tvae_f1[idx]
    s_sort = [SHORT[i] for i in idx]

    # ── per-bar colour classification ──────────────────────────────────────
    # baseline bar:  gray if F1=0, else blue
    # CB-HAS+CE bar:   orange if baseline was 0 (0→nonzero ↑), else green
    # dagger "†":    mark classes where CB-HAS+CE F1 < 0.25 (persists difficult)
    col_base, col_tvae, dagger = [], [], []
    for b, t in zip(b_sort, t_sort):
        zero_base = (b < 1e-9)
        col_base.append(CB["gray"]   if zero_base else CB["blue"])
        col_tvae.append(CB["orange"] if zero_base else CB["green"])
        dagger.append(t < 0.25 and t > 0)   # non-zero but still low after augmentation

    n = len(s_sort)
    x = np.arange(n)
    w = 0.37

    fig, ax = plt.subplots(figsize=(13, 5))
    bars_b = ax.bar(x - w/2, b_sort, width=w, color=col_base,
                    edgecolor="black", linewidth=0.55, zorder=3)
    bars_t = ax.bar(x + w/2, t_sort, width=w, color=col_tvae,
                    edgecolor="black", linewidth=0.55, zorder=3)

    # Dagger annotation above TVAE bar for persists-difficult classes
    for i, (use_dag, tv) in enumerate(zip(dagger, t_sort)):
        if use_dag:
            ax.text(x[i] + w/2, tv + 0.018, "†", ha="center", va="bottom",
                    fontsize=11, color=CB["red"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(s_sort, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=0)

    legend_patches = [
        Patch(facecolor=CB["blue"],   edgecolor="black", label="No Aug.+CE (baseline)"),
        Patch(facecolor=CB["gray"],   edgecolor="black", label="No Aug.+CE  (F1 = 0 in baseline)"),
        Patch(facecolor=CB["green"],  edgecolor="black", label="CB-HAS+CE"),
        Patch(facecolor=CB["orange"], edgecolor="black", label="CB-HAS+CE  (0 → non-zero ↑)"),
        Patch(facecolor="white", edgecolor="white",
              label="† persists difficult: CB-HAS+CE F1 < 0.25"),
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9.5,
              framealpha=0.93, edgecolor="gray")
    fig.tight_layout()
    save_fig(fig, "fig_perclass_f1")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Architecture ablation Macro-F1
# ══════════════════════════════════════════════════════════════════════════════
def plot_ablation():
    print("\n[Fig 4] Architecture ablation ...")
    df = pd.read_csv(os.path.join(TABLE, "ABLATION_architecture.csv"))
    label_map = {
        "cnn_bilstm" : "CNN-BiLSTM\n(Full model)",
        "cnn_only"   : "CNN-only\n(no BiLSTM)",
        "bilstm_only": "BiLSTM-only\n(no CNN)",
    }
    order_arch = ["cnn_bilstm", "cnn_only", "bilstm_only"]
    df = df.set_index("arch").reindex(order_arch).reset_index()
    f1 = df["macro_f1"].values
    labels = [label_map[a] for a in df["arch"]]

    colors = [CB["blue"], CB["orange"], CB["green"]]
    x = np.arange(3)
    w = 0.42

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    bars = ax.bar(x, f1, width=w, color=colors, edgecolor="black", linewidth=0.8, zorder=3)

    # Value labels on bar tops
    for bar, v in zip(bars, f1):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10.5)

    # Dashed reference line at full-model F1
    full_f1 = f1[0]
    ax.axhline(full_f1, color="black", linestyle="--", linewidth=0.9,
               alpha=0.55, zorder=2)

    # Drop annotations: double-headed arrow + pp label between reference and bar top
    drop_pp = [(full_f1 - f1[1]) * 100, (full_f1 - f1[2]) * 100]   # 8.07, 11.38
    for i, pp in zip([1, 2], drop_pp):
        y_lo  = f1[i]
        y_hi  = full_f1
        x_bar = x[i]
        # Vertical bracket arrow
        ax.annotate(
            "", xy=(x_bar, y_lo + 0.003), xytext=(x_bar, y_hi - 0.003),
            arrowprops=dict(arrowstyle="<->", color="black", lw=1.1),
        )
        ax.text(x_bar + 0.23, (y_lo + y_hi) / 2,
                f"−{pp:.2f} pp", va="center", fontsize=10, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Macro-F1")
    y_lo_lim = max(0.0, f1.min() - 0.07)
    ax.set_ylim(y_lo_lim, f1.max() + 0.08)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=0)

    legend_patches = [
        Patch(facecolor=CB["blue"],   edgecolor="black", label="CNN-BiLSTM (Full)"),
        Patch(facecolor=CB["orange"], edgecolor="black", label="CNN-only"),
        Patch(facecolor=CB["green"],  edgecolor="black", label="BiLSTM-only"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=10,
              framealpha=0.93, edgecolor="gray")
    fig.tight_layout()
    save_fig(fig, "fig_ablation")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("[Inference] Loading tvae_ce_seed42.pt on test set ...")
    y_true, y_pred, class_names = run_inference()
    print(f"  test samples={len(y_true):,}  classes={len(class_names)}")

    plot_confusion_matrix(y_true, y_pred)
    plot_config_comparison()
    plot_perclass_f1()
    plot_ablation()

    print(f"\n[Done] 4 figures saved to {FIG}")
