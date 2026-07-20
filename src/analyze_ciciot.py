"""
analyze_ciciot.py — aggregate CICIoT2023 external-validation results.

Reads results/tables/ciciot_results.csv (none x {ce, focal-fixed, focal-sqrtinv}
x N seeds, produced by run_ciciot_grid.py on the per-seed-split pipeline) and
emits:

  results/tables/ciciot_summary.csv        — mean, std, min, max per config
  results/tables/ciciot_bootstrap_ci.csv   — paired bootstrap 95% CI + p-value

Comparisons produced (paired by seed), mirroring the CIC-IDS2017 Table 8
none-arm contrasts so the two datasets can be read side by side:

  none+FL(fixed)    vs none+CE     (external check of the RQ2 equivalence claim)
  none+FL(sqrt_inv) vs none+CE     (external check of the robust-harm claim)

Bootstrap procedure identical to analyze_phase1.py: B=10,000 resamples of the
K seeds with replacement; 95% CI from the 2.5/97.5 quantiles; two-sided
p = 2 * min(P(diff<=0), P(diff>=0)).

Also prints the cross-dataset seed-variance comparison used for RQ3
(CICIoT2023 std vs CIC-IDS2017's 4.55 pp none+CE std).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")
CSV_IN    = os.path.join(TABLE_DIR, "ciciot_results.csv")
SUMMARY   = os.path.join(TABLE_DIR, "ciciot_summary.csv")
BOOTSTRAP = os.path.join(TABLE_DIR, "ciciot_bootstrap_ci.csv")

METRIC = "macro_f1"

# CIC-IDS2017 reference values (phase1_summary.csv / phase5) for the RQ3
# cross-dataset variance comparison printed at the end.
CIC2017_NONE_CE_STD = 4.552
CIC2017_SQRTINV_DELTA = -7.423   # none+FL(sqrt_inv) - none+CE, pp


def paired_bootstrap(x, y, n_boot=10000, seed=0):
    """x, y are K-length arrays aligned by seed. Returns (mean_diff, lo, hi, p)."""
    x, y = np.asarray(x), np.asarray(y)
    d = x - y
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        means[b] = d[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    p_pos = (means >= 0).mean()
    p_neg = (means <= 0).mean()
    p = 2 * min(p_pos, p_neg)
    return float(d.mean()), float(lo), float(hi), float(p)


def main():
    if not os.path.exists(CSV_IN):
        print(f"[ERR] {CSV_IN} not found. Run src/run_ciciot_grid.py first.")
        return
    df = pd.read_csv(CSV_IN)
    df["alpha_mode"] = df["alpha_mode"].fillna("N/A")

    # Guard against accidentally analyzing the retracted fixed-split results:
    # the new pipeline always writes a separate `tag` column.
    assert "tag" in df.columns, (
        "[FATAL] no `tag` column — this looks like the retracted fixed-split "
        "CSV, not the per-seed-split results.")

    configs = [
        ("CE",       "none", "ce",    "N/A"),
        ("FL-fixed", "none", "focal", "fixed"),
        ("FL-sqrt",  "none", "focal", "sqrt_inverse"),
    ]

    # ── 1. Per-config summary ────────────────────────────────────────────────
    summary_rows = []
    seed_map = {}
    for tag, sampler, loss, alpha in configs:
        q = (df.sampler == sampler) & (df.loss == loss) & (df.alpha_mode == alpha)
        rows = df[q].sort_values("seed").reset_index(drop=True)
        f1s   = rows[METRIC].values * 100
        seeds = rows["seed"].values
        seed_map[tag] = {int(s): float(f) for s, f in zip(seeds, f1s)}
        summary_rows.append({
            "tag"     : tag,
            "config"  : f"{sampler}+{loss}" + (f"({alpha})" if loss == "focal" else ""),
            "n_seeds" : len(f1s),
            "seeds"   : ",".join(map(str, seeds)),
            "mean_f1" : round(f1s.mean(), 3) if len(f1s) else float("nan"),
            "std_f1"  : round(f1s.std(ddof=1), 3) if len(f1s) > 1 else float("nan"),
            "min_f1"  : round(f1s.min(), 3) if len(f1s) else float("nan"),
            "max_f1"  : round(f1s.max(), 3) if len(f1s) else float("nan"),
        })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(SUMMARY, index=False)
    print("\n── CICIoT2023 multiseed summary (Macro-F1, %) ──────────────────")
    print(df_sum.to_string(index=False))
    print(f"\n  Saved → {SUMMARY}")

    # ── 2. Paired bootstrap: FL variants vs CE ───────────────────────────────
    boot_rows = []
    for a, b, desc in [("FL-fixed", "CE", "none+FL(fixed) vs none+CE"),
                       ("FL-sqrt",  "CE", "none+FL(sqrt_inv) vs none+CE")]:
        seeds_common = sorted(set(seed_map[a].keys()) & set(seed_map[b].keys()))
        if len(seeds_common) < 2:
            print(f"\n  [SKIP] {desc}: only {len(seeds_common)} common seeds")
            continue
        x = [seed_map[a][s] for s in seeds_common]
        y = [seed_map[b][s] for s in seeds_common]
        md, lo, hi, p = paired_bootstrap(x, y)
        boot_rows.append({
            "comparison" : desc,
            "n_seeds"    : len(seeds_common),
            "seeds"      : ",".join(map(str, seeds_common)),
            "mean_delta" : round(md, 3),
            "CI95_lo"    : round(lo, 3),
            "CI95_hi"    : round(hi, 3),
            "p_value"    : round(p, 4),
            "significant_at_0.05" : (lo > 0) or (hi < 0),
        })
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(BOOTSTRAP, index=False)
    print("\n── Paired bootstrap 95% CI (delta = FL - CE, %-points) ─────────")
    if len(df_boot):
        print(df_boot.to_string(index=False))
    print(f"\n  Saved → {BOOTSTRAP}")

    # ── 3. Cross-dataset comparison for the paper (Sec 5.10 material) ────────
    print("\n── Cross-dataset reference (for v6 Sec 5.10) ───────────────────")
    ce_std = next((r["std_f1"] for r in summary_rows if r["tag"] == "CE"), float("nan"))
    print(f"  none+CE seed std : CICIoT2023 {ce_std} pp  vs  CIC-IDS2017 {CIC2017_NONE_CE_STD} pp")
    sqrt_row = next((r for r in boot_rows if "sqrt" in r["comparison"]), None)
    if sqrt_row:
        print(f"  FL(sqrt_inv)-CE  : CICIoT2023 {sqrt_row['mean_delta']:+.2f} pp "
              f"[{sqrt_row['CI95_lo']:+.2f}, {sqrt_row['CI95_hi']:+.2f}]  "
              f"vs  CIC-IDS2017 {CIC2017_SQRTINV_DELTA:+.2f} pp (p<0.001)")
    print("  Smallest test-class support: CICIoT2023 ~239 (UPLOADING_ATTACK)"
          "  vs  CIC-IDS2017 2/4/7 (Heartbleed/SQLi/Infiltration)")


if __name__ == "__main__":
    main()
