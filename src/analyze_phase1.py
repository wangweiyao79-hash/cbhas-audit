"""
analyze_phase1.py — aggregate multiseed results for M1.

Reads results/tables/phase1_multiseed.csv (4 configs x 5 seeds each) and emits:

  results/tables/phase1_summary.csv          — mean, std, min, max per config
  results/tables/phase1_bootstrap_ci.csv     — paired bootstrap 95% CI + p-value
                                                for three key comparisons

Comparisons produced (paired by seed):
  C vs A : CB-HAS (tvae+ce)     vs no aug (none+ce)
  C vs B : CB-HAS (tvae+ce)     vs CTGAN (ctgan+ce)
  C vs D : CB-HAS (tvae+ce)     vs CB-HAS+FL (tvae+focal/sqrt_inverse)

Paired bootstrap:
  For each of B=10000 resamples, sample the K seeds with replacement,
  compute the per-seed Macro-F1 difference, take the mean, report the
  2.5% and 97.5% quantiles as the 95% CI, and the two-sided p-value as
  2 * min(P(diff<=0), P(diff>=0)).
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")
CSV_IN    = os.path.join(TABLE_DIR, "phase1_multiseed.csv")
SUMMARY   = os.path.join(TABLE_DIR, "phase1_summary.csv")
BOOTSTRAP = os.path.join(TABLE_DIR, "phase1_bootstrap_ci.csv")

METRIC = "macro_f1"


def key(sampler, loss, alpha):
    return f"{sampler}+{loss}" + (f"({alpha})" if loss == "focal" else "")


def config_rows(df, sampler, loss, alpha):
    q = (df.sampler == sampler) & (df.loss == loss)
    if loss == "focal":
        q &= (df.alpha_mode == alpha)
    return df[q].sort_values("seed").reset_index(drop=True)


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
    # Two-sided p-value: how often the resampled mean crosses 0
    p_pos = (means >= 0).mean()
    p_neg = (means <= 0).mean()
    p = 2 * min(p_pos, p_neg)
    return float(d.mean()), float(lo), float(hi), float(p)


def main():
    if not os.path.exists(CSV_IN):
        print(f"[ERR] {CSV_IN} not found. Run src/run_phase1.ps1 first.")
        return
    df = pd.read_csv(CSV_IN)
    df["alpha_mode"] = df["alpha_mode"].fillna("N/A")

    configs = [
        ("A", "none",  "ce",    "N/A"),
        ("B", "ctgan", "ce",    "N/A"),
        ("C", "tvae",  "ce",    "N/A"),
        ("D", "tvae",  "focal", "sqrt_inverse"),
    ]

    # ── 1. Per-config summary ─────────────────────────────────────────────────
    summary_rows = []
    seed_map = {}
    for tag, sampler, loss, alpha in configs:
        rows = config_rows(df, sampler, loss, alpha)
        f1s  = rows[METRIC].values * 100
        seeds = rows["seed"].values
        seed_map[tag] = {int(s): float(f) for s, f in zip(seeds, f1s)}
        summary_rows.append({
            "tag"      : tag,
            "config"   : key(sampler, loss, alpha),
            "n_seeds"  : len(f1s),
            "seeds"    : ",".join(map(str, seeds)),
            "mean_f1"  : round(f1s.mean(), 3) if len(f1s) else float("nan"),
            "std_f1"   : round(f1s.std(ddof=1), 3) if len(f1s) > 1 else float("nan"),
            "min_f1"   : round(f1s.min(), 3) if len(f1s) else float("nan"),
            "max_f1"   : round(f1s.max(), 3) if len(f1s) else float("nan"),
        })
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(SUMMARY, index=False)
    print("\n── Multiseed summary (Macro-F1, %) ─────────────────────────────")
    print(df_sum.to_string(index=False))
    print(f"\n  Saved → {SUMMARY}")

    # ── 2. Paired bootstrap for 3 key comparisons ────────────────────────────
    def paired(a_tag, b_tag):
        seeds_common = sorted(set(seed_map[a_tag].keys()) & set(seed_map[b_tag].keys()))
        if len(seeds_common) < 2:
            return None, None, None, None, seeds_common
        x = [seed_map[a_tag][s] for s in seeds_common]
        y = [seed_map[b_tag][s] for s in seeds_common]
        return paired_bootstrap(x, y) + (seeds_common,)

    boot_rows = []
    for a, b, desc in [("C", "A", "CB-HAS vs no aug"),
                       ("C", "B", "CB-HAS vs CTGAN"),
                       ("C", "D", "CB-HAS+CE vs CB-HAS+FL(sqrt_inv)")]:
        result = paired(a, b)
        if result[0] is None:
            print(f"\n  [SKIP] {desc}: only {len(result[-1])} common seeds")
            continue
        md, lo, hi, p, seeds = result
        boot_rows.append({
            "comparison" : desc,
            "n_seeds"    : len(seeds),
            "seeds"      : ",".join(map(str, seeds)),
            "mean_delta" : round(md, 3),
            "CI95_lo"    : round(lo, 3),
            "CI95_hi"    : round(hi, 3),
            "p_value"    : round(p, 4),
            "significant_at_0.05" : (lo > 0) or (hi < 0),
        })
    df_boot = pd.DataFrame(boot_rows)
    df_boot.to_csv(BOOTSTRAP, index=False)
    print("\n── Paired bootstrap 95% CI (delta = A - B, %-points) ───────────")
    if len(df_boot):
        print(df_boot.to_string(index=False))
    print(f"\n  Saved → {BOOTSTRAP}")


if __name__ == "__main__":
    main()
