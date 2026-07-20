"""
analyze_phase23_extra.py — multi-seed variance check for M3/M5/M6.

Phase 2/3 originally shipped single-seed (seed=42) point estimates for the
routing ablation (M3: tvae_all, interp_all) and the matched-budget /
strict-CTGAN baselines (M5/M6). src/run_phase2_3_extra_seeds.sh added seeds
123 and 2024 for those six configs. This script pairs them against CB-HAS
(tvae+ce) at the same three seeds and runs the same paired-bootstrap
procedure as analyze_phase1.py.

Reads:
  results/tables/main_results.csv      (CB-HAS seed=42)
  results/tables/phase1_multiseed.csv  (CB-HAS seed=123,2024)
  results/tables/phase2_results.csv    (tvae_all, interp_all)
  results/tables/phase3_results.csv    (ros/smote/adasyn_matched, ctgan_strict)

Emits:
  results/tables/phase23_extra_bootstrap_ci.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")
OUT = os.path.join(TABLE_DIR, "phase23_extra_bootstrap_ci.csv")

SEEDS = [42, 123, 2024]


def paired_bootstrap(x, y, n_boot=10000, seed=0):
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
    main_df = pd.read_csv(os.path.join(TABLE_DIR, "main_results.csv"))
    p1_df   = pd.read_csv(os.path.join(TABLE_DIR, "phase1_multiseed.csv"))
    p2_df   = pd.read_csv(os.path.join(TABLE_DIR, "phase2_results.csv"))
    p3_df   = pd.read_csv(os.path.join(TABLE_DIR, "phase3_results.csv"))

    cbhas = {}
    r42 = main_df[(main_df.sampler == "tvae") & (main_df.loss == "ce") & (main_df.seed == 42)]
    cbhas[42] = r42.iloc[0].macro_f1 * 100
    for s in [123, 2024]:
        r = p1_df[(p1_df.sampler == "tvae") & (p1_df.loss == "ce") & (p1_df.seed == s)]
        cbhas[s] = r.iloc[0].macro_f1 * 100

    def get(df, tag, s):
        r = df[(df.sampler == tag) & (df.loss == "ce") & (df.seed == s)]
        return r.iloc[0].macro_f1 * 100 if len(r) else None

    configs = [
        ("tvae_all",      p2_df, "CB-HAS vs TVAE-all (M3)"),
        ("interp_all",    p2_df, "CB-HAS vs Interp-all (M3)"),
        ("ros_matched",   p3_df, "CB-HAS vs ROS matched (M5)"),
        ("smote_matched", p3_df, "CB-HAS vs SMOTE matched (M5)"),
        ("adasyn_matched",p3_df, "CB-HAS vs ADASYN matched (M5)"),
        ("ctgan_strict",  p3_df, "CB-HAS vs CTGAN strict (M6)"),
    ]

    rows = []
    for tag, df, desc in configs:
        vals = {s: get(df, tag, s) for s in SEEDS}
        seeds_common = [s for s in SEEDS if vals[s] is not None]
        x = [cbhas[s] for s in seeds_common]
        y = [vals[s] for s in seeds_common]
        md, lo, hi, p = paired_bootstrap(x, y)
        rows.append({
            "comparison": desc,
            "n_seeds": len(seeds_common),
            "seeds": ",".join(map(str, seeds_common)),
            "mean_delta": round(md, 3),
            "CI95_lo": round(lo, 3),
            "CI95_hi": round(hi, 3),
            "p_value": round(p, 4),
            "significant_at_0.05": (lo > 0) or (hi < 0),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT, index=False)
    print("── M3/M5/M6 multi-seed paired bootstrap 95% CI (delta = CB-HAS - other, %-points) ──")
    print(df_out.to_string(index=False))
    print(f"\n  Saved -> {OUT}")


if __name__ == "__main__":
    main()
