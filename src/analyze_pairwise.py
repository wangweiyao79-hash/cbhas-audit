"""
analyze_pairwise.py — all pairwise paired-bootstrap contrasts among the nine
data-level strategies (CE loss throughout), closing the gap between Table 7
(which tests every strategy against CB-HAS only) and the RQ1 claim that no
strategy differs from any other.

Sources (per-seed Macro-F1):
  phase1_multiseed.csv   none+ce, ctgan+ce, tvae+ce (=CB-HAS)   seeds {7,42,123,1337,2024}
  phase3_results_v2.csv  ros/smote/adasyn_matched (decontaminated) seeds {42,123,2024}
  phase3_results_strictctgan.csv  ctgan_strict                     seeds {42,123,2024}
  phase2_results.csv     tvae_all, interp_all                      seeds {42,123,2024}

Output: results/tables/pairwise_bootstrap.csv  (36 rows)
Bootstrap identical to analyze_phase1.py (B=10,000, 95% quantile CI,
two-sided p = 2*min(P(diff<=0), P(diff>=0)); paired on common seeds).
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")
OUT = os.path.join(TABLE_DIR, "pairwise_bootstrap.csv")

def load(csvname, sampler, loss="ce"):
    df = pd.read_csv(os.path.join(TABLE_DIR, csvname))
    df["alpha_mode"] = df.get("alpha_mode", pd.Series(["N/A"] * len(df))).fillna("N/A")
    q = (df.sampler == sampler) & (df.loss == loss)
    rows = df[q].sort_values("seed")
    # de-duplicate (some CSVs contain a re-run row with alpha_mode both '' and 'N/A')
    rows = rows.drop_duplicates(subset="seed", keep="last")
    return {int(s): float(f) * 100 for s, f in zip(rows["seed"], rows["macro_f1"])}

CONFIGS = {
    "none":        load("phase1_multiseed.csv", "none"),
    "ROS-m":       load("phase3_results_v2.csv", "ros_matched"),
    "SMOTE-m":     load("phase3_results_v2.csv", "smote_matched"),
    "ADASYN-m":    load("phase3_results_v2.csv", "adasyn_matched"),
    "CTGAN":       load("phase1_multiseed.csv", "ctgan"),
    "strictCTGAN": load("phase3_results_strictctgan.csv", "ctgan_strict"),
    "TVAE-all":    load("phase2_results.csv", "tvae_all"),
    "Interp-all":  load("phase2_results.csv", "interp_all"),
    "CB-HAS":      load("phase1_multiseed.csv", "tvae"),
}

def paired_bootstrap(x, y, n_boot=10000, seed=0):
    x, y = np.asarray(x), np.asarray(y)
    d = x - y
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(d), len(d))
        means[b] = d[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min((means >= 0).mean(), (means <= 0).mean())
    return float(d.mean()), float(lo), float(hi), float(p)

rows = []
for a, b in itertools.combinations(CONFIGS, 2):
    common = sorted(set(CONFIGS[a]) & set(CONFIGS[b]))
    if len(common) < 2:
        print(f"[SKIP] {a} vs {b}: {len(common)} common seeds")
        continue
    x = [CONFIGS[a][s] for s in common]
    y = [CONFIGS[b][s] for s in common]
    md, lo, hi, p = paired_bootstrap(x, y)
    rows.append({
        "config_A": a, "config_B": b,
        "n_seeds": len(common), "seeds": ",".join(map(str, common)),
        "mean_delta_A_minus_B": round(md, 3),
        "CI95_lo": round(lo, 3), "CI95_hi": round(hi, 3),
        "p_value": round(p, 4),
        "significant_at_0.05": (lo > 0) or (hi < 0),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\n{len(df)} pairwise contrasts  →  {OUT}\n")
print(df.to_string(index=False))
sig = df[df["significant_at_0.05"]]
print(f"\nSignificant at 0.05: {len(sig)} of {len(df)}")
if len(sig):
    print(sig.to_string(index=False))
