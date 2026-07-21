"""
analyze_interaction.py — formal augmentation × loss interaction contrasts (RQ2).

Table 8 reports within-arm simple effects (FL − CE inside each data level).
The interaction RQ2 actually poses is the difference-in-differences:

    dd_s = (F1[CB-HAS, FL, s] − F1[CB-HAS, CE, s])
         − (F1[none,  FL, s] − F1[none,  CE, s])

computed per seed s, for each Focal variant. This script reports the mean
interaction contrast with the study's standard paired bootstrap (10,000
resamples, percentile CI, two-sided tail probability) plus a paired t-test.

Output: results/tables/interaction_contrasts.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")

TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "tables")
OUT = os.path.join(TABLE_DIR, "interaction_contrasts.csv")


def vals(csv, sampler, loss="ce", alpha=None):
    df = pd.read_csv(os.path.join(TABLE_DIR, csv))
    df["alpha_mode"] = df["alpha_mode"].fillna("N/A")
    q = (df.sampler == sampler) & (df.loss == loss)
    if alpha:
        q &= (df.alpha_mode == alpha)
    r = df[q].drop_duplicates(subset="seed", keep="last").sort_values("seed")
    return {int(s): float(f) * 100 for s, f in zip(r.seed, r.macro_f1)}


def paired_bootstrap(d, n_boot=10000, seed=0):
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = d[rng.integers(0, len(d), len(d))].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    p = 2 * min((means >= 0).mean(), (means <= 0).mean())
    return float(lo), float(hi), float(p)


none_ce = vals("phase1_multiseed.csv", "none")
cb_ce = vals("phase1_multiseed.csv", "tvae")
arms = {
    "fixed-alpha": (vals("phase6_cbhas_flfixed_multiseed.csv", "tvae", "focal", "fixed"),
                    vals("phase5_nonefl_multiseed.csv", "none", "focal", "fixed")),
    "sqrt-inverse": (vals("phase1_multiseed.csv", "tvae", "focal", "sqrt_inverse"),
                     vals("phase5_nonefl_multiseed.csv", "none", "focal", "sqrt_inverse")),
}

rows = []
for name, (cb_fl, none_fl) in arms.items():
    seeds = sorted(set(none_ce) & set(cb_ce) & set(cb_fl) & set(none_fl))
    dd = np.array([(cb_fl[s] - cb_ce[s]) - (none_fl[s] - none_ce[s]) for s in seeds])
    lo, hi, p_boot = paired_bootstrap(dd)
    t, p_t = stats.ttest_rel([cb_fl[s] - cb_ce[s] for s in seeds],
                             [none_fl[s] - none_ce[s] for s in seeds])
    rows.append({
        "contrast": f"interaction ({name}): (CB-HAS FL−CE) − (none FL−CE)",
        "n_seeds": len(seeds), "seeds": ",".join(map(str, seeds)),
        "per_seed_dd": ";".join(f"{x:+.3f}" for x in dd),
        "mean_dd": round(dd.mean(), 3), "sd_dd": round(dd.std(ddof=1), 3),
        "boot_CI95_lo": round(lo, 3), "boot_CI95_hi": round(hi, 3),
        "boot_tail_p": round(p_boot, 4),
        "t_stat": round(float(t), 3), "t_p": round(float(p_t), 4),
    })

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(df.to_string(index=False))
print(f"\nSaved → {OUT}")
