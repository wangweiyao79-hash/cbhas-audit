import os
import sys
import glob
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "data_check.txt")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

lines = []

def log(msg=""):
    print(msg)
    lines.append(str(msg))

# ── 1. 文件列表及大小 ────────────────────────────────────────────────────────
csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
log("=" * 70)
log(f"[1] CSV 文件列表（共 {len(csv_files)} 个）")
log("=" * 70)
for f in csv_files:
    size_mb = os.path.getsize(f) / 1024 / 1024
    log(f"  {os.path.basename(f):50s}  {size_mb:8.2f} MB")

# ── 2. 逐个读取：行数、列数 ──────────────────────────────────────────────────
log()
log("=" * 70)
log("[2] 各文件行数 / 列数")
log("=" * 70)
dfs = []
all_columns = {}
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    name = os.path.basename(f)
    log(f"  {name:50s}  {len(df):>8,} 行  {df.shape[1]:>3} 列")
    dfs.append(df)
    all_columns[name] = list(df.columns)

# ── 3. 列名一致性检查 ────────────────────────────────────────────────────────
log()
log("=" * 70)
log("[3] 列名一致性检查")
log("=" * 70)
reference_name = os.path.basename(csv_files[0])
reference_cols = all_columns[reference_name]
log(f"  基准文件: {reference_name}  ({len(reference_cols)} 列)")
log()

all_same = True
for name, cols in all_columns.items():
    if cols == reference_cols:
        log(f"  [OK] {name}")
    else:
        all_same = False
        log(f"  [!!] {name}  <- 列名不一致!")
        extra   = set(cols) - set(reference_cols)
        missing = set(reference_cols) - set(cols)
        if extra:
            log(f"      多出列: {extra}")
        if missing:
            log(f"      缺失列: {missing}")

log()
if all_same:
    log("  >> 所有文件列名完全一致。")
    log()
    log("  列名列表:")
    for i, c in enumerate(reference_cols, 1):
        log(f"    {i:3d}. {c}")
else:
    log("  >> 存在列名不一致，请核查上方标注。")

# ── 4. 合并 ──────────────────────────────────────────────────────────────────
log()
log("=" * 70)
log("[4] 合并 DataFrame")
log("=" * 70)
combined = pd.concat(dfs, ignore_index=True)
log(f"  合并后总行数: {len(combined):,}")
log(f"  合并后总列数: {combined.shape[1]}")

# ── 5. 标签列分布 ─────────────────────────────────────────────────────────────
log()
log("=" * 70)
log("[5] 标签列分布")
log("=" * 70)
label_col = None
for candidate in combined.columns:
    if candidate.strip() == "Label":
        label_col = candidate
        break

if label_col is None:
    log("  !! 未找到标签列（Label / ' Label'）")
else:
    log(f"  标签列名: '{label_col}'  (原始名含空格: {repr(label_col)})")
    log()
    counts = combined[label_col].value_counts()
    total = counts.sum()
    log(f"  {'类别':<40} {'样本数':>10}  {'占比':>7}")
    log(f"  {'-'*40} {'-'*10}  {'-'*7}")
    for label, cnt in counts.items():
        log(f"  {str(label):<40} {cnt:>10,}  {cnt/total*100:>6.2f}%")
    log()
    log(f"  共 {len(counts)} 个类别，总样本 {total:,}")

# ── 6. Inf / NaN 统计 ─────────────────────────────────────────────────────────
log()
log("=" * 70)
log("[6] Infinity / NaN 统计")
log("=" * 70)
numeric_df = combined.select_dtypes(include=[np.number])

inf_total = np.isinf(numeric_df).sum().sum()
nan_total = combined.isna().sum().sum()
log(f"  Infinity 总数: {inf_total:,}")
log(f"  NaN      总数: {nan_total:,}")

if inf_total > 0:
    inf_by_col = np.isinf(numeric_df).sum()
    inf_cols = inf_by_col[inf_by_col > 0]
    log(f"\n  含 Inf 的列（共 {len(inf_cols)} 列）:")
    for col, cnt in inf_cols.items():
        log(f"    {col:<45} {cnt:>8,}")

if nan_total > 0:
    nan_by_col = combined.isna().sum()
    nan_cols = nan_by_col[nan_by_col > 0]
    log(f"\n  含 NaN 的列（共 {len(nan_cols)} 列）:")
    for col, cnt in nan_cols.items():
        log(f"    {col:<45} {cnt:>8,}")

# ── 保存 ──────────────────────────────────────────────────────────────────────
output_path = os.path.normpath(OUTPUT_PATH)
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nSaved to {output_path}")
