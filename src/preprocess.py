import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding="utf-8")

RAW_DIR       = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
PROC_DIR      = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
RESULTS_DIR   = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
TABLES_DIR    = os.path.join(RESULTS_DIR, "tables")

os.makedirs(PROC_DIR,  exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

def log(msg=""):
    print(msg)

# ── 1. 读取并合并 ─────────────────────────────────────────────────────────────
log("=" * 65)
log("[1] 读取并合并 8 个 CSV")
log("=" * 65)
csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
dfs = []
for f in csv_files:
    df = pd.read_csv(f, low_memory=False)
    log(f"  {os.path.basename(f):55s} {len(df):>9,} 行")
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)
log(f"\n  合并后: {len(combined):,} 行 x {combined.shape[1]} 列")

# ── 2. 列名清洗 ───────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("[2] 列名清洗（strip 前导/尾随空格）")
log("=" * 65)
combined.columns = combined.columns.str.strip()
log(f"  标签列确认: '{combined.columns[-1]}'  (最后一列)")

# ── 3. 脏数据处理：Inf → NaN → 删行 ─────────────────────────────────────────
log()
log("=" * 65)
log("[3] Infinity → NaN → 删除含 NaN 行")
log("=" * 65)
n_before = len(combined)
combined.replace([np.inf, -np.inf], np.nan, inplace=True)
combined.dropna(inplace=True)
n_after = len(combined)
log(f"  删除含 NaN/Inf 行: {n_before - n_after:,}  (剩余 {n_after:,} 行)")

# ── 4. 去重 ───────────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("[4] 去重")
log("=" * 65)
n_before = len(combined)
combined.drop_duplicates(inplace=True)
combined.reset_index(drop=True, inplace=True)
n_after = len(combined)
log(f"  删除重复行: {n_before - n_after:,}  (剩余 {n_after:,} 行)")

# ── 5. 分离 X / y ─────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("[5] 分离特征 X 和标签 y")
log("=" * 65)
X = combined.drop(columns=["Label"]).select_dtypes(include=[np.number])
y = combined["Label"]
log(f"  X shape: {X.shape}  (特征维数 d = {X.shape[1]})")
log(f"  y shape: {y.shape}")

# ── 6. 类别分布表 ─────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("[6] 类别分布（table2_class_distribution.csv）")
log("=" * 65)
counts = y.value_counts()
total  = counts.sum()
dist_df = pd.DataFrame({
    "Label":   counts.index,
    "Count":   counts.values,
    "Percent": (counts.values / total * 100).round(2),
}).reset_index(drop=True)
dist_df.index += 1          # 1-based 序号

csv_path = os.path.join(TABLES_DIR, "table2_class_distribution.csv")
dist_df.to_csv(csv_path, index_label="No.")
log(f"  已保存: {csv_path}")
log()
log(f"  {'No.':>4}  {'Label':<40} {'Count':>10}  {'Percent':>8}")
log(f"  {'-'*4}  {'-'*40} {'-'*10}  {'-'*8}")
for idx, row in dist_df.iterrows():
    log(f"  {idx:>4}  {row['Label']:<40} {int(row['Count']):>10,}  {row['Percent']:>7.2f}%")
log(f"\n  共 {len(dist_df)} 个类别，总样本 {total:,}")

# ── 7. 标签编码 ───────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("[7] 标签编码（LabelEncoder）")
log("=" * 65)
le = LabelEncoder()
y_enc = le.fit_transform(y)
mapping_path = os.path.join(RESULTS_DIR, "label_mapping.txt")
with open(mapping_path, "w", encoding="utf-8") as f:
    f.write("编号 -> 类别名\n")
    f.write("-" * 40 + "\n")
    for idx, name in enumerate(le.classes_):
        f.write(f"  {idx:2d}  {name}\n")
log(f"  标签映射已保存: {mapping_path}")
for idx, name in enumerate(le.classes_):
    log(f"    {idx:2d} -> {name}")

# ── 8. 分层切分 + MinMaxScaler（先切后 fit，避免数据泄露）────────────────────
log()
log("=" * 65)
log("[8] 分层切分 6:2:2 + MinMaxScaler（fit on train only）")
log("=" * 65)
X_np = X.values.astype(np.float32)

# 先按 8:2 切出 test，再在剩余 80% 中按 7.5:2.5 = 6:2 切出 val
X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(
    X_np, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
X_train, X_val, y_train, y_val   = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train).astype(np.float32)
X_val   = scaler.transform(X_val).astype(np.float32)
X_test  = scaler.transform(X_test).astype(np.float32)

log(f"  Train : {X_train.shape[0]:>8,} 样本")
log(f"  Val   : {X_val.shape[0]:>8,} 样本")
log(f"  Test  : {X_test.shape[0]:>8,} 样本")

# ── 9. 保存 ──────────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("[9] 保存到 data/processed/（.npz）")
log("=" * 65)

feature_names = X.columns.tolist()

np.savez_compressed(os.path.join(PROC_DIR, "train.npz"),
                    X=X_train, y=y_train)
np.savez_compressed(os.path.join(PROC_DIR, "val.npz"),
                    X=X_val,   y=y_val)
np.savez_compressed(os.path.join(PROC_DIR, "test.npz"),
                    X=X_test,  y=y_test)

# 保存 scaler 和 label encoder 供后续推理使用
import pickle
with open(os.path.join(PROC_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(PROC_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)
with open(os.path.join(PROC_DIR, "feature_names.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(feature_names))

log(f"  train.npz / val.npz / test.npz  -> {PROC_DIR}")
log(f"  scaler.pkl / label_encoder.pkl  -> {PROC_DIR}")

# ── 10. 最终统计 ──────────────────────────────────────────────────────────────
log()
log("=" * 65)
log("[10] 最终统计")
log("=" * 65)
log(f"  清洗后总样本数 : {len(X_np):,}")
log(f"  特征维数 d     : {X.shape[1]}")
log(f"  Train 样本数   : {X_train.shape[0]:,}  ({X_train.shape[0]/len(X_np)*100:.1f}%)")
log(f"  Val   样本数   : {X_val.shape[0]:,}  ({X_val.shape[0]/len(X_np)*100:.1f}%)")
log(f"  Test  样本数   : {X_test.shape[0]:,}  ({X_test.shape[0]/len(X_np)*100:.1f}%)")
log()
log("  预处理完成。")
