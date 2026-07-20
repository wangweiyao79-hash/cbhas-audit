import pandas as pd

df = pd.read_csv(r'E:\ctgan-cbf-nids - new\results\tables\ciciot_results.csv')
ce = df[df['loss'] == 'ce']
focal = df[df['loss'] == 'focal']

print("=== CICIoT2023 汇总 ===")
print(f"CE   (3组): MF1={ce['macro_f1'].mean():.4f} ± {ce['macro_f1'].std():.4f}  [{ce['macro_f1'].min():.4f}, {ce['macro_f1'].max():.4f}]")
print(f"Focal(6组): MF1={focal['macro_f1'].mean():.4f} ± {focal['macro_f1'].std():.4f}  [{focal['macro_f1'].min():.4f}, {focal['macro_f1'].max():.4f}]")

fixed = focal[focal['alpha_mode'] == 'fixed']
sqrt_inv = focal[focal['alpha_mode'] == 'sqrt_inverse']
print(f"\nfixed(3组)     : MF1={fixed['macro_f1'].mean():.4f} ± {fixed['macro_f1'].std():.4f}")
print(f"sqrt_inv(3组)  : MF1={sqrt_inv['macro_f1'].mean():.4f} ± {sqrt_inv['macro_f1'].std():.4f}")
print(f"\n全部 9 组: MF1={df['macro_f1'].mean():.4f} ± {df['macro_f1'].std():.4f}  [{df['macro_f1'].min():.4f}, {df['macro_f1'].max():.4f}]")