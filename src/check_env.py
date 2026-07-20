import sys
import os

lines = []

def log(msg=""):
    print(msg)
    lines.append(msg)

log(f"Python: {sys.version}")
log()

libs = [
    ("torch",           lambda m: m.__version__),
    ("sklearn",         lambda m: m.__version__),
    ("pandas",          lambda m: m.__version__),
    ("numpy",           lambda m: m.__version__),
    ("matplotlib",      lambda m: m.__version__),
    ("imblearn",        lambda m: m.__version__),
    ("ctgan",           lambda m: m.__version__),
    ("sdv",             lambda m: m.__version__),
    ("tqdm",            lambda m: m.__version__),
    ("yaml",            lambda m: m.__version__),
]

for mod_name, ver_fn in libs:
    try:
        import importlib
        m = importlib.import_module(mod_name)
        log(f"{mod_name}: {ver_fn(m)}")
    except Exception as e:
        log(f"{mod_name}: ERROR - {e}")

log()

try:
    import torch
    cuda_available = torch.cuda.is_available()
    log(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            log(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        log("  No GPU detected (CPU only)")
except Exception as e:
    log(f"torch CUDA check ERROR: {e}")

output_path = os.path.join(os.path.dirname(__file__), "..", "results", "env_info.txt")
output_path = os.path.normpath(output_path)
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"\nSaved to {output_path}")
