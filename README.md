# CB-HAS Audit — Data Augmentation and Focal Loss for Class-Imbalanced NIDS

Code, per-seed results, and released artifacts for the paper:

> **Rethinking Data Augmentation and Focal Loss for Class-Imbalanced Intrusion
> Detection: A Multi-Seed, Budget-Fair Audit on CIC-IDS2017**
> Wei-Yao Wang. *PeerJ Computer Science* (under review).

The paper audits a complete augmentation pipeline (CB-HAS: cardinality-routed
TVAE + interpolation, vs. ROS / SMOTE / ADASYN / CTGAN / Focal Loss) under a
protocol most published comparisons omit: five seeds with paired bootstrap
CIs, strictly budget-matched baselines, and runtime invariant checks — and
documents a preprocessing contamination that manufactured p < 0.001
"advantages" (Section 4.3 of the paper). Contaminated results are retained
here, clearly marked `*_retracted` / `*_contaminated`, as a reference case.

## Repository layout

```
src/                    All experiment code (see "Reproducing" below)
results/tables/         Every CSV/markdown evidence file cited in the paper
results/figures/        Paper figures (PNG/PDF/SVG)
checkpoints/            Trained model weights (~550 KB each, 13.4e4-param model)
checkpoints/ciciot/     CICIoT2023 external-check weights (Section 5.9)
requirements.txt        Python dependencies (Python 3.9, PyTorch 2.6 + CUDA 12.4)
```

Key result files:

| File | Paper section |
|---|---|
| `main_results.csv` | Table 5 reference-seed results |
| `phase1_*.csv` | Table 6/7 multi-seed statistics |
| `phase3_results_v2.csv`, `phase3_bootstrap_ci_v2.csv` | Decontaminated matched baselines (Tables 3/5/7) |
| `phase3_results_strictctgan.csv` | Clean strict-CTGAN runs (Tables 6/7) |
| `phase3_results_contaminated.csv`, `main_results_default_retracted.csv` | **Retracted** contamination case study (Section 4.3) — invalid runs live only in `contaminated`/`retracted`-suffixed files |
| `phase4_sensitivity.csv` | Threshold sweep (Section 5.7) |
| `phase5_*.csv`, `phase6_*.csv` | Augmentation × loss 2×3 design (Table 8) |
| `pairwise_bootstrap.csv` | Full 36-contrast pairwise matrix (Section 5.4) |
| `rare_class_wilson.csv` | Wilson CIs for ultra-rare classes (Section 5.8) |
| `augmentation_budget.csv` | Per-class routing/budget audit (Table 2) |
| `ciciot_*.csv` | CICIoT2023 external check (Section 5.9, Table 10) |
| `ciciot_results_fixedsplit_retracted.csv` | **Retracted** early CICIoT2023 runs (split-seed coupling bug, disclosed) |

## Data

Raw datasets are **not** redistributed here (license and size). Download from
the Canadian Institute for Cybersecurity and place as follows:

- **CIC-IDS2017** — https://www.unb.ca/cic/datasets/ids-2017.html
  → the eight `*_ISCX.csv` files into `data/raw/`
- **CICIoT2023** — https://www.unb.ca/cic/datasets/iotdataset-2023.html
  → the merged CSV shards (`Merged01.csv` … `Merged63.csv`) into `MERGED_CSV/MERGED_CSV/`

All preprocessing is deterministic given the seed, so every split used in the
paper is exactly regenerable from these raw files.

## Environment

```bash
python -m venv venv
venv/Scripts/activate          # Windows
pip install -r requirements.txt
python src/check_env.py        # verifies CUDA + package versions
```

Reference hardware: single NVIDIA RTX 4060 Laptop GPU (8 GB). Every training
run in the paper fits this budget.

## Reproducing

**CIC-IDS2017 (main audit):**

```bash
python src/preprocess.py                      # clean + 6:2:2 split
python src/augment_tvae.py                    # CB-HAS augmentation (paper policy)
python src/augment_ctgan.py                   # CTGAN baseline augmentation
python src/train.py --sampler tvae --loss ce  # one configuration
bash  src/run_phase1.sh                       # multi-seed grid (Table 6/7)
bash  src/run_phase2_and_3.sh                 # routing ablations + matched baselines
bash  src/run_phase4.sh                       # threshold sensitivity
python src/analyze_phase1.py                  # bootstrap CIs
```

`train.py --budget matched` reproduces the budget-fair baselines; the
runtime invariant assertions (non-augmented classes byte-for-byte preserved)
are active in every resampling path and will abort on any silent contamination
of the kind documented in Section 4.3.

**CICIoT2023 (external check, Section 5.9):**

```bash
python src/run_ciciot_grid.py                 # 5 seeds x {CE, FL-fixed, FL-sqrtinv};
                                              # per-seed re-split; resumable
python src/analyze_ciciot.py                  # summary + paired bootstrap
```

**Verifying paper numbers without retraining:** every `checkpoints/*.pt` can
be loaded with `src/model.py`'s `CNNBiLSTM` and evaluated directly against the
corresponding split to reproduce the tables.

## Citation

```bibtex
@article{wang_cbhas_audit,
  author  = {Wang, Wei-Yao},
  title   = {Rethinking Data Augmentation and Focal Loss for Class-Imbalanced
             Intrusion Detection: A Multi-Seed, Budget-Fair Audit on CIC-IDS2017},
  journal = {PeerJ Computer Science},
  note    = {under review}
}
```
