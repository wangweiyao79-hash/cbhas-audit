# Abstract number reconciliation (M2)

CB-HAS+CE Macro-F1: **83.91**  (Table 4 value)

| Claim                                              | Baseline F1 | Δ (pp) |
|----------------------------------------------------|-------------|--------|
| vs none+CE (paper claims +13.77)                     |      70.14  |  +13.77 |
| vs ROS+CE (best oversampling by F1?)                 |      65.55  |  +18.36 |
| vs SMOTE+CE                                          |      61.38  |  +22.53 |
| vs ADASYN+CE (paper text implies 'best oversampling') |      63.38  |  +20.53 |
| vs CTGAN+CE (strongest data-level baseline)          |      79.79  |   +4.12 |

**Interpretation for M2**:
- ROS, SMOTE, ADASYN are the three 'traditional oversampling' methods.
- The maximum among their Macro-F1 values is the true 'best oversampling'.
- The abstract's `20.53 pp over best oversampling` matches ADASYN (63.38%), not the maximum.
- Reviewers M2 asks for one of two fixes:
  (a) Rewrite abstract to say 'best interpolation-based method' and cite ADASYN, or
  (b) Rewrite abstract to say 'strongest data-level baseline (CTGAN)' and cite +4.12 pp.

- Accuracy sanity check: 99.76 - 98.43 = **1.32 pp**  (paper text says 1.32; correct value is above).