# Traceability of numeric claims in the abstract

Every number the abstract cites is reproduced from these CSV rows.
Reviewers can grep `main_results.csv` with the (sampler, loss, alpha_mode, seed)
keys below to verify each figure.

| Abstract claim                          | Source row key                              |
|-----------------------------------------|---------------------------------------------|
| Macro-F1 83.91 (proposed)               | sampler=tvae, loss=ce, alpha_mode=N/A, seed=42 |
| Accuracy 99.76                          | same row, `accuracy` field                  |
| Macro-Precision 90.22                   | same row, `macro_precision` field           |
| +13.77 pp vs no aug                     | none/ce/N/A vs tvae/ce/N/A                  |
| +20.53 pp vs 'best oversampling'        | **ADASYN, not ROS** — see abstract_numbers.md |
| Focal Loss −6.26 to −14.50 pp           | tvae/focal/fixed and tvae/focal/sqrt_inverse |