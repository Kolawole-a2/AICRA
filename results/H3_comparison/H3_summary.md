
# H3 Deterministic vs Learned Mapping — Comparison

## Mapping Integrity

- Deterministic Coverage: **100.0%** (PASS)  
- Deterministic Consistency: **100.0%** (PASS)  
- Learned Coverage: **100.0%** (PASS)  
- Learned Consistency: **100.0%** (PASS)

## Actionable Precision (Register-Level)

- Deterministic Precision: **0.9941**  
- Learned Precision: **0.9941**  
- Δ Precision (Det - Learn): **0.0**  
- Wilcoxon p-value (paired actionable correctness): **None**

## Score Consistency (Stability)

- Deterministic Variance Reduction: **0.004956**  
- Learned Variance Reduction: **0.004956**  
- Δ Variance Reduction (Det - Learn): **0.0**

## Baseline Discrimination & Calibration (for context)

- AUROC: **0.9984300940438872**, PR-AUC: **0.99938730235757**  
- Brier: **0.054733**, ECE: **0.180211**

## Expected Loss (optional)

- Sum(Expected Loss): **None**

## Reproducibility

- Deterministic lookup SHA256: `46a0ac102ab150b8d2909b97190232f97b9e9583ae1d83b2a704ebf6408a9ee4`
- Learned mapping SHA256: `87816619570f80b598c933009d9c914887e8201c5efc2c61b9b459bcbadc30b6`
- Reference pairs SHA256: `46a0ac102ab150b8d2909b97190232f97b9e9583ae1d83b2a704ebf6408a9ee4`

## Decision Guide

- Prefer deterministic mapping if it achieves **coverage ≥ 85.0%**, **consistency ≥ 90.0%**, **Δ precision > 0**, and **variance reduction > 0**.
