# Baseline Methodology

## Overview

H1 baseline metrics are **empirically computed** by training simple models on the canonical EMBER-2024 train partition and evaluating on the held-out test partition. They are not taken from external literature.

## Baseline Models

### 1. Logistic Regression
- scikit-learn `LogisticRegression` (default parameters)
- Threshold: 0.5

### 2. Majority Classifier
- scikit-learn `DummyClassifier` with `strategy='most_frequent'`

### 3. Best Baseline
- Highest AUROC between the two (typically logistic regression)

## Example Values (current H1 run)

| Metric | Baseline | AICRA (aggregated) |
|--------|----------|-------------------|
| AUROC | 0.7811 | 0.9610 |
| Precision | 0.7726 | 0.6398* |
| Recall | 0.6378 | 0.9985 |
| F1 | 0.6988 | 0.7794 |
| FN rate | 36.2% | 0.20% |

*Lower precision reflects banking-optimized threshold (recall prioritized).

## Code

- `aicra/core/benchmarks.py` — `compute_h1_baselines()`
- `aicra/experiments/h1_classification.py` — H1 experiment

## Related Artifacts

- `results/H1_classification/H1_summary.md`
- `results/H1_classification/H1_full_results.json`
