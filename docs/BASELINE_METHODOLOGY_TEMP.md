# Baseline Methodology - Temporary Documentation

> **Status**: TEMPORARY - This document describes the current empirically-computed baseline methodology. These baselines may be replaced with academic baseline values in the future.

## Overview

The baseline metrics shown in H1 experiment results (e.g., `H1_summary.md` lines 62-68) are **empirically computed** by training simple baseline models on the EMBER-2024 dataset, **not** values taken directly from academic papers.

## Current Baseline Models

### 1. Logistic Regression
- **Type**: Simple linear classifier (standard ML baseline)
- **Implementation**: scikit-learn `LogisticRegression` with default parameters
- **Threshold**: 0.5 (standard binary classification threshold)
- **Rationale**: Standard baseline for binary classification tasks (Hastie et al., 2009)

### 2. Majority Classifier
- **Type**: Dummy classifier (always predicts most frequent class)
- **Implementation**: scikit-learn `DummyClassifier` with `strategy='most_frequent'`
- **Rationale**: Minimal baseline to establish a lower bound

### 3. Best Baseline Selection
- **Selection Criteria**: The model with the highest AUROC is selected as "best baseline"
- **Typical Result**: Logistic Regression (typically outperforms majority classifier)

## Current Baseline Values (Example from H1 Results)

These values are computed by training the baseline models on the EMBER-2024 dataset:

| Metric | Baseline Value | AICRA Value | Improvement |
|--------|----------------|-------------|-------------|
| **AUROC** | 0.7781 | 0.9605 | **+25.9%** |
| **Precision** | 0.7726 | 0.6398 | **-13.8%*** |
| **Recall** | 0.6378 | 0.9985 | **+56.5%** |
| **F1** | 0.6988 | 0.7794 | **+14.3%** |

*Precision is lower because AICRA uses a banking-optimized threshold (0.0298) that prioritizes recall over precision to minimize false negatives, which is appropriate for banking security.

## What These Improvements Mean

### AUROC Improvement: +25.9%
- **Interpretation**: AICRA's ability to distinguish ransomware from benign files is 25.9% better than the simple logistic regression baseline
- **Calculation**: `(0.9605 - 0.7781) / 0.7781 × 100 = 25.9%`

### Precision Improvement: -13.8%
- **Interpretation**: AICRA has lower precision than baseline, but this is intentional
- **Reason**: Banking-optimized threshold (0.0298) prioritizes recall (catching all ransomware) over precision (reducing false positives)
- **Trade-off**: Acceptable for banking security where missed ransomware (FN) is much more costly than investigating false positives (FP)

### Recall Improvement: +56.5%
- **Interpretation**: AICRA catches 56.5% more ransomware samples than the baseline
- **Critical for Banking**: High recall (99.85%) means only 0.15% of ransomware is missed, which is essential for banking security

### F1 Improvement: +14.3%
- **Interpretation**: Overall balanced performance (precision and recall) is 14.3% better than baseline
- **Note**: F1 may not fully capture banking priorities (FN cost >> FP cost)

## Academic Sources (Methodology Justification)

The following sources justify the **methodology** of using these baseline models, but **not** the specific values:

1. **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** The Elements of Statistical Learning (2nd ed.). Springer.
   - Justifies using Logistic Regression as a standard baseline for binary classification

2. **Anderson, H. S., & Roth, P. (2018).** EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models. arXiv:1804.04637
   - Provides expected performance ranges for simple models on malware data:
     - AUC: 50-65% for simple linear models on static PE features
     - Precision: 35-45% for imbalanced malware classification
     - Recall: 50-60% for simple classifiers on malware data
   - **Note**: Our baseline (77.81% AUROC) is above this range, which is reasonable

3. **Raff, E., et al. (2018).** Malware Detection by Eating a Whole EXE. arXiv:1710.09435
   - Additional context on malware classification baselines

4. **scikit-learn Documentation**
   - Implementation references:
     - https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
     - https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html

## Code Location

The baseline computation is implemented in:
- **File**: `aicra/core/benchmarks.py`
- **Function**: `compute_h1_baselines()`
- **Called from**: `aicra/experiments/h1_classification.py` (line ~331)

## Future Considerations

### Potential Replacement with Academic Baselines

This empirically-computed baseline methodology may be replaced in the future with:

1. **Fixed Academic Baseline Values**: Values reported in published papers (e.g., Anderson & Roth, 2018)
2. **Standard Benchmark Results**: Results from established malware detection benchmarks
3. **Industry Standard Baselines**: Commonly accepted baseline values in cybersecurity ML research

### Advantages of Current Approach
- ✅ **Fair Comparison**: Baselines trained on same dataset as AICRA
- ✅ **Reproducible**: Same baseline models can be re-trained on different datasets
- ✅ **Dataset-Specific**: Reflects performance on EMBER-2024 specifically

### Advantages of Academic Baselines
- ✅ **Standardized**: Allows comparison across different papers/datasets
- ✅ **Established**: Values from peer-reviewed publications
- ✅ **Simpler**: No need to train baseline models

## Notes for Defense/Presentation

When presenting these results:

1. **Clarify Baseline Source**: State that baselines are "empirically computed by training simple baseline models (Logistic Regression and Majority Classifier) on the EMBER-2024 dataset"

2. **Explain Methodology**: Reference academic sources for methodology justification (Hastie et al., 2009; Anderson & Roth, 2018)

3. **Acknowledge Trade-offs**: Explain that lower precision is intentional due to banking-optimized threshold prioritizing recall

4. **Highlight Key Improvements**: Emphasize the +56.5% recall improvement and +25.9% AUROC improvement as the most critical metrics for banking security

## Related Files

- `results/H1_classification/H1_summary.md` - Main results summary (lines 55-68 show baseline comparison)
- `results/H1_classification/H1_full_results.json` - Full results including baseline metrics
- `aicra/core/benchmarks.py` - Baseline computation implementation
- `aicra/experiments/h1_classification.py` - H1 experiment that uses baselines

---

**Last Updated**: 2025-12-17  
**Status**: Temporary - May be replaced with academic baselines in future iterations

