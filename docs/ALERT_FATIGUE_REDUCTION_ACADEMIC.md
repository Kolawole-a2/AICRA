# Alert Fatigue Reduction (Empirical FN-Rate Comparison)

## Summary

Alert fatigue reduction in AICRA is measured as **false-negative (FN) rate reduction** vs the **empirical best baseline** (logistic regression on the same held-out EMBER-2024 test split).

## Current H1 Numbers (full test set)

| Metric | Empirical baseline (logistic regression) | AICRA |
|--------|------------------------------------------|-------|
| Recall | 63.63% | 99.85% |
| FN rate | **36.4%** (~1670 / 4592 positives) | **0.15%** (7 / 4592) |
| FN rate reduction | — | **~99.6%** |

Formula:

```
FN rate reduction = (baseline_fn_rate - aicra_fn_rate) / baseline_fn_rate
```

## Banking context

- FN cost >> FP cost in banking SOC workflows.
- H1 uses a banking-optimized threshold (0.0248) that prioritizes recall.
- High FP volume is operationally manageable; missed ransomware drives incident cost and analyst stress.

## Implementation

- `aicra/core/benchmarks.py` — `compute_h1_improvements()`
- `results/H1_classification/H1_summary.md` — reported metrics
- `results/H1_classification/temporal_split_verification.json` — time-ordered split integrity
