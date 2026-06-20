# Deprecated

Alert-fatigue / FN-rate comparisons now use the **empirical best baseline** (logistic regression on the same held-out test split), not literature-derived FN rates.

Implementation: `aicra/core/benchmarks.py` (`compute_h1_improvements`) and `results/H1_classification/H1_summary.md`.
