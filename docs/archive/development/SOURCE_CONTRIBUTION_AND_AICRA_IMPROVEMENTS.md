> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# Deprecated

This file previously listed academic source contributions and literature-based baseline percentages. AICRA now reports **empirical baselines only** (same EMBER-2024 splits and artifacts):

- **H1:** logistic regression + majority classifier vs LightGBM — see `results/H1_classification/H1_summary.md`
- **H2:** F1-optimal vs cost-optimal thresholding on H1 probabilities — see `results/H2_calibration_thresholds/H2_summary.md`
- **H3:** learned vs deterministic mapping — see `results/H3_full_evaluation/H3_full_summary.md`

See `README.md` and `docs/BENCHMARK_NOTES.md` for current methodology.
