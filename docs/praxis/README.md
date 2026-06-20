# AICRA Praxis Documentation

Central index for the Doctor of Engineering praxis (production) by **Kolawole Afolabi**: *Machine Learning-Based Cyber Risk Advisor with Analytics for Endpoint Ransomware Defense in U.S. Banking Organizations*.

**Contact:** [kolawole.afolabi@gwmail.gwu.edu](mailto:kolawole.afolabi@gwmail.gwu.edu) · [ako.afolabi@gmail.com](mailto:ako.afolabi@gmail.com)

## Start here

| Document | Purpose |
|----------|---------|
| [../../README.md](../../README.md) | Repository overview and quick-start commands |
| [EXPERIMENTS_GUIDE.md](EXPERIMENTS_GUIDE.md) | How to run H1, H2, H3 (canonical paths) |
| [../EXPERIMENTS.md](../EXPERIMENTS.md) | Step-by-step reproduction |
| [../BENCHMARK_NOTES.md](../BENCHMARK_NOTES.md) | Current metric snapshot from saved artifacts |
| [../RESULTS_SUMMARY.md](../RESULTS_SUMMARY.md) | Research-ready results tables |
| [../../results/praxis_validation_report.md](../../results/praxis_validation_report.md) | Consolidated validation across H1–H3 |

## Hypothesis artifacts (canonical — do not overwrite casually)

| Hypothesis | Results | Summary |
|------------|---------|---------|
| **H1** | `results/H1_classification/H1_full_results.json` | `results/H1_classification/H1_summary.md` |
| **H1 OOF (supplementary)** | `results/H1_oof_robust_eval/oof_robust_metrics.json` | `results/H1_oof_robust_eval/oof_robust_summary.md` |
| **H2** | `results/H2_calibration_thresholds/H2_full_results.json` | `results/H2_calibration_thresholds/H2_summary.md` |
| **H3** | `results/H3_full_evaluation/H3_full_results.json` | `results/H3_full_evaluation/H3_full_summary.md` |

## Statistical validation

| Document | Purpose |
|----------|---------|
| [../HYPOTHESIS_TESTING_PVALUES.md](../HYPOTHESIS_TESTING_PVALUES.md) | P-value methodology (H1 benchmark **> 0.88**; H2 calibration **help test**; H3 perfect separation when variance is zero) |
| [../h3_dac_statistical_validation.md](../h3_dac_statistical_validation.md) | H3 DAC statistical tests |
| `results/pvalues_summary.json` | Computed p-values (if present) |

## Design & validity

| Document | Purpose |
|----------|---------|
| [../THREATS_TO_VALIDITY.md](../THREATS_TO_VALIDITY.md) | Validity threats and mitigations |
| [../PRECISION_RECALL_TRADE_OFF_BANKING.md](../PRECISION_RECALL_TRADE_OFF_BANKING.md) | Banking threshold trade-off |
| [../BASELINE_METHODOLOGY_TEMP.md](../BASELINE_METHODOLOGY_TEMP.md) | Empirical baseline methodology |
| [../ALERT_FATIGUE_REDUCTION_ACADEMIC.md](../ALERT_FATIGUE_REDUCTION_ACADEMIC.md) | FN-rate / alert-fatigue framing |
| [../CANONICAL_VS_REBUILD_EXPLANATION.md](../CANONICAL_VS_REBUILD_EXPLANATION.md) | Canonical vs optional rebuild pipeline |
| [../adversarial_limitations.md](../adversarial_limitations.md) | Robustness limitations |

## Reviewer / defense

| Document | Purpose |
|----------|---------|
| [../REVIEWER_GUIDE.md](../REVIEWER_GUIDE.md) | Navigation and reproduction for reviewers |
| [../FINAL_AUDIT_CHECKLIST.md](../FINAL_AUDIT_CHECKLIST.md) | Pre-defense checklist |
| [PROJECT_LAYOUT.md](PROJECT_LAYOUT.md) | Repository layout explained |

## Code entry points

```bash
# H1 — static PE classification (multi-split)
python -m aicra.experiments.h1_classification --splits-config config/h1_splits.yaml

# H2 — post-hoc calibration test (Platt/isotonic) & cost-aware thresholds (requires H1)
python -m aicra.experiments.h2_calibration_thresholds --splits-config config/h2_splits.yaml

# H3 — deterministic vs learned mapping
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml

# All three
python scripts/run_all_hypotheses.py
```

## Archive

Development notes, fix summaries, and superseded reports live under `docs/archive/development/`. They are kept for traceability but are **not** part of the praxis narrative.
