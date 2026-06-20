# AICRA Experiments: Step-by-Step Reproduction Guide

> **Praxis hub:** [praxis/README.md](praxis/README.md) · **Canonical commands:** [praxis/EXPERIMENTS_GUIDE.md](praxis/EXPERIMENTS_GUIDE.md)

This document provides step-by-step instructions for reproducing all H1–H3 experiments.

## Prerequisites

1. **Data Setup**: Ensure EMBER-2024 dataset is available
   ```bash
   bash scripts/fetch_data.sh      # Linux/Mac
   .\scripts\fetch_data.ps1        # Windows
   ```

2. **Dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

## H1: Static PE Classification

```bash
python -m aicra.experiments.h1_classification \
  --output results/H1_classification \
  --model-type lgbm \
  --splits-config config/h1_splits.yaml
```

**Outputs:** `results/H1_classification/H1_full_results.json`, `H1_summary.md`

**Key metrics:** AUROC (primary), PR-AUC, Brier, ECE, banking-optimized precision/recall

## H2: Calibration & Cost-Aware Thresholding

Run after H1.

```bash
python -m aicra.experiments.h2_calibration_thresholds \
  --output results/H2_calibration_thresholds \
  --splits-config config/h2_splits.yaml
```

**Outputs:** `results/H2_calibration_thresholds/H2_full_results.json`, `H2_summary.md`

**Key metrics:** Expected loss (cost-optimal vs F1-optimal), Brier, ECE

## H3: Deterministic vs Learned Mapping

```bash
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml
```

**Outputs:** `results/H3_full_evaluation/H3_full_results.json`, `H3_full_summary.md`

**Key metrics:** DAC_internal, actionable precision, variance reduction

## Run all hypotheses

```bash
python scripts/run_all_hypotheses.py
```

## Supplementary evaluations

| Script | Output |
|--------|--------|
| `scripts/evaluate_h1_oof_robust.py` | `results/H1_oof_robust_eval/` |
| `scripts/generate_h3_mapping_comparison.py` | `results/H3_full_evaluation/mapping_comparison_*` |
| `scripts/generate_praxis_validation_report.py` | `results/praxis_validation_report.md` |

## Verification

```bash
pytest tests/
python scripts/compute_pvalues.py   # if configured for your results paths
```

See also: [BENCHMARK_NOTES.md](BENCHMARK_NOTES.md), [results/praxis_validation_report.md](../results/praxis_validation_report.md)
