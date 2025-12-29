# AICRA Experiments: Step-by-Step Reproduction Guide

This document provides step-by-step instructions for reproducing all H1-H3 experiments.

## Prerequisites

1. **Data Setup**: Ensure EMBER-2024 dataset is available
   ```bash
   # Check data availability
   bash scripts/fetch_data.sh  # Linux/Mac
   .\scripts\fetch_data.ps1    # Windows
   
   # Or set environment variable
   export AICRA_EMBER2024_DIR=/path/to/ember2024_real  # Linux/Mac
   $env:AICRA_EMBER2024_DIR = "C:\path\to\ember2024_real"  # Windows
   ```

2. **Dependencies**: Install all requirements
   ```bash
   pip install -r requirements-dev.txt
   ```

## H1: Static PE Classification Reliability

**Hypothesis**: LightGBM on EMBER-2024 static PE features predicts ransomware susceptibility.

### Run H1 Experiment

```bash
# From repository root
python experiments/h1_train_eval.py
```

### Expected Outputs

- `artifacts/H1_classification/metrics.json` - All metrics including:
  - AUROC, Precision, Recall, F1
  - Baseline comparisons (logistic regression, majority classifier)
  - % improvements over baseline
  - Temporal test metrics
  - Out-of-family test metrics (if families available)
  - Confusion matrix at banking-optimized threshold

- `artifacts/H1_classification/summary.md` - Human-readable summary

### Key Metrics

- **AUROC**: Should be ≥ 0.95
- **Precision**: At banking-optimized threshold (FN cost >> FP cost)
- **Out-of-Family AUROC**: Performance on held-out malware families
- **Temporal Test**: Performance on chronologically later samples

### Verification

```bash
# Check that outputs exist
ls artifacts/H1_classification/

# View summary
cat artifacts/H1_classification/summary.md
```

---

## H2: Cost-Aware Thresholding

**Research Question (RQ2)**: Does cost-aware thresholding reduce expected loss compared to F1-optimized thresholds under banking-style asymmetric costs (FN cost >> FP cost)?

**Hypothesis (H2)**: Cost-aware thresholding produces lower expected loss than F1-optimized thresholds under banking-style asymmetric costs (FN cost >> FP cost), demonstrating more decision-aligned susceptibility scores for operational deployment.

### Prerequisites

H2 requires H1 to be run first (uses trained model from H1).

### Run H2 Experiment

```bash
# From repository root
python experiments/h2_calibration_eval.py
```

### Expected Outputs

- `artifacts/H2_calibration_thresholds/metrics.json` - All metrics including:
  - Brier score (pre/post calibration)
  - ECE (Expected Calibration Error) (pre/post calibration)
  - % improvements over uncalibrated baseline
  - Temporal calibration check (calibrate on earlier window, test on later)
  - Threshold comparisons (F1-optimized vs cost-optimal)

- `artifacts/H2_calibration_thresholds/summary.md` - Human-readable summary

### Key Metrics

- **Brier Score Improvement**: Reduction in Brier score after calibration
- **ECE Improvement**: Reduction in Expected Calibration Error
- **Temporal Calibration**: Calibration performance on later time window

### Verification

```bash
# Check that outputs exist
ls artifacts/H2_calibration_thresholds/

# View summary
cat artifacts/H2_calibration_thresholds/summary.md
```

---

## H3: Deterministic vs Learned Mapping Comparison

**Hypothesis**: Deterministic ATT&CK–D3FEND lookup beats learned mapping.

### Prerequisites

H3 requires:
- Pre-computed risk scores from H1/H2 (stored in `results/` directories)
- Deterministic mapping file: `data/mappings/deterministic_lookup.csv`
- Learned mapping file: `data/mappings/learned_mapping.csv`
- Reference pairs: `d3fend_reference_pairs.csv`
- H3 splits configuration: `config/h3_splits.yaml`

### Run H3 Experiment

```bash
# From repository root
python experiments/h3_mapping_compare.py
```

### Expected Outputs

- `artifacts/H3_full_evaluation/H3_full_results.json` - Complete results including:
  - Coverage % (deterministic vs learned)
  - Defense-Attack Consistency (DAC) %
  - Risk score variance reduction
  - Statistical tests (paired t-test, bootstrap CI)
  - Per-split metrics

- `artifacts/H3_full_evaluation/H3_full_summary.md` - Human-readable summary

- `artifacts/H3_full_evaluation/h3_mapping_integrity.json` - Mapping integrity check
  - Verifies learned mapping is NOT identical to deterministic mapping
  - Reports overlap percentage

### Key Metrics

- **Coverage Improvement**: % increase in technique-control mapping coverage
- **Variance Reduction**: % reduction in risk score variance
- **DAC (Defense-Attack Consistency)**: Agreement with deterministic mapping

### Verification

```bash
# Check that outputs exist
ls artifacts/H3_full_evaluation/

# View summary
cat artifacts/H3_full_evaluation/H3_full_summary.md

# Check mapping integrity
cat artifacts/H3_full_evaluation/h3_mapping_integrity.json
```

---

## Running All Experiments

To run all experiments in sequence:

```bash
# Run all experiments
python scripts/run_all_hypotheses.py

# Or run individually
python experiments/h1_train_eval.py
python experiments/h2_calibration_eval.py
python experiments/h3_mapping_compare.py
```

## Benchmark Improvements Report

After running experiments, a consolidated benchmark report is generated:

- `artifacts/benchmark_improvements.csv` - Machine-readable table
- `artifacts/benchmark_improvements.md` - Human-readable summary

To regenerate the report manually:

```bash
python -m aicra.utils.benchmark_reporter
```

## Reproducibility

All experiments:
- Use fixed random seeds (default: 42)
- Log git commit hash in metadata
- Save complete configuration in `experiment_metadata.json`
- Use time-ordered splits to prevent temporal leakage
- Document imbalanced data handling (class weights, pos_weight)

## Risk Register Artifacts by Dataset Scale

This repository includes risk register outputs at different dataset scales to demonstrate pipeline correctness and scalability:

### Small EMBER Risk Registers

Small EMBER risk registers are stored in the `register/` directory:
- `register/risk_register_small_ember.csv` and `.json` - Complete risk register for small EMBER subset
- `register/smoke_test_register.csv` and `.json` - Smoke test risk register
- These files are included in Git to demonstrate end-to-end correctness, structure, and reproducibility

### Full EMBER Derived Outputs

Full EMBER evaluations generate **derived artifacts only** (not raw data):
- Risk scores: `results/*/risk_scores.csv` - Calibrated risk scores per split
- Diagnostics: `results/*/diagnostics/` - Mapping metrics and statistical tests
- Mapping metrics: Coverage, DAC, variance reduction metrics in `results/H3_full_evaluation/`

**Raw EMBER JSONL files are excluded by design** due to:
- Size constraints (~30GB dataset)
- Licensing considerations
- Repository hygiene best practices

See `docs/DATA.md` for data availability and exclusion rationale.

These artifacts support H1–H3 evaluation without requiring raw data in Git, enabling reviewers to verify pipeline correctness and scalability while maintaining a manageable repository size.

## Troubleshooting

### Data Not Found

If you see "EMBER-2024 directory not found":
1. Run `scripts/fetch_data.sh` or `scripts/fetch_data.ps1`
2. Set `AICRA_EMBER2024_DIR` environment variable
3. See `docs/DATA.md` for details

### H2 Fails (Model Not Found)

If H2 fails with "Model not found":
1. Run H1 experiment first: `python experiments/h1_train_eval.py`
2. Verify model exists: `ls artifacts/models/h1_lgbm.joblib`

### H3 Fails (Mapping Integrity Check)

If H3 fails with "Learned mapping is identical to deterministic mapping":
1. This indicates a bug in mapping generation
2. Regenerate learned mapping using embedding-based heuristics
3. Verify learned mapping is different from deterministic mapping

### Missing Risk Scores for H3

If H3 fails with "Risk scores file not found":
1. Ensure H1/H2 have been run and generated risk scores
2. Check `config/h3_splits.yaml` for correct paths
3. Verify risk score files exist in `results/` directories

## Additional Resources

- **Data Management**: See `docs/DATA.md`
- **Configuration**: See `aicra/config.py` and experiment-specific config files
- **Benchmark Sources**: See `aicra/core/benchmarks.py` for baseline methodology

