> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# Hypothesis Experiments Guide

This guide explains how to run and validate all hypothesis experiments (H1, H2, H3) in the AICRA praxis repository.

## Overview

The repository now has canonical experiment files for all three hypotheses:

- **H1**: Static PE Classification Reliability (`aicra/experiments/h1_classification.py`)
- **H2**: Post-Hoc Calibration Test & Cost-Aware Thresholding (`aicra/experiments/h2_calibration_thresholds.py`)
- **H3**: Deterministic vs Learned Mapping Comparison (`aicra/experiments/h3_evaluation.py`)

## Prerequisites

1. **Data Files**:
   - EMBER-2024 dataset in `data/ember2024_real/`
   - Deterministic mapping: `data/mappings/deterministic_lookup.csv`
   - Learned mapping: `data/mappings/learned_mapping.csv` (will be validated)
   - Reference pairs: `d3fend_reference_pairs.csv` (created from YAML)

2. **Configuration**:
   - H3 splits config: `config/h3_splits.yaml`

## Quick Start

### Run All Experiments
```bash
python scripts/run_all_hypotheses.py
```

This will:
1. Run H1 classification experiment
2. Run H2 post-hoc calibration test and cost-aware thresholding experiment
3. Run H3 mapping comparison experiment
4. Print summary of all results

### Run Individual Experiments

**H1: Classification```bash
python -m aicra.experiments.h1_classification
```

**H2: Calibration test```bash
python -m aicra.experiments.h2_calibration_thresholds
```

**H3: Mapping Comparison```bash
python run_h3_evaluation.py
```

## H1: Static PE Classification Reliability

**Hypothesis**: Static PE features enable reliable ransomware classification with AUROC ≥ 0.95 and operational precision suitable for banking environments.

**What it does**:
- Trains LightGBM model on EMBER-2024 with static PE features
- Evaluates on time-ordered train/test split, multi-split slices, and supplementary OOF (scripts/evaluate_h1_oof_robust.py)
- Computes AUROC, PR-AUC, Precision, Recall, F1, Brier, ECE, Lift@k
- Multi-split evaluation plus supplementary out-of-family generalization (OOF AUROC 0.9615)

**Results Location**: `results/H1_classification/`
- `metrics.json` - All computed metrics
- `summary.md` - Human-readable summary

**Key Metrics**:
- AUROC (reliability benchmark **> 0.88**; design target ≥ 0.95; empirical logistic baseline ≈ 0.778)
- PR-AUC
- Operational Precision/Recall/F1 at threshold
- Brier Score, ECE
- Lift@1%, Lift@5%, Lift@10%

## H2: Post-Hoc Calibration Test & Cost-Aware Thresholding

**Hypothesis**: Cost-aware thresholding reduces expected loss vs F1-optimized thresholds under banking-style costs; Platt/isotonic post-hoc calibration tested whether calibration helps (does not improve expected loss on this model).

**What it does**:
- Loads model from H1
- Calibrates predictions using Platt/Isotonic regression
- Compares F1-optimized vs cost-optimal thresholds
- Computes Expected Loss at different thresholds
- Tests whether post-hoc calibration helps (Brier, ECE, expected loss); primary H2 metric is expected loss

**Results Location**: `results/H2_calibration_thresholds/`
- `metrics.json` - All computed metrics
- `summary.md` - Human-readable summary

**Key Metrics**:
- Brier Score (before/after calibration)
- ECE (before/after calibration)
- F1-optimized threshold and metrics
- Cost-optimal threshold and Expected Loss
- Comparison of calibrated vs uncalibrated

## H3: Deterministic vs Learned Mapping Comparison

**Hypothesis**: Deterministic mapping achieves higher DAC_internal and actionable precision than learned mapping across all splits (variance reduction 0.0 on all splits; perfect separation).

**What it does**:
- Compares deterministic mapping vs learned mapping across evaluation splits
- Computes mapping metrics: Coverage, DAC, Correctness
- Computes register-level metrics: Actionable Precision, Variance Reduction
- Runs statistical tests for DAC and precision (variance tests not applicable when variance reduction is identically 0.0)
- Generates plots and comprehensive report

**Results Location**: `results/H3_full_evaluation/`
- `H3_full_results.json` - Complete results with per-split and aggregated metrics
- `H3_full_summary.md` - Comprehensive markdown report
- `plots/` - Visualization plots

**Key Metrics**:
- Coverage % (techniques with mapped controls)
- DAC % (Defense-Attack Consistency)
- Actionable Precision & F1
- Variance/IQR Reduction (0.0 on all splits for both mappings; not used for H3 validation)
- Delta metrics (Deterministic - Learned)
- Statistical tests (p-values)

## Validation and Sanity Checks

The H3 experiment includes automatic validation that will **fail fast** if configurations are invalid:

### 1. Reference Pairs Validation
- **Check**: Ensures `d3fend_reference_pairs.csv` ≠ `deterministic_lookup.csv`
- **Error**: Raises `RuntimeError` if file hashes are identical
- **Fix**: Run `python scripts/create_reference_pairs.py`

### 2. Mapping Difference Validation
- **Check**: Ensures `learned_mapping.csv` ≠ `deterministic_lookup.csv`
- **Error**: Raises `RuntimeError` if pair sets are identical
- **Fix**: Regenerate learned mapping using `python generate_learned_mapping.py`

### 3. File Hash Verification
- All mapping files have SHA256 hashes computed and logged
- Hashes are saved in results for reproducibility

## Troubleshooting

### H3: "Reference pairs file is identical to deterministic mapping"
**Solution**: 
```bash
python scripts/create_reference_pairs.py
```

### H3: "Deterministic and learned mappings are IDENTICAL"
**Solution**: 
```bash
python generate_learned_mapping.py
```

### H1: "Model not found"
**Solution**: H1 will train a new model. If you want to use an existing model, ensure it's at `models/h1_lgbm.joblib` or update the path in the code.

### H2: "Model not found"
**Solution**: Run H1 first to generate the model, or update the model path in H2 code.

### Missing Risk Scores for H3
**Solution**: Ensure `config/h3_splits.yaml` points to valid risk scores CSV files. Each file must have columns: `asset_id`, `risk_score`, `predicted_label`, `true_label`, `technique_id`.

## File Structure

```
aicra/experiments/
├── h1_classification.py          # Canonical H1 experiment
├── h2_calibration_thresholds.py   # Canonical H2 experiment
├── h3_evaluation.py               # Canonical H3 experiment
├── h3_learned_mapping_eval.py     # Helper (may be obsolete)
├── h3_prepare_metrics.py          # Helper (may be obsolete)
└── h3_stat_tests.py               # Helper (may be obsolete)

scripts/
├── create_reference_pairs.py      # Creates canonical reference pairs
└── run_all_hypotheses.py          # Orchestrates all experiments

results/
├── H1_classification/            # H1 results
├── H2_calibration_thresholds/     # H2 results
└── H3_full_evaluation/           # H3 results
```

## Expected Results

### H1
- AUROC should be ≥ 0.95 (hypothesis support)
- PR-AUC should be ≥ 0.85
- Brier Score should be < 0.15
- ECE should be < 0.05

### H2
- Brier Score should improve after calibration
- ECE should improve after calibration
- Cost-optimal threshold should have lower Expected Loss than F1-optimized

### H3
- Deterministic and learned mappings should produce **different** metrics
- Delta metrics should be non-zero
- Statistical tests should be computable (p-values may or may not be significant)

## Reproducibility

All experiments:
- Use fixed random seeds where applicable
- Log file hashes (SHA256) for input files
- Save complete metrics to JSON
- Generate human-readable summaries
- Include command-line arguments in logs

## Next Steps After Running

1. **Review Results**: Check all `summary.md` files for human-readable summaries
2. **Validate H3**: Ensure deterministic and learned metrics are different
3. **Check Sanity**: Verify all sanity checks passed (no RuntimeErrors)
4. **Update Documentation**: If results differ from expectations, update methodology docs

## Support

For issues or questions:
1. Check `EXPERIMENT_FIXES_SUMMARY.md` for known issues and fixes
2. Review error messages - they include instructions on how to fix
3. Check file hashes in results to verify inputs haven't changed
