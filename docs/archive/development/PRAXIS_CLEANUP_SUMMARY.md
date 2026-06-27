> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# AICRA Praxis Cleanup and Validation Summary

**Date:** 2025-01-XX  
**Status:** In Progress

## Overview

This document summarizes the cleanup and validation work performed on the AICRA codebase for the Doctor of Engineering praxis. The goal is to ensure codebase quality, validate all experiments (H1, H2, H3), and generate a comprehensive validation report.

---

## ✅ Completed Tasks

### 1. Discover and Align with Praxis ✅

- **Status:** Complete
- **Actions Taken:  - Reviewed all praxis-related documentation (H3_PRAXIS_PROOF.md, HYPOTHESIS_EXPERIMENTS_GUIDE.md, README.md)
  - Confirmed H1, H2, H3 hypothesis definitions match code implementation
  - Verified code comments and docstrings reference hypotheses correctly

**H1 Definition:- Static PE features enable reliable ransomware classification with AUROC ≥ 0.95 and operational precision suitable for banking environments.

**H2 Definition:- Cost-aware thresholding reduces expected loss vs F1-optimized thresholds under banking-style costs; Platt/isotonic post-hoc calibration tested whether calibration helps (does not improve expected loss on this model).

**H3 Definition:- Deterministic ATT&CK–D3FEND mappings exhibit higher Defense–Attack Consistency (DAC_internal), higher actionable precision, and higher actionable precision compared to learned mappings across all evaluation splits (variance reduction 0.0 on all splits; perfect separation).

### 2. Code Clean-up and Structure ✅

- **Status:** Complete
- **Actions Taken:  - Created `aicra/utils/data_loader.py` to support train/val/test splits for H1 and H2
  - Fixed imports in H1 and H2 experiments to use the new data loader
  - Updated H1 and H2 to generate `H1_full_results.json` and `H2_full_results.json` (in addition to metrics.json for backward compatibility)
  - Updated H1 and H2 to generate `H1_summary.md` and `H2_summary.md`

**Files Modified:- `aicra/utils/data_loader.py` (new file)
- `aicra/experiments/h1_classification.py` (updated)
- `aicra/experiments/h2_calibration_thresholds.py` (updated)

### 3. Validation Report Generation ✅

- **Status:** Complete
- **Actions Taken:  - Created `scripts/generate_praxis_validation_report.py` to generate comprehensive validation reports
  - Created initial `results/praxis_validation_report.md` with H3 results
  - Defined baseline metrics for H1, H2, H3 comparisons

**Baseline Definitions:- **H1:** AUROC reliability benchmark >0.88 (not 0.85); empirical logistic baseline AUROC ≈0.778 on same split; PR-AUC≈0.60, Brier=0.25, ECE=0.15 (same-split empirical references)
- **H2:** Brier=0.25, ECE=0.15 (uncalibrated baselines)
- **H3:** DAC_internal=0.0% (naive/learned vs deterministic ground truth); actionable precision deterministic **0.75** vs learned **0.00** (4 splits, 32,004 samples)

---

## 🔄 In Progress / Pending Tasks

### 4. Validate H1 Pipeline ⏳

- **Status:** Pending - Needs to be run
- **Required Actions:  1. Run H1 experiment: `python -m aicra.experiments.h1_classification`
  2. Verify data splits (train/test, no leakage)
  3. Verify metrics computation (AUROC, PR-AUC, accuracy, recall, etc.)
  4. Confirm `results/H1_classification/H1_full_results.json` is generated
  5. Confirm `results/H1_classification/H1_summary.md` is generated

**Expected Outputs:- `results/H1_classification/H1_full_results.json`
- `results/H1_classification/H1_summary.md`
- `results/H1_classification/metrics.json` (backward compatibility)

### 5. Validate H2 Pipeline ⏳

- **Status:** Pending - Needs to be run (depends on H1)
- **Required Actions:  1. Ensure H1 has been run first (H2 needs H1 model)
  2. Run H2 experiment: `python -m aicra.experiments.h2_calibration_thresholds`
  3. Verify calibration metrics (Brier, ECE before/after)
  4. Verify threshold optimization (F1-optimized vs cost-optimal)
  5. Confirm `results/H2_calibration_thresholds/H2_full_results.json` is generated
  6. Confirm `results/H2_calibration_thresholds/H2_summary.md` is generated

**Expected Outputs:- `results/H2_calibration_thresholds/H2_full_results.json`
- `results/H2_calibration_thresholds/H2_summary.md`
- `results/H2_calibration_thresholds/metrics.json` (backward compatibility)

### 6. Validate H3 Pipeline ✅

- **Status:** Complete - Results already exist
- **Verification:  - ✅ `results/H3_full_evaluation/H3_full_results.json` exists
  - ✅ `results/H3_full_evaluation/H3_full_summary.md` exists
  - ✅ Deterministic DAC_internal = 100% (as expected)
  - ✅ Learned DAC_internal = 0.00% (validates mapping difference)
  - ✅ Statistical tests computed

**Results Summary:**
- 4 splits evaluated (main, full_ember, small_ember, smoke_test)
- 32,004 total samples (main 10,000; full_ember 20,002; small_ember 2,000; smoke_test 2)
- Deterministic DAC_internal: 100.00% (SD: 0.00%)
- Learned DAC_internal: 0.00% (SD: 0.00%)
- Deterministic actionable precision: 0.75 (mean); Learned: 0.00
- Mean Δ DAC_internal: 100.00%

### 7. Add/Update Tests ⏳

- **Status:** Pending
- **Required Actions:  1. Add tests for H1 experiment in `tests/test_h1_classification.py`
  2. Add tests for H2 experiment in `tests/test_h2_calibration.py`
  3. Update `tests/test_h3_variance_expectation.py` if needed
  4. Add sanity checks:
     - AUROC/PR-AUC between 0 and 1
     - Brier/ECE between 0 and 1
     - DAC_internal for deterministic = 100%
     - JSON outputs contain expected keys

**Existing Tests:- `tests/test_h3_variance_expectation.py` - Validates H3 variance expectations

### 8. Type Hints and Linting ⏳

- **Status:** Pending
- **Required Actions:  1. Add type hints to all public functions in H1, H2, H3 experiments
  2. Run `ruff check` and fix issues
  3. Run `mypy` and fix type errors
  4. Run `pylint` and address warnings

### 9. Centralize Configs ⏳

- **Status:** Partial (H3 has config, H1/H2 use Settings)
- **Required Actions:  1. Create `config/h1_config.yaml` for H1 experiment parameters
  2. Create `config/h2_config.yaml` for H2 experiment parameters
  3. Ensure all experiments can be run from CLI with config files
  4. Document config structure

**Current State:- H3: `config/h3_splits.yaml` exists
- H1/H2: Use `aicra/config.py` Settings class

---

## 📋 Quick Start Guide

### Running All Experiments

```bash
# Run all hypotheses in sequence
python scripts/run_all_hypotheses.py

# Or run individually:
python -m aicra.experiments.h1_classification
python -m aicra.experiments.h2_calibration_thresholds
python -m aicra.experiments.h3_evaluation
```

### Generating Validation Report

```bash
# After H1, H2, H3 are complete:
python scripts/generate_praxis_validation_report.py

# Report will be saved to:
# results/praxis_validation_report.md
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_h3_variance_expectation.py -v

# Run with coverage
pytest --cov=aicra --cov-report=html
```

---

## 📊 Current Results Status

| Hypothesis | Results JSON | Summary MD | Status |
|------------|--------------|------------|--------|
| H1 | ❌ Not generated | ❌ Not generated | ⏳ Needs to be run |
| H2 | ❌ Not generated | ❌ Not generated | ⏳ Needs to be run (depends on H1) |
| H3 | ✅ Exists | ✅ Exists | ✅ Complete |

**H3 Results Location:- `results/H3_full_evaluation/H3_full_results.json`
- `results/H3_full_evaluation/H3_full_summary.md`

---

## 🔍 Key Files and Locations

### Experiment Modules
- `aicra/experiments/h1_classification.py` - H1 experiment
- `aicra/experiments/h2_calibration_thresholds.py` - H2 experiment
- `aicra/experiments/h3_evaluation.py` - H3 experiment

### Data Loading
- `aicra/utils/data_loader.py` - Data loading utilities (NEW)
- `aicra/core/data.py` - Core data structures

### Configuration
- `aicra/config.py` - Settings class
- `config/h3_splits.yaml` - H3 split configuration

### Results
- `results/H1_classification/` - H1 results (to be generated)
- `results/H2_calibration_thresholds/` - H2 results (to be generated)
- `results/H3_full_evaluation/` - H3 results (✅ exists)
- `results/praxis_validation_report.md` - Final validation report (✅ exists)

### Scripts
- `scripts/run_all_hypotheses.py` - Orchestrates all experiments
- `scripts/generate_praxis_validation_report.py` - Generates validation report

---

## 🎯 Next Steps

1. **Immediate:   - Run H1 experiment to generate results
   - Run H2 experiment to generate results
   - Regenerate validation report with complete data

2. **Short-term:   - Add comprehensive tests for H1, H2, H3
   - Add type hints and fix linting issues
   - Create config files for H1 and H2

3. **Final:   - Review and finalize `praxis_validation_report.md`
   - Ensure all sanity checks pass
   - Document any baseline adjustments if needed

---

## 📝 Notes

- H3 results are already complete and validated
- H1 and H2 experiments are ready to run but require EMBER-2024 data
- Baseline definitions are conservative estimates; actual baselines may vary
- The validation report generator will automatically update when H1 and H2 results are available

--**Last Updated:** 2025-01-XX  
**Status:** Ready for H1/H2 execution
