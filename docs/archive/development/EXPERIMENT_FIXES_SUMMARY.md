> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# Experiment Fixes Summary

This document summarizes the fixes applied to ensure H1, H2, and H3 experiments are correct, non-duplicated, and scientifically meaningful.

## Issues Fixed

### 1. H3: Reference Pairs Identical to Deterministic Mapping

**Problem:- `d3fend_reference_pairs.csv` had the same SHA256 hash as `deterministic_lookup.csv`
- This meant the reference pairs were pointing to the deterministic mapping instead of the canonical ATT&CK-D3FEND reference
- H3 results showed identical metrics for deterministic and learned mappings

**Solution:- Created `scripts/create_reference_pairs.py` to generate canonical reference pairs from `data/lookups/attack_to_d3fend.yaml`
- The canonical reference now contains 15 pairs (5 techniques × 3 controls each) from the authoritative YAML mapping
- Updated `d3fend_reference_pairs.csv` with the correct canonical reference

### 2. H3: Missing Sanity Checks

**Problem:- No validation that reference_pairs ≠ deterministic_mapping
- No validation that learned_mapping ≠ deterministic_mapping
- Experiments could run with invalid configurations

**Solution:- Added three sanity checks in `aicra/experiments/h3_evaluation.py`:
  1. **Reference vs Deterministic**: Checks file hashes and raises RuntimeError if identical
  2. **Reference vs Deterministic (set comparison)**: Warns if pair sets are identical
  3. **Deterministic vs Learned**: Raises RuntimeError if mappings are identical
- All checks provide clear error messages with instructions on how to fix

### 3. H1: Missing Canonical Experiment File

**Problem:- No dedicated H1 experiment file for static PE classification
- Training and evaluation were scattered across multiple files

**Solution:- Created `aicra/experiments/h1_classification.py`:
  - Canonical H1 experiment module
  - Evaluates static PE classification on EMBER-2024
  - Computes all required metrics: AUROC, PR-AUC, Precision, Recall, F1, Brier, ECE, Lift@k
  - Supports out-of-family generalization evaluation
  - Saves results to `results/H1_classification/`

### 4. H2: Missing Canonical Experiment File

**Problem:- No dedicated H2 experiment file for calibration and thresholding
- Calibration and threshold optimization were not integrated

**Solution:- Created `aicra/experiments/h2_calibration_thresholds.py`:
  - Canonical H2 experiment module
  - Evaluates calibration (Platt/Isotonic) with Brier and ECE metrics
  - Compares F1-optimized vs cost-optimal thresholds
  - Computes Expected Loss at different thresholds
  - Saves results to `results/H2_calibration_thresholds/`

### 5. Missing Orchestration Script

**Problem:- No single script to run all hypothesis experiments
- Difficult to reproduce complete experimental pipeline

**Solution:- Created `scripts/run_all_hypotheses.py`:
  - Runs H1, H2, H3 experiments in order
  - Validates inputs and dependencies
  - Provides summary of all results
  - Supports skipping individual experiments

## Files Created/Modified

### New Files
1. `scripts/create_reference_pairs.py` - Creates canonical reference pairs from YAML
2. `aicra/experiments/h1_classification.py` - Canonical H1 experiment
3. `aicra/experiments/h2_calibration_thresholds.py` - Canonical H2 experiment
4. `scripts/run_all_hypotheses.py` - Orchestration script for all experiments

### Modified Files
1. `aicra/experiments/h3_evaluation.py` - Added sanity checks for mapping validation
2. `d3fend_reference_pairs.csv` - Regenerated with canonical reference (15 pairs)

## How to Run Experiments

### Individual Experiments

**H1: Static PE Classification```bash
python -m aicra.experiments.h1_classification
# Or with custom options:
python -m aicra.experiments.h1_classification --model-type lgbm --threshold 0.5
```

**H2: Calibration test and Thresholding```bash
python -m aicra.experiments.h2_calibration_thresholds
# Or with custom options:
python -m aicra.experiments.h2_calibration_thresholds --cost-fn 10.0 --cost-fp 1.0
```

**H3: Mapping Comparison```bash
python run_h3_evaluation.py
# Or:
python -m aicra.experiments.h3_evaluation --config config/h3_splits.yaml
```

### All Experiments
```bash
python scripts/run_all_hypotheses.py
# Or skip specific experiments:
python scripts/run_all_hypotheses.py --skip-h1
```

## Results Locations

- **H1**: `results/H1_classification/`
  - `metrics.json` - All metrics
  - `summary.md` - Human-readable summary

- **H2**: `results/H2_calibration_thresholds/`
  - `metrics.json` - All metrics
  - `summary.md` - Human-readable summary

- **H3**: `results/H3_full_evaluation/`
  - `H3_full_results.json` - Complete results with per-split and aggregated metrics
  - `H3_full_summary.md` - Comprehensive markdown report
  - `plots/` - Visualization plots

## Validation Checks

The H3 experiment now includes automatic validation:

1. **Reference Pairs Check**: Ensures `d3fend_reference_pairs.csv` is not identical to `deterministic_lookup.csv`
2. **Mapping Difference Check**: Ensures `learned_mapping.csv` is not identical to `deterministic_lookup.csv`
3. **File Hash Verification**: Computes SHA256 hashes for reproducibility

If any check fails, the experiment will raise a `RuntimeError` with clear instructions on how to fix the issue.

## Next Steps

1. **Regenerate Learned Mapping** (if needed):
   ```bash
   python generate_learned_mapping.py
   ```
   This ensures the learned mapping is different from deterministic.

2. **Run All Experiments**:
   ```bash
   python scripts/run_all_hypotheses.py
   ```

3. **Verify Results**:
   - Check that H3 metrics show differences between deterministic and learned mappings
   - Verify that reference_pairs hash ≠ deterministic_lookup hash
   - Confirm all sanity checks pass

## Notes

- The canonical reference pairs (15 pairs) come from `data/lookups/attack_to_d3fend.yaml`
- The deterministic mapping (175 pairs) comes from a different source and is more comprehensive
- The learned mapping should be generated using embedding-based similarity, not copied from deterministic
- All experiments use type hints and avoid `allow_pickle=True` for security
- Base datasets (EMBER/SOREL) are never overwritten
