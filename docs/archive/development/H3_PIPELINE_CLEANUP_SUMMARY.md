> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# H3 Pipeline Cleanup and Unification Summary

## Overview

All previous, partial, or poor H3-related experiment code and outputs have been **OVERWRITTEN and CLEANED UP** and replaced with **ONE clean, consistent H3 experiment pipeline**.

## What Was Removed

### Old Scripts (Root Directory)
- ✅ `run_h3_experiment.py` - Removed
- ✅ `run_h3_full_experiment.py` - Removed
- ✅ `run_h3_validation.py` - Removed
- ✅ `prepare_h3_inputs.py` - Removed
- ✅ `prepare_h3_validation_inputs.py` - Removed
- ✅ `reprocess_h3.py` - Removed
- ✅ `setup_and_run_h3.py` - Removed

### Old Result Directories
- ✅ `results/H3_comparison/` - Removed
- ✅ `results/H3_validation/` - Removed

## What Was Created

### New Canonical Pipeline

1. **`aicra/experiments/h3_evaluation.py`** (563 lines)
   - Canonical H3 evaluation module
   - Compares deterministic vs learned mappings across all splits
   - Computes: DAC, Coverage, Precision, Variance Reduction
   - Handles column name variations automatically
   - Generates comprehensive reports

2. **`run_h3_evaluation.py`** (Main Entry Point)
   - Simple, clean entry point script
   - Validates all inputs before running
   - Provides clear error messages

3. **`config/h3_splits.yaml`** (Configuration)
   - Centralized configuration for all evaluation splits
   - Easy to add/remove splits
   - Paths relative to repo root

4. **`docs/h3_evaluation_README.md`** (Documentation)
   - Complete usage guide
   - Configuration instructions
   - Output format documentation

## What Was Preserved

### Protected Files (NOT Modified)
- ✅ `data/mappings/deterministic_lookup.csv` - Preserved
- ✅ `data/mappings/learned_mapping.csv` - Preserved
- ✅ `data/ontology/...` source files - Preserved

### Supporting Modules (Kept for Reference)
- `aicra/experiments/h3_stat_tests.py` - Statistical tests (can be used for additional analysis)
- `aicra/experiments/h3_prepare_metrics.py` - Metrics preparation (legacy, not required)
- `aicra/experiments/h3_learned_mapping_eval.py` - Learned mapping eval (legacy, not required)
- `aicra/analysis/h3_dac_stats.py` - Statistical analysis (can be used for additional analysis)

## New Output Structure

All H3 outputs are now saved to:
```
results/H3_full_evaluation/
├── h3_results_by_split.csv      # Detailed metrics per split
├── h3_summary.json              # Summary statistics
└── h3_report.md                 # Human-readable report
```

## Usage

### Basic Usage
```bash
python run_h3_evaluation.py
```

### Custom Configuration
```bash
python run_h3_evaluation.py \
    --splits-config config/h3_splits.yaml \
    --deterministic data/mappings/deterministic_lookup.csv \
    --learned data/mappings/learned_mapping.csv \
    --reference d3fend_reference_pairs.csv \
    --output results/H3_full_evaluation
```

## Configuration

Edit `config/h3_splits.yaml` to add/remove evaluation splits:

```yaml
splits:
  time_test: "results/time_test/risk_scores.csv"
  oof_test: "results/oof_test/risk_scores.csv"
  seed1_time_test: "results/seed1/time_test/risk_scores.csv"
```

## Metrics Computed

For each split:
1. **DAC (Defense-Attack Consistency)**: Proportion of correctly aligned pairs
2. **Coverage**: % of techniques with mapped controls
3. **Actionable Precision**: Precision for actionable positives
4. **Variance Reduction**: Reduction in risk score variance
5. **Deltas**: Differences (deterministic - learned)

## Key Features

- ✅ **Automatic column name normalization** (technique_id ↔ attack_id, control_id ↔ defense_id)
- ✅ **Robust error handling** (skips missing splits with warnings)
- ✅ **Comprehensive reporting** (CSV, JSON, Markdown)
- ✅ **Clean, maintainable code** (single canonical module)
- ✅ **Well-documented** (docstrings, README, inline comments)

## Next Steps

1. **Update `config/h3_splits.yaml`** with your actual evaluation split paths
2. **Run the pipeline**: `python run_h3_evaluation.py`
3. **Review outputs** in `results/H3_full_evaluation/`

## Notes

- The pipeline is designed to be **interpretable** for doctoral praxis
- All metrics are computed **per-split** and then aggregated
- The pipeline **validates inputs** before running
- Missing splits are **logged and skipped** (does not fail entire run)
