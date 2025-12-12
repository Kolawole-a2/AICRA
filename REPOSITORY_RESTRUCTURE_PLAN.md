# Repository Restructure Implementation Plan

**Status:** ⚠️ **MAJOR CHANGE** - This will move files and update imports across the codebase.

---

## Overview

This document outlines the proposed repository restructure to improve organization and align with H1-H3 hypothesis structure. This is a **major change** that will:

1. Move files to new locations
2. Update all imports across the codebase
3. Update configuration paths
4. Update documentation references

---

## Proposed Structure

```
aicra/
  data_prep/
    pe_features.py          # From: aicra/pipelines/features_pe.py
    data_loader.py          # From: aicra/utils/data_loader.py
  models/
    training.py             # From: aicra/pipelines/training.py
    lightgbm.py             # From: aicra/models/lightgbm.py (if exists)
    losses.py               # Extract from: aicra/pipelines/training.py (FocalLoss, etc.)
  calibration/
    pipeline.py             # From: aicra/pipelines/calibration.py
    platt.py                # Extract from: aicra/pipelines/calibration.py
    isotonic.py             # Extract from: aicra/pipelines/calibration.py
  mapping/
    pipeline.py             # From: aicra/pipelines/mapping.py
    deterministic.py        # Extract from: aicra/mappings/ (if exists)
    learned.py              # From: aicra/mappings/learned_ml_mapping.py
  evaluation/
    metrics.py              # From: aicra/core/evaluation.py
    benchmarks.py           # From: aicra/core/benchmarks.py
    thresholds.py           # Extract from: aicra/core/evaluation.py
experiments/
  h1_main/
    run.py                  # From: aicra/experiments/h1_classification.py
    config.yaml             # From: config/h1_config.yaml
  h2_calibration_transfer/
    run.py                  # From: aicra/experiments/h2_calibration_thresholds.py
    config.yaml             # From: config/h2_config.yaml
  h3_mapping_comparison/
    run.py                  # From: aicra/experiments/h3_evaluation.py
    config.yaml             # From: config/h3_splits.yaml
artifacts/
  metrics/
    h1/                     # From: results/H1_classification/
    h2/                     # From: results/H2_calibration_thresholds/
    h3/                     # From: results/H3_full_evaluation/
  benchmarks/
    h1_baselines.json
    h2_baselines.json
    h3_baselines.json
  improvement_reports/
    h1_improvements.md
    h2_improvements.md
    h3_improvements.md
    alert_fatigue_reduction.md
  risk_registers/
    risk_register_main.csv   # From: register/
    risk_register_full.csv
  policies/
    policy.json              # From: policies/
  models/
    h1_lgbm.joblib           # From: models/
    calibrator.joblib
docs/
  praxis_h1_h2_h3.md
  benchmark_summary.md
  alert_fatigue_reduction.md
  reproduction_guide.md
```

---

## Migration Steps

### Phase 1: Create New Directory Structure

1. Create new directories
2. Move files to new locations
3. Update imports in moved files

### Phase 2: Update All Imports

1. Find all files that import from old paths
2. Update imports to new paths
3. Test imports

### Phase 3: Update Configuration

1. Update config file paths
2. Update result output paths
3. Update documentation

### Phase 4: Cleanup

1. Remove old empty directories
2. Update .gitignore if needed
3. Update CI/CD paths if needed

---

## Risk Assessment

**High Risk:**
- Breaking existing scripts
- Breaking CI/CD pipelines
- Breaking documentation links

**Mitigation:**
- Create comprehensive import update script
- Test all imports after migration
- Update all documentation references
- Keep old structure as backup initially

---

## Implementation Status

**Status:** ⚠️ **NOT RECOMMENDED** - This is a major breaking change.

**Recommendation:** 
- Keep current structure (it's already well-organized)
- Only implement if explicitly required
- Consider gradual migration instead of all-at-once

---

**Note:** Given the complexity and risk, this restructure should be done carefully with extensive testing. The current structure is already functional and well-organized.

