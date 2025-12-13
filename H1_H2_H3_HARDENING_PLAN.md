# H1-H3 Experiment Hardening Plan

## Overview
This document outlines the plan to harden and standardize all H1–H3 experiments for reproducibility, correctness, completeness, and robustness.

## Current State Analysis

### ✅ Already Implemented
- Time-ordered split logic exists in `aicra/core/data.py::load_ember_2024()`
- Baseline computation functions exist in `aicra/core/benchmarks.py`
- Class imbalance handling exists in `aicra/pipelines/training.py` (class_weight, scale_pos_weight)
- Some out-of-family evaluation code exists in `aicra/experiments/h1_classification.py`
- H1, H2, H3 experiment modules exist

### ❌ Needs Implementation
- Standardized entrypoint scripts (experiments/h1_train_eval.py, etc.)
- Explicit out-of-family split (train on some families, test on held-out)
- Temporal calibration check in H2
- Learned == deterministic bug check in H3
- Consolidated benchmark improvements CSV/MD generator
- Documentation updates

## Implementation Plan

### A) Standardized Entrypoints
**Files to create:**
- `experiments/h1_train_eval.py` - Wrapper around h1_classification.py
- `experiments/h2_calibration_eval.py` - Wrapper around h2_calibration_thresholds.py  
- `experiments/h3_mapping_compare.py` - Wrapper around h3_evaluation.py

**Requirements:**
- Use `get_ember2024_dir()` from `aicra.utils.data_paths`
- Write outputs to `artifacts/`
- Log seeds, configs, timestamps
- Runnable from repo root

### B) Time-Ordered Split + Out-of-Family Test
**Changes needed:**
- Verify time-ordered split is used consistently in H1
- Implement explicit out-of-family split (hold out families)
- Produce separate metrics for "Temporal Test" and "Out-of-Family Test"

### C) Metrics and Thresholding
**H1:**
- ✅ AUROC, Precision, Recall, F1 (already computed)
- ✅ Confusion matrix at banking threshold (already computed)
- ✅ Threshold stored in JSON (already done)
- Add: Explicit temporal and out-of-family metrics

**H2:**
- ✅ Brier, ECE pre/post (already computed)
- Add: Temporal calibration check (calibrate on earlier window, test on later)

**H3:**
- ✅ Coverage, consistency, precision, variance (already computed)
- ✅ Statistical tests (already computed)
- Add: Learned == deterministic bug check

### D) Benchmarks + % Improvements
**Status:** Already computed in benchmarks.py, but need to:
- Generate consolidated `artifacts/benchmark_improvements.csv`
- Generate `artifacts/benchmark_improvements.md`
- Ensure all experiments write to these files

### E) Imbalanced Data Handling
**Status:** Already implemented in training pipeline
- ✅ class_weight="balanced" support
- ✅ scale_pos_weight computation
- Need: Document which strategy is used and log to artifacts

### F) Learned == Deterministic Bug Check
**New requirement:**
- Add hard check in H3: if learned mapping == deterministic mapping, fail with error
- Write diagnostic report to `artifacts/h3_mapping_integrity.json`

### G) Documentation
- Update README with experiment commands
- Create `docs/EXPERIMENTS.md` with step-by-step reproduction

## Files to Create/Modify

### New Files
1. `experiments/h1_train_eval.py`
2. `experiments/h2_calibration_eval.py`
3. `experiments/h3_mapping_compare.py`
4. `aicra/utils/benchmark_reporter.py` (consolidated benchmark CSV/MD generator)
5. `docs/EXPERIMENTS.md`

### Modified Files
1. `aicra/experiments/h1_classification.py` - Add explicit out-of-family split
2. `aicra/experiments/h2_calibration_thresholds.py` - Add temporal calibration check
3. `aicra/experiments/h3_evaluation.py` - Add learned == deterministic check
4. `README.md` - Add experiment commands and benchmarks section

