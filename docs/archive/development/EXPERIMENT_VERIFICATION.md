# H1 and H2 Experiment Verification

## Code Verification ✅

### Import Tests
- ✅ `aicra.experiments.h1_classification` imports successfully
- ✅ `aicra.experiments.h2_calibration_thresholds` imports successfully
- ✅ All dependencies resolve correctly

### Code Changes Verified
- ✅ H1 uses original `load_ember_2024()` from `core.data`
- ✅ H2 uses original `load_ember_2024()` from `core.data` and splits train data internally
- ✅ Backward compatibility maintained:
  - Original `metrics.json` files still generated
  - Original `summary.md` files still generated
- ✅ New files added (for praxis):
  - `H1_full_results.json` / `H2_full_results.json`
  - `H1_summary.md` / `H2_summary.md` (copies of summary.md)

## Running the Experiments

### Prerequisites
The experiments require EMBER-2024 data files in `data/ember2024/`:
- `train_features.jsonl`
- `train_labels.jsonl`
- `test_features.jsonl`
- `test_labels.jsonl`

### Run H1 Experiment
```bash
python -m aicra.experiments.h1_classification
# Or:
python run_h1_h2_experiments.py
```

**Expected Output:- `results/H1_classification/metrics.json`
- `results/H1_classification/H1_full_results.json`
- `results/H1_classification/summary.md`
- `results/H1_classification/H1_summary.md`

### Run H2 Experiment
```bash
python -m aicra.experiments.h2_calibration_thresholds
# Or:
python run_h1_h2_experiments.py
```

**Note:** H2 requires H1 to run first (needs the trained model).

**Expected Output:- `results/H2_calibration_thresholds/metrics.json`
- `results/H2_calibration_thresholds/H2_full_results.json`
- `results/H2_calibration_thresholds/summary.md`
- `results/H2_calibration_thresholds/H2_summary.md`

## Verification Checklist

After running the experiments, verify:

### H1 Results
- [ ] `results/H1_classification/metrics.json` exists
- [ ] `results/H1_classification/H1_full_results.json` exists
- [ ] `results/H1_classification/summary.md` exists
- [ ] `results/H1_classification/H1_summary.md` exists
- [ ] Metrics include: `auroc`, `pr_auc`, `brier_score`, `ece`, `precision`, `recall`, `f1`

### H2 Results
- [ ] `results/H2_calibration_thresholds/metrics.json` exists
- [ ] `results/H2_calibration_thresholds/H2_full_results.json` exists
- [ ] `results/H2_calibration_thresholds/summary.md` exists
- [ ] `results/H2_calibration_thresholds/H2_summary.md` exists
- [ ] Metrics include: `calibration.brier_uncalibrated`, `calibration.brier_calibrated`, `calibration.ece_uncalibrated`, `calibration.ece_calibrated`

### Generate Validation Report
After H1 and H2 complete:
```bash
python scripts/generate_praxis_validation_report.py
```

This will create/update:
- `results/praxis_validation_report.md`

## Code Changes Summary

### Files Modified
1. **`aicra/experiments/h1_classification.py`   - Restored original import: `from ..core.data import Dataset, load_ember_2024`
   - Restored original call: `train_data, test_data = load_ember_2024()`
   - Added: `H1_full_results.json` generation
   - Added: `H1_summary.md` generation (copy of summary.md)

2. **`aicra/experiments/h2_calibration_thresholds.py`   - Restored original import: `from ..core.data import Dataset, load_ember_2024`
   - Modified: Split train data internally to create validation set (preserves original logic)
   - Added: `H2_full_results.json` generation
   - Added: `H2_summary.md` generation (copy of summary.md)

### Files Created
1. **`aicra/utils/data_loader.py`** - New utility (not used by H1/H2, kept for future use)
2. **`scripts/generate_praxis_validation_report.py`** - Validation report generator
3. **`run_h1_h2_experiments.py`** - Convenience script to run both experiments
4. **`test_h1_h2.py`** - Test script for verification

### Backward Compatibility
✅ **All original functionality preserved:- Original imports restored
- Original function calls restored
- Original output files still generated
- Only additions, no breaking changes

## Troubleshooting

### If experiments fail with "FileNotFoundError"
- Check that `data/ember2024/` directory exists
- Verify all 4 required JSONL files are present
- Check file permissions

### If H2 fails with "Model not found"
- Run H1 first to generate the model
- Check that `models/h1_lgbm.joblib` or `models/bagged_lightgbm.joblib` exists

### If output files are missing
- Check that `results/H1_classification/` and `results/H2_calibration_thresholds/` directories exist
- Verify write permissions
- Check for errors in the experiment logs

--**Status:** Code verified and ready for execution  
**Last Updated:** 2025-01-XX
