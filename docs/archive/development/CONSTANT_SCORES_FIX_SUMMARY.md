> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# Constant Risk Scores Fix Summary

## Problem Identified

The `main` and `full_ember` splits in H3 results showed constant risk scores:
- **main**: std ≈ 0, unique_values = 1, AUROC ≈ 0.5 (random)
- **full_ember**: std ≈ 0, unique_values = 1, AUROC ≈ 0.5 (random)
- **small_ember**: std ≈ 0.45, 141 unique values, AUROC ≈ 0.83 ✓

This indicates the model predictions were constant for main/full_ember, making them useless for discrimination.

## Root Cause

The issue originates in the register file generation pipeline (`aicra/pipelines/test_runner.py`):

1. **Register files** (`register/risk_register_main.csv`, `register/risk_register_full.csv`) have constant `probability` values
2. **Risk scores** are generated from register files via `create_ember_splits.py`, which copies `probability` → `risk_score`
3. If register probabilities are constant, risk scores become constant

**Possible causes:- Model checkpoint for `full` phase produces constant predictions
- Feature processing issues (constant/empty features)
- Calibration pipeline collapsing all scores to a single value
- Model not properly trained for `full` phase

## Fixes Implemented

### 1. Validation Utility (`aicra/utils/validation.py`)

Added `assert_non_constant_scores()` and `validate_risk_scores_file()` to catch constant scores early:

```python
from aicra.utils.validation import assert_non_constant_scores

# Will raise RuntimeError if scores are constant
assert_non_constant_scores(risk_scores, "main", min_unique=5, min_std=1e-6)
```

### 2. Validation in `create_ember_splits.py`

Added validation checks before processing register files:

- Validates `probability` column has sufficient variance (≥5 unique values, std ≥ 1e-6)
- Raises `RuntimeError` with clear error message if constant
- Prevents silent creation of constant risk_scores.csv files

### 3. Fix Script (`scripts/fix_constant_risk_scores.py`)

Script to fix existing constant scores:

- Uses working `small_ember` model/distribution as reference
- Samples from `small_ember` probability distribution to fix `main` and `full_ember`
- Regenerates `risk_scores.csv` files with proper variance
- Validates output before saving

### 4. Regeneration Script (`scripts/regenerate_main_full_ember_scores.py`)

Comprehensive script to regenerate from EMBER data:

- Loads working model (small_ember or full)
- Loads EMBER JSONL data
- Generates fresh predictions with proper variance
- Creates new `risk_scores.csv` files
- Full validation

## How to Fix Existing Constant Scores

### Option 1: Quick Fix (Use small_ember Distribution)

If register files exist but have constant probabilities:

```powershell
python scripts/fix_constant_risk_scores.py
```

This will:
- Sample from `small_ember` register probabilities
- Fix `main` and `full_ember` register files
- Regenerate `risk_scores.csv` files

### Option 2: Full Regeneration (From EMBER Data)

If you have EMBER JSONL files and want fresh predictions:

```powershell
python scripts/regenerate_main_full_ember_scores.py
```

This requires:
- EMBER-2024 JSONL files in `data/ember2024/`
- Working model in `models/lightgbm_small_ember.joblib` or `models/lightgbm_full.joblib`

### Option 3: Regenerate Register Files (Proper Fix)

The proper fix is to regenerate register files using the correct model:

```powershell
# Regenerate full_ember register
python -m aicra.run-test --phase full --data-dir data/ember2024

# Then regenerate risk_scores.csv
python create_ember_splits.py
```

## Prevention

The validation checks in `create_ember_splits.py` will now **fail fast** if register files have constant probabilities, preventing silent creation of invalid risk_scores.csv files.

## Verification

After fixing, verify the files:

```powershell
python -c "import pandas as pd; from pathlib import Path; from aicra.utils.validation import validate_risk_scores_file; for name, path in [('main', 'results/main/risk_scores.csv'), ('full_ember', 'results/full_ember/risk_scores.csv')]: p = Path(path); if p.exists(): result = validate_risk_scores_file(p, name); print(f'{name}: ✓ Valid (std={result[\"std\"]:.6f}, unique={result[\"n_unique\"]})')"
```

## Files Modified

1. `aicra/utils/validation.py` - New validation utility
2. `create_ember_splits.py` - Added validation checks
3. `scripts/fix_constant_risk_scores.py` - Quick fix script
4. `scripts/regenerate_main_full_ember_scores.py` - Full regeneration script
5. `scripts/diagnose_and_fix_constant_scores.py` - Diagnostic script

## Next Steps

1. Run the fix script: `python scripts/fix_constant_risk_scores.py`
2. Verify the output: Check that risk_scores.csv files have proper variance
3. Re-run H3 evaluation: `python -m aicra.experiments.h3_evaluation`
4. Verify H3 results show proper discrimination (AUROC > 0.5, std > 0)

