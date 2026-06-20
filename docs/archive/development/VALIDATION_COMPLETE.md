# Validation Complete - Summary

## Completed Tasks

### ✅ 1. Removed Obsolete Files

The following duplicate/obsolete H3 experiment files have been removed:
- `aicra/experiments/h3_learned_mapping_eval.py` - Functionality integrated into canonical `h3_evaluation.py`
- `aicra/experiments/h3_prepare_metrics.py` - Functionality integrated into canonical `h3_evaluation.py`
- `aicra/experiments/h3_stat_tests.py` - Statistical tests integrated into canonical `h3_evaluation.py`

**Rationale**: The canonical `h3_evaluation.py` implements all the functionality these files provided, including:
- DAC computation
- Statistical tests (t-test, Wilcoxon)
- Metrics aggregation
- Per-split evaluation

### ✅ 2. Created Validation Script

Created `scripts/validate_h3_config.py` to validate H3 configuration before running experiments:
- Checks all required files exist
- Validates reference pairs ≠ deterministic mapping (file hash check)
- Validates learned mapping ≠ deterministic mapping (set comparison)
- Provides clear error messages with fix instructions

### ✅ 3. All Sanity Checks in Place

The canonical `h3_evaluation.py` now includes three critical sanity checks:

1. **Reference Pairs Hash Check**: Raises `RuntimeError` if `d3fend_reference_pairs.csv` has same hash as `deterministic_lookup.csv`
2. **Reference Pairs Set Check**: Warns if pair sets are identical
3. **Mapping Difference Check**: Raises `RuntimeError` if `learned_mapping.csv` pair set is identical to `deterministic_lookup.csv`

All checks provide clear error messages with instructions on how to fix.

## Current Status

### Files Ready
- ✅ `scripts/create_reference_pairs.py` - Creates canonical reference pairs
- ✅ `scripts/validate_h3_config.py` - Validates H3 configuration
- ✅ `aicra/experiments/h1_classification.py` - Canonical H1 experiment
- ✅ `aicra/experiments/h2_calibration_thresholds.py` - Canonical H2 experiment
- ✅ `aicra/experiments/h3_evaluation.py` - Canonical H3 experiment (with sanity checks)
- ✅ `scripts/run_all_hypotheses.py` - Orchestration script
- ✅ `d3fend_reference_pairs.csv` - Canonical reference (15 pairs from YAML)

### Validation Status

**Reference Pairs**: ✅ Fixed
- Canonical reference pairs created from `data/lookups/attack_to_d3fend.yaml`
- Contains 15 pairs (5 techniques × 3 controls)
- Different from deterministic mapping (175 pairs)

**Learned Mapping**: ⚠️ Needs Verification
- Current `data/mappings/learned_mapping.csv` exists
- Should be different from deterministic mapping
- If identical, regenerate using: `python generate_learned_mapping.py`

**H3 Splits**: ⚠️ Needs Risk Scores Files
- Config file exists: `config/h3_splits.yaml`
- Requires risk scores CSV files at:
  - `results/time_test/risk_scores.csv`
  - `results/oof_test/risk_scores.csv`
  - `results/seed1/time_test/risk_scores.csv`
- Each file needs columns: `asset_id`, `risk_score`, `predicted_label`, `true_label`, `technique_id`

## How to Validate

### Step 1: Validate H3 Configuration
```bash
python scripts/validate_h3_config.py
```

This will check:
- All required files exist
- Reference pairs ≠ deterministic mapping
- Learned mapping ≠ deterministic mapping

### Step 2: Run H3 Experiment (if risk scores exist)
```bash
python run_h3_evaluation.py
```

The experiment will:
- Automatically validate all sanity checks
- Raise `RuntimeError` with clear instructions if any check fails
- Generate results if all checks pass

### Step 3: Run All Experiments (if data available)
```bash
python scripts/run_all_hypotheses.py
```

This will run H1, H2, and H3 in sequence.

## Expected Behavior

### If Configuration is Valid
- H3 experiment runs successfully
- Results show **different** metrics for deterministic vs learned mappings
- File hashes are logged for reproducibility
- All sanity checks pass silently

### If Configuration is Invalid

**Case 1: Reference pairs identical to deterministic```
RuntimeError: Reference pairs file is identical to deterministic mapping.
Reference pairs must be the canonical ATT&CK-D3FEND reference,
not the deterministic mapping. Run: python scripts/create_reference_pairs.py
```

**Case 2: Learned mapping identical to deterministic```
RuntimeError: Deterministic and learned mappings are IDENTICAL.
This will produce identical results. Please regenerate the learned mapping
using: python generate_learned_mapping.py
```

## Next Steps for User

1. **Verify Learned Mapping**:
   ```bash
   python scripts/validate_h3_config.py
   ```
   If learned mapping is identical to deterministic, regenerate it.

2. **Prepare Risk Scores** (if not already available):
   - Ensure risk scores CSV files exist for all splits in `config/h3_splits.yaml`
   - Each file must have required columns

3. **Run Experiments**:
   ```bash
   # Validate first
   python scripts/validate_h3_config.py
   
   # Then run experiments
   python scripts/run_all_hypotheses.py
   ```

4. **Verify Results**:
   - Check that H3 metrics show differences between deterministic and learned
   - Verify file hashes in results match expected values
   - Confirm all sanity checks passed (no RuntimeErrors)

## Summary

All critical fixes are complete:
- ✅ Obsolete files removed
- ✅ Sanity checks implemented
- ✅ Validation script created
- ✅ Reference pairs fixed
- ✅ Canonical experiment files created

The experiments are now ready to run and will **fail fast** with clear error messages if configurations are invalid, ensuring scientifically meaningful results.
