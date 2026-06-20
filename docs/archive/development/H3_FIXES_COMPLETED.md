> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# H3 Evaluation Fixes - Completed Tasks

## Summary of Changes

All requested fixes have been implemented in the code. The H3 evaluation now:

1. ✅ **Uses single DAC metric** (removed internal/external split)
2. ✅ **Fixed actionable precision** to use deterministic mapping as reference
3. ✅ **Updated all code references** from `dac_internal`/`dac_external` to `dac`
4. ✅ **Updated markdown generation** to use single DAC metric

## Key Changes Made

### 1. Single DAC Metric
- **Before:** `DAC_internal` (agreement with deterministic) and `DAC_external` (agreement with external reference pairs)
- **After:** Single `DAC` metric (agreement with deterministic mapping - ransomware-focused ground truth)
- **Function:** `compute_dac()` replaces `compute_dac_internal()` and `compute_dac_external()`
- **JSON field:** `dac_%` replaces `dac_internal_%` and `dac_external_%`

### 2. Fixed Actionable Precision Logic
- **Before:** Used external reference pairs to determine "actionable"
- **After:** Uses deterministic mapping (ransomware-focused) as reference
- **Logic:  - For deterministic mapping: All controls are ransomware-relevant → precision = 1.0
  - For learned mapping: Only controls that overlap with deterministic are ransomware-relevant → precision < 1.0
- **Result:** Deterministic precision will always be >= learned precision

### 3. Code Updates
- ✅ `compute_dac()` function created (replaces `compute_dac_internal()`)
- ✅ `compute_actionable_metrics()` updated to use deterministic mapping as reference
- ✅ `compute_mapping_metrics()` updated to return `dac_%` instead of `dac_internal_%`/`dac_external_%`
- ✅ All aggregation functions updated
- ✅ All statistical test functions updated
- ✅ All plotting functions updated
- ✅ Markdown generation updated (table headers, section titles, narrative)

## Files Modified

1. **`aicra/experiments/h3_evaluation.py`   - Updated `compute_dac()` function (single metric)
   - Updated `compute_actionable_metrics()` to use deterministic mapping
   - Updated `compute_mapping_metrics()` to return `dac_%`
   - Updated all aggregation and statistical test functions
   - Updated markdown generation function
   - Updated plotting functions

## Expected Results After Re-run

When you re-run the H3 evaluation, you should see:

1. **JSON Output:   - `dac_%` field (not `dac_internal_%` or `dac_external_%`)
   - Deterministic `dac_%` = 100.0%
   - Learned `dac_%` < 100.0%

2. **Actionable Precision:   - Deterministic precision >= Learned precision
   - Deterministic precision = 1.0 (if it has controls for techniques in risk scores)
   - Learned precision < 1.0 (only some controls are ransomware-relevant)

3. **Markdown Summary:   - Single "DAC" column (not "DAC_int" and "DAC_ext")
   - Updated narrative referring to single DAC metric
   - No references to "DAC_internal" or "DAC_external"

## Next Steps

To see the updated results:

1. **Clear Python cache** (if needed):
   ```powershell
   Get-ChildItem -Path "aicra" -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force
   ```

2. **Re-run H3 evaluation:   ```powershell
   python -m aicra.experiments.h3_evaluation `
     --config config/h3_splits.yaml `
     --deterministic data/mappings/deterministic_attack_defense_lookup.csv `
     --learned data/mappings/learned_mapping.csv `
     --output results/H3_full_evaluation
   ```

3. **Verify results:   - Check `results/H3_full_evaluation/H3_full_results.json` has `dac_%` field
   - Check deterministic precision > learned precision
   - Check markdown summary uses single DAC metric

## Notes

- The `compute_dac_external()` function still exists in the code but is no longer called
- All references to `dac_internal` and `dac_external` in code have been updated to `dac`
- The markdown generation will automatically use the new field names when regenerated

--**Status:** All code changes completed. Ready for re-run to generate updated results.

