# Fix: Full EMBER Dataset Has 0 Techniques

## Problem

The `full_ember` split shows 0 techniques, which is incorrect. The issue is that the `technique_id` column in `results/full_ember/risk_scores.csv` contains empty strings instead of actual technique IDs.

## Root Cause

When `create_ember_splits.py` runs, it extracts technique IDs from the `attack_techniques` column in the source register file. If the source file (`register/risk_register_full.csv`) has empty or missing `attack_techniques`, the extraction function returns `None`, which gets saved as empty strings in the CSV.

## Solution

1. **Fixed `load_risk_scores()` function** to convert empty strings to NaN:
   - Empty strings in `technique_id` are now treated as missing values
   - This ensures proper counting of techniques

2. **Check source register file** to see if `attack_techniques` is populated:
   - If empty, the source data needs to be regenerated with proper technique extraction
   - If populated, the extraction function may need fixing

3. **Regenerate full_ember split** if needed:
   ```bash
   python create_ember_splits.py
   ```

## Next Steps

1. Verify the source register file has `attack_techniques` populated
2. If not, investigate why the register file is missing technique data
3. Regenerate the split files if needed
4. Re-run H3 evaluation to get correct technique counts
