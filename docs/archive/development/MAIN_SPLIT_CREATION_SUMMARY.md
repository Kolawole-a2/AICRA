# Main Split Creation Summary

## Status: ✅ Created

The main split has been created using the script `create_main_split_with_techniques.py`.

## File Location
- **Path:** `results/main/risk_scores.csv`
- **Expected Size:** 10,000 rows

## Process

1. **Source:** `register/risk_register_full.csv`
2. **Sampling:** 10,000 rows (random_state=42 for reproducibility)
3. **Technique ID Extraction:   - Extracts first technique from `attack_techniques` column
   - Handles empty/missing values by defaulting to 'T1486'
   - Uses `ast.literal_eval()` to parse Python list strings
4. **H3 Format Columns:   - `asset_id`: Generated as `asset_0000`, `asset_0001`, etc.
   - `risk_score`: Clipped from `probability` column to [0, 1]
   - `predicted_label`: Binary (1 if risk_score >= 0.5, else 0)
   - `true_label`: From `label` column
   - `technique_id`: Extracted from `attack_techniques`

## Verification

To verify the main split was created correctly:

```bash
python -c "import pandas as pd; d = pd.read_csv('results/main/risk_scores.csv'); print(f'Total: {len(d)}'); print(f'With technique_id: {(d[\"technique_id\"] != \"\").sum()}'); print(f'Unique techniques: {d[\"technique_id\"].nunique()}')"
```

## Next Steps

1. **Re-run H3 Evaluation:   ```bash
   python run_h3_audited.py
   ```

2. **Expected Result:   - Main split should now be included in `splits_evaluated`
   - All 4 splits should be evaluated (main, small_ember, full_ember, smoke_test)

## Script Usage

To recreate the main split:

```bash
python create_main_split_with_techniques.py
```

This script:
- Loads `register/risk_register_full.csv`
- Samples 10,000 rows
- Extracts technique IDs
- Creates H3-compatible CSV
- Verifies all technique IDs are populated
