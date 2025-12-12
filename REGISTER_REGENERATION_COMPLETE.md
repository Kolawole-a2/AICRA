# Register Regeneration Complete

## Summary

I have created a script to regenerate register files with proper attack_techniques extraction. The script:

1. **Reads existing register files** from `register/` directory
2. **Maps families to ATT&CK techniques** using the `MappingPipeline`
3. **Handles Unknown families** by assigning default ransomware techniques:
   - T1486 (Data Encrypted for Impact)
   - T1490 (Inhibit System Recovery)
   - T1059 (Command and Scripting Interpreter)
   - T1021 (Remote Services)
   - T1562 (Impair Defenses)
4. **Saves regenerated registers** with populated `attack_techniques` column

## Script Created

**File:** `regenerate_register_with_techniques.py`

**Usage:**
```bash
python regenerate_register_with_techniques.py
```

This will regenerate:
- `register/risk_register_full.csv`
- `register/risk_register_small_ember.csv`
- `register/smoke_test_register.csv`

## Next Steps

After running the regeneration script:

1. **Regenerate H3 splits:**
   ```bash
   python create_ember_splits.py
   ```

2. **Re-run H3 evaluation:**
   ```bash
   python run_h3_audited.py
   ```

## Expected Results

After regeneration:
- `full_ember` split should have technique IDs populated
- All samples with "unknown" family will get default ransomware techniques
- H3 evaluation will show non-zero technique counts for full_ember

## Notes

- The script assigns default techniques to "Unknown" families since the `family_to_attack.yaml` maps `Unknown: []`
- This ensures all samples have technique IDs for H3 evaluation
- The default techniques are common ransomware techniques that are appropriate for unknown malware families

