# Mappings Experiment Files Copied to AICRA

## Summary

All mapping experiment files have been successfully copied from the `mappings` project to the `AICRA` repository and are ready for H3 experiments.

## Files Copied

### Mapping Data Files

1. **`data/mappings/deterministic_lookup.csv`**
   - Source: `../mappings/data/mappings/deterministic_attack_defense_lookup.csv`
   - Purpose: Deterministic ATT&CK→D3FEND mapping (gold standard)
   - Statistics: 173 rows, 46 techniques, 9 defenses

2. **`data/mappings/deterministic_lookup.parquet`**
   - Source: `../mappings/data/mappings/deterministic_attack_defense_lookup.parquet`
   - Purpose: Binary format of deterministic mapping

3. **`data/mappings/learned_mapping.csv`**
   - Source: `../mappings/data/mappings/heuristic/learned_mapping.csv`
   - Purpose: Learned/heuristic mapping for H3 comparison
   - Statistics: 183 rows, 46 techniques, 71 controls
   - Note: Ransomware-filtered version with 100% technique coverage

### Scripts

1. **`scripts/check_mappings_sanity.py`**
   - Validates both mappings exist and are different
   - Checks file hashes and schema

2. **`scripts/compare_mappings_summary.py`**
   - Comprehensive comparison summary
   - Coverage, DAC, and mapping statistics

3. **`scripts/compute_dac_metrics.py`**
   - Computes DAC metrics for both mappings
   - Comparison with reference pairs

4. **`scripts/validate_mappings.py`**
   - Additional validation script

## Verification

Both mappings have been verified:
- ✅ Files exist and are different (different SHA256 hashes)
- ✅ Same technique coverage: 46/46 techniques (100%)
- ✅ Different mappings: 0% pair overlap
- ✅ Similar size: 183 vs 173 mappings

## Usage

### Check Mappings
```bash
python scripts/check_mappings_sanity.py
python scripts/compare_mappings_summary.py
```

### Run H3 Validation
```bash
python -m aicra.experiments.h3_validation \
    --deterministic data/mappings/deterministic_lookup.csv \
    --learned data/mappings/learned_mapping.csv \
    --output results/H3_comparison
```

## File Locations in AICRA

- **Deterministic**: `data/mappings/deterministic_lookup.csv`
- **Learned**: `data/mappings/learned_mapping.csv`
- **Scripts**: `scripts/check_mappings_sanity.py`, `scripts/compare_mappings_summary.py`
- **Results**: `results/H3_comparison/` (created when running H3 validation)

All files are ready for H3 experiment comparison!

