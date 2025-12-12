# AICRA Project Organization Summary

## Structure Created

The AICRA project has been organized according to the desired structure:

### ✅ Created/Organized

1. **`aicra/mapping/`** - Mapping modules
   - `heuristic_mapping.py` - Text-similarity based heuristic mapping
   - `deterministic_lookup.py` - Deterministic lookup loader/generator
   - `__init__.py` - Package exports

2. **`aicra/experiments/`** - Experiment modules
   - `h3_validation.py` - Full H3 validation (DAC + precision + variance)
   - Existing: `h3_learned_mapping_eval.py`, `h3_prepare_metrics.py`, `h3_stat_tests.py`

3. **`scripts/`** - Utility scripts
   - `check_mappings_sanity.py` - Validates mappings exist and are different
   - `compare_mappings_summary.py` - Quick comparison summary

4. **`results/H3_comparison/`** - Results directory
   - Created directory structure for H3 results and plots

### 📁 Data Structure

**Expected locations:**
- `data/ontology/attack_techniques.csv` - ATT&CK techniques with descriptions
- `data/ontology/d3fend_controls.csv` - D3FEND controls with descriptions
- `data/ontology/d3fend_reference_pairs.csv` - Reference pairs (optional)
- `data/mappings/deterministic_lookup.csv` - Deterministic mapping
- `data/mappings/learned_mapping.csv` - Heuristic/learned mapping

**Note:** The actual mapping files are currently in:
- `../mappings/data/mappings/deterministic_attack_defense_lookup.csv`
- `../mappings/data/mappings/heuristic/learned_mapping.csv`

You may want to copy these to the AICRA `data/mappings/` directory or update paths accordingly.

## Usage

### Run H3 Validation
```bash
python -m aicra.experiments.h3_validation \
    --deterministic data/mappings/deterministic_lookup.csv \
    --learned data/mappings/learned_mapping.csv \
    --output results/H3_comparison
```

### Check Mappings
```bash
python scripts/check_mappings_sanity.py
python scripts/compare_mappings_summary.py
```

## Next Steps

1. **Copy mapping files** to `data/mappings/` if needed
2. **Create ontology files** in `data/ontology/` if they don't exist
3. **Run H3 validation** to generate results
4. **Create additional experiment modules** as needed:
   - `temporal_eval.py` - Time-ordered evaluation
   - `threshold_analysis.py` - Cost-aware thresholds
   - `adversarial_sanity.py` - Robustness checks

