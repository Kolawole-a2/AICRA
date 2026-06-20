> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# STEP 1: Mapping Locations and Dataset Splits Summary

## Deterministic Lookup Mapping (Authoritative Baseline)

**Location:** `data/lookups/attack_to_d3fend.yaml`
- **Type:** Static YAML file with curated ATT&CK → D3FEND mappings
- **Format:** YAML with technique_id as keys, list of D3FEND control IDs as values
- **Status:** Authoritative, expected to achieve highest DAC
- **Coverage:** 5 ATT&CK techniques mapped to 11 D3FEND controls (15 pairs total)
- **Integration Point:** `aicra/pipelines/mapping.py` - `MappingPipeline` class loads this

**Key Techniques:- T1486 (Data Encrypted for Impact) → D3-BDR, D3-BAC, D3-SAW
- T1490 (Inhibit System Recovery) → D3-BDR, D3-BAC, D3-SAW
- T1059 (Command and Scripting Interpreter) → D3-SAW, D3-CR, D3-AL
- T1021 (Remote Services) → D3-NFP, D3-VPM, D3-AA
- T1070 (Indicator Removal on Host) → D3-EDR, D3-SIEM, D3-AV

## Learned/Heuristic Mapping

**Current Status:** Not yet implemented as a separate file
- **Expected Location:** `learned_mapping.csv` or `data/lookups/learned_attack_to_d3fend.csv`
- **Type:** Model-driven or heuristic inference (may contain errors)
- **Format:** CSV with columns: `technique_id`, `control_id`, optionally `score` (confidence)
- **Integration Point:** Will be loaded in H3 experiment for comparison

**Note:** For the experiment, we'll generate a learned mapping that intentionally has some differences from deterministic to demonstrate the performance gap.

## Dataset Splits

**Test Phases (defined in `aicra/pipelines/test_runner.py`):1. **smoke** - Minimal test with synthetic data
   - Purpose: Quick validation
   - Size: ~1000 samples
   - Location: Generated synthetically

2. **small_ember** - Subset of EMBER-2024 data
   - Purpose: Medium-scale validation
   - Size: Configurable (default 10,000 samples)
   - Location: `data/ember2024_real/` (JSONL files)
   - Split: 80% train, 20% test

3. **full** - Complete EMBER-2024 dataset
   - Purpose: Full-scale evaluation
   - Size: All available data
   - Location: `data/ember2024_real/` (all JSONL files)
   - Split: 80% train, 20% test (or time-ordered if `time_split=True`)

**Time-Ordered Splits (defined in `aicra/core/splits.py`):- `time_ordered_split()` - Splits by timestamp
- Default ratios: 70% train, 15% val, 15% test
- Ensures temporal ordering (no data leakage)

## DAC Integration Points

**Where DAC logic will integrate:1. **New module:** `aicra/metrics/dac.py` - Core DAC computation
2. **New experiment:** `aicra/experiments/h3_dac_validation.py` - H3 experiment runner
3. **New analysis:** `aicra/analysis/h3_dac_stats.py` - Statistical testing
4. **Results directory:** `results/h3_dac_*` - All outputs

## File Structure Summary

```
data/lookups/
  ├── attack_to_d3fend.yaml          # Deterministic mapping (authoritative)
  ├── family_to_attack.yaml          # Family → ATT&CK mapping
  └── canonical_families.yaml        # Family normalization

aicra/
  ├── metrics/
  │   └── dac.py                     # [TO CREATE] DAC computation
  ├── experiments/
  │   └── h3_dac_validation.py       # [TO CREATE] H3 experiment
  └── analysis/
      └── h3_dac_stats.py            # [TO CREATE] Statistical tests

results/
  ├── h3_dac_metrics_by_split.csv    # [TO CREATE] Metrics per split
  ├── h3_dac_stats_summary.csv       # [TO CREATE] Statistical summary
  ├── h3_dac_artifacts.jsonl         # [TO CREATE] Reproducibility hashes
  └── plots/
      ├── h3_dac_vs_precision.png    # [TO CREATE]
      └── h3_dac_vs_variance_reduction.png  # [TO CREATE]

docs/
  └── h3_dac_validation_results.md   # [TO CREATE] Human-readable report
```

## Next Steps

1. Create `aicra/metrics/dac.py` with DAC computation
2. Create `aicra/experiments/h3_dac_validation.py` to run across all splits
3. Create `aicra/analysis/h3_dac_stats.py` for statistical testing
4. Generate learned mapping for comparison
5. Run experiment across all phases: smoke → small_ember → full

