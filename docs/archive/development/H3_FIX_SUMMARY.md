> **Archive alignment (2026):** Historical development note. Canonical narrative: H1 time-ordered + multi-split + OOF (AUROC **> 0.88**, empirical baseline ≈ 0.778); H2 calibration **help test** + cost-optimal thresholds; H3 **perfect separation** when variance is zero. See [../../../praxis/README.md](../../../praxis/README.md) and [../../../README.md](../../../README.md).

# H3 Pipeline Fix Summary

## Problem Identified

The H3_comparison results were showing identical values for deterministic and learned mappings because:

1. **Root-level files are identical**: `deterministic_lookup.csv` and `learned_mapping.csv` in the root directory have the SAME pairs (only 16 pairs)
2. **Wrong deterministic file**: `run_h3_experiment.py` was using the small root-level `deterministic_lookup.csv` instead of the full one from `mappings_project/data/mappings/`
3. **Embedding-based learned mapping not generated**: The new embedding-based learned mapping file may not exist or wasn't being used

## Files Updated

### 1. `run_h3_experiment.py`
- Updated `IN_DET` to use full deterministic lookup from `data/mappings/deterministic_attack_defense_lookup.csv`
- Updated `IN_LRN` to use `data/mappings/learned_embedding_attack_defense_mapping.csv`
- Added column name handling (attack_id/defense_id ↔ technique_id/control_id)
- Added verification to check if mappings are different

### 2. `aicra/experiments/h3_prepare_metrics.py`
- Now computes precision and variance metrics on-the-fly from mappings
- Uses embedding-based learned mapping
- No longer relies on pre-computed `ransomware_performance_by_attack.csv`

## Required Setup Steps

Before running H3 experiment, ensure:

1. **Full deterministic lookup exists**:
   ```bash
   # Copy from mappings_project if needed
   mkdir -p data/mappings
   cp mappings_project/data/mappings/deterministic_attack_defense_lookup.csv data/mappings/
   ```

2. **Generate embedding-based learned mapping**:
   ```bash
   python -m aicra.mappings.embedding_learned_mapping
   ```

3. **Verify mappings are different**:
   The script will print a comparison showing overlap vs unique pairs.

## How to Run

```bash
# Step 1: Generate embedding-based learned mapping
python -m aicra.mappings.embedding_learned_mapping

# Step 2: Run H3 experiment
python run_h3_experiment.py

# Step 3: Prepare H3 metrics
python -m aicra.experiments.h3_prepare_metrics

# Step 4: Run statistical tests
python -m aicra.experiments.h3_stat_tests
```

## Expected Results

After the fix:
- Deterministic and learned mappings should show **different** values
- Coverage, consistency, precision, and variance should differ
- The embedding-based learned mapping should have different pairs than deterministic
- H3_comparison plots should show differences

## Verification

Check `results/H3_comparison/H3_results.json`:
- `deterministic_mapping.pairs_count` should be ~175 (full lookup)
- `learned_mapping.pairs_count` should be different
- `delta_precision` should NOT be 0.0
- `delta_variance_reduction` should NOT be 0.0

