# H3 DAC Statistical Validation

## Overview

This document describes the H3 statistical validation pipeline that compares deterministic ATT&CK→D3FEND mapping with learned ML-based mapping.

## Background

### Deterministic Mapping
The deterministic mapping represents the ontology truth—authoritative, curated relationships between ATT&CK techniques and D3FEND defensive controls derived from MITRE's ontology. This serves as the gold standard against which learned mappings are evaluated.

### Learned Mapping
The learned mapping is produced by an ML classifier that predicts D3FEND defenses from ATT&CK technique text (attack names/descriptions). The classifier is trained on the deterministic mapping but must infer mappings based on language features rather than ontology edges at inference time.

### Defense-Attack Consistency (DAC)
DAC measures the consistency between the deterministic and learned mappings. It is computed as:
- **Global DAC**: `overlap(deterministic, learned) / total(deterministic pairs)`
- **Per-attack DAC**: Same formula computed for each attack_id individually

DAC quantifies how well the learned mapping aligns with the ontology-driven deterministic mapping.

### Operational Metrics
- **Δprecision**: `precision_learned - precision_det` - Measures the difference in precision between learned and deterministic mappings
- **Variance reduction**: `variance_det - variance_learned` - Measures the reduction in risk score variance when using deterministic vs learned mapping

## Pipeline Components

### 1. DAC Computation (`aicra/metrics/dac.py`)

The `dac.py` module provides two key functions:

- **`compute_dac_between_mappings(df_det, df_learned)`**: Computes global DAC metrics
- **`compute_dac_per_attack(df_det, df_learned)`**: Computes DAC metrics for each attack_id

Per-attack metrics include:
- `n_det_pairs`: Number of deterministic pairs for this attack
- `n_learned_pairs`: Number of learned pairs for this attack
- `n_overlap_pairs`: Number of overlapping pairs
- `dac_attack`: DAC score for this attack (overlap / det_pairs)
- `precision_learned_wrt_det_attack`: Precision of learned pairs (overlap / learned_pairs)
- `coverage_det`: 1 if det_pairs > 0 else 0
- `coverage_learned`: 1 if learned_pairs > 0 else 0

### 2. Metrics Preparation (`aicra/experiments/h3_prepare_metrics.py`)

This script:
1. Loads deterministic and learned mappings
2. Loads `results/ransomware_performance_by_attack.csv` (contains `attack_id`, `precision_det`, `precision_learned`, `variance_det`, `variance_learned`)
3. Computes DAC per attack using `compute_dac_per_attack()`
4. Merges DAC metrics with performance metrics
5. Adds `delta_precision` and `variance_reduction` columns
6. Saves to `results/h3_metrics_attack_level.csv`

### 3. Statistical Tests (`aicra/experiments/h3_stat_tests.py`)

This script performs statistical validation:
1. Loads `results/h3_metrics_attack_level.csv`
2. Runs paired t-tests:
   - Δprecision vs 0 (tests if learned mapping improves precision)
   - Variance reduction vs 0 (tests if deterministic mapping reduces variance)
3. Computes Spearman correlations:
   - DAC vs Δprecision (tests if higher DAC correlates with precision improvement)
   - DAC vs variance_reduction (tests if higher DAC correlates with variance reduction)
4. Saves results to `results/h3_dac_stat_tests.json`

## Usage

### Step 1: Prepare Metrics

```bash
python -m aicra.experiments.h3_prepare_metrics
```

This will:
- Load mappings from `data/mappings/deterministic_attack_defense_lookup.csv` and `data/mappings/learned_attack_defense_mapping.csv`
- Load performance metrics from `results/ransomware_performance_by_attack.csv`
- Generate `results/h3_metrics_attack_level.csv`

### Step 2: Run Statistical Tests

```bash
python -m aicra.experiments.h3_stat_tests
```

This will:
- Load metrics from `results/h3_metrics_attack_level.csv`
- Perform statistical tests
- Save results to `results/h3_dac_stat_tests.json`

## Input Files

### Required Inputs

1. **`data/mappings/deterministic_attack_defense_lookup.csv`**
   - Columns: `attack_id`, `defense_id` (and optionally others)
   - Gold standard mapping from ontology

2. **`data/mappings/learned_attack_defense_mapping.csv`**
   - Columns: `attack_id`, `defense_id` (and optionally others)
   - ML-based learned mapping

3. **`results/ransomware_performance_by_attack.csv`**
   - Columns: `attack_id`, `precision_det`, `precision_learned`, `variance_det`, `variance_learned`
   - Performance metrics for each attack

## Output Files

### Generated Outputs

1. **`results/h3_metrics_attack_level.csv`**
   - One row per attack_id
   - Contains DAC metrics, performance metrics, and delta metrics
   - Columns: `attack_id`, `n_det_pairs`, `n_learned_pairs`, `n_overlap_pairs`, `dac_attack`, `precision_learned_wrt_det_attack`, `coverage_det`, `coverage_learned`, `precision_det`, `precision_learned`, `variance_det`, `variance_learned`, `delta_precision`, `variance_reduction`

2. **`results/h3_dac_stat_tests.json`**
   - Statistical test results
   - Contains:
     - `delta_precision_ttest`: Paired t-test results for Δprecision
     - `variance_reduction_ttest`: Paired t-test results for variance reduction
     - `dac_vs_delta_precision_spearman`: Spearman correlation between DAC and Δprecision
     - `dac_vs_variance_reduction_spearman`: Spearman correlation between DAC and variance reduction

## Interpretation

### Paired t-tests
- **Δprecision t-test**: Tests if `precision_learned - precision_det` is significantly different from 0
  - Positive mean with p < 0.05: Learned mapping significantly improves precision
  - Negative mean with p < 0.05: Deterministic mapping significantly improves precision
  
- **Variance reduction t-test**: Tests if `variance_det - variance_learned` is significantly different from 0
  - Positive mean with p < 0.05: Deterministic mapping significantly reduces variance
  - Negative mean with p < 0.05: Learned mapping significantly reduces variance

### Spearman Correlations
- **DAC vs Δprecision**: Tests if higher DAC (better alignment with deterministic mapping) correlates with precision improvement
- **DAC vs variance_reduction**: Tests if higher DAC correlates with variance reduction

These correlations validate Hypothesis H3: that deterministic mapping (higher DAC) yields better operational outcomes (precision and consistency).



























