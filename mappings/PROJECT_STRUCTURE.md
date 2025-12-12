# AICRA Project Structure

This document describes the organized structure of the AICRA project for H3 experiments.

## Directory Structure

```
aicra/                      # main package
  mapping/
    deterministic_lookup.py # generator for deterministic table
    heuristic_mapping.py    # learned/text-similarity mapping
    __init__.py             # package exports

  experiments/
    h3_validation.py        # full DAC + precision/variance comparison
    h3_learned_mapping_eval.py  # existing H3 evaluation
    h3_prepare_metrics.py   # existing metrics preparation
    h3_stat_tests.py        # existing statistical tests
    __init__.py

data/
  ontology/
    attack_techniques.csv       # ATT&CK techniques (id, name, description)
    d3fend_controls.csv         # D3FEND controls (id, name, description)
    d3fend_reference_pairs.csv  # canonical ATT&CK–D3FEND pairs (optional)

  mappings/
    deterministic_lookup.csv    # hand-curated mapping (baseline for DAC)
    learned_mapping.csv         # heuristic/text-similarity mapping (H3 baseline)

results/
  H3_comparison/
    H3_results.json
    H3_summary.md
    plots/
      coverage.png
      consistency.png
      precision.png
      variance_reduction.png

scripts/
  check_mappings_sanity.py      # confirms both mappings exist, no overwrite
  compare_mappings_summary.py   # quick coverage/DAC-style summary
```

## Key Files

### Mapping Modules

- **`aicra/mapping/deterministic_lookup.py`**: Functions to load and save deterministic lookup tables
- **`aicra/mapping/heuristic_mapping.py`**: Text-similarity based heuristic mapping using sentence transformers

### Experiment Modules

- **`aicra/experiments/h3_validation.py`**: Full H3 validation comparing deterministic vs learned mappings
  - Computes coverage, DAC, precision delta, and variance reduction
  - Generates JSON results and markdown summary

### Scripts

- **`scripts/check_mappings_sanity.py`**: Validates that both mapping files exist and are different
- **`scripts/compare_mappings_summary.py`**: Quick comparison summary with coverage and DAC metrics

## Usage Examples

### Generate Heuristic Mapping
```bash
python -m aicra.mapping.heuristic_mapping \
    --attack data/ontology/attack_techniques.csv \
    --d3fend data/ontology/d3fend_controls.csv \
    --out data/mappings/learned_mapping.csv
```

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

## File Locations

- **Deterministic mapping**: `data/mappings/deterministic_lookup.csv` (or `deterministic_attack_defense_lookup.csv`)
- **Learned mapping**: `data/mappings/learned_mapping.csv`
- **Results**: `results/H3_comparison/H3_results.json` and `H3_summary.md`

