# AICRA Project Structure

This document describes the organized structure of the AICRA project.

## Directory Structure

```
aicra/                      # main package
  mapping/
    deterministic_lookup.py # generator for deterministic table
    heuristic_mapping.py    # learned/text-similarity mapping
    __init__.py             # package exports

  experiments/
    h3_validation.py        # full DAC + precision/variance comparison
    temporal_eval.py        # time-ordered / out-of-sample evaluation
    threshold_analysis.py   # expected loss / cost-aware thresholds
    adversarial_sanity.py   # simple robustness checks
    __init__.py

data/
  ontology/
    attack_techniques.csv       # ATT&CK techniques (id, name, description)
    d3fend_controls.csv         # D3FEND controls (id, name, description)
    d3fend_reference_pairs.csv # canonical ATT&CK–D3FEND pairs

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

## Usage

### Generate Deterministic Mapping
```bash
python -m aicra.mapping.deterministic_lookup
```

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

### Check Mappings Sanity
```bash
python scripts/check_mappings_sanity.py
```

### Compare Mappings
```bash
python scripts/compare_mappings_summary.py
```

