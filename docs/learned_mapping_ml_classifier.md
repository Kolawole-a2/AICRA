# Learned Mapping ML Classifier

## Overview

The deterministic mapping is constructed from MITRE ATT&CK–D3FEND ontology and treated as a gold-standard reference for the DAC metric. This mapping represents the authoritative, curated relationships between ATT&CK techniques and D3FEND defensive controls, serving as the ground truth against which learned mappings are evaluated.

The learned mapping uses a multi-label ML classifier that predicts D3FEND defenses from ATT&CK technique text (attack names/descriptions). The deterministic mapping provides training labels, but the classifier must infer mappings based on language features rather than ontology edges at inference time. This approach enables the system to generalize beyond the explicit ontology relationships and discover potential mappings through semantic similarity in text descriptions.

The learned mapping produces, for each attack_id, a ranked list of likely defenses. This is used to compute Defense–Attack Consistency (DAC) against the deterministic mapping and to assess how well a data-driven approach can approximate the ontology. The DAC metric measures the proportion of learned mappings that align with the deterministic gold standard, providing a quantitative assessment of the classifier's ability to replicate ontology-driven relationships through text-based inference.

## How to Run

### 1. Train ML classifier and build learned mapping

```bash
python -m aicra.mappings.learned_ml_mapping
```

This will:
- Load the deterministic mapping from `data/mappings/deterministic_attack_defense_lookup.csv`
- Train a multi-label classifier using TF-IDF features from attack names/descriptions
- Generate learned mappings with top-k defenses per attack
- Save results to:
  - `data/mappings/learned_attack_defense_mapping.csv`
  - `data/mappings/learned_attack_defense_mapping.parquet`

### 2. Compute DAC between deterministic and learned mapping

```bash
python -m aicra.metrics.dac
```

This will:
- Load both deterministic and learned mappings
- Compute DAC metrics (overlap, precision, etc.)
- Log summary statistics

### 3. Run full evaluation experiment (optional)

```bash
python -m aicra.experiments.h3_learned_mapping_eval
```

This will:
- Run the complete evaluation pipeline
- Save detailed metrics to `results/h3_dac_learned_vs_deterministic.json`

## Reproducibility

All training runs fix `RANDOM_SEED=42` for reproducibility.




























