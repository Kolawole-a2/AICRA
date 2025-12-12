# DAC Embedding Results

## Purpose

This document describes the embedding-based learned mapping pipeline and DAC (Defense-Attack Consistency) evaluation for ATT&CK→D3FEND mappings.

### Learned Mapping (Method B – Embeddings)

The learned mapping uses **sentence-transformers** (specifically `all-MiniLM-L6-v2`) to embed ATT&CK technique names and D3FEND defense names into a shared semantic space. For each ATT&CK technique, the pipeline:

1. Embeds the technique name using the sentence-transformer model
2. Embeds all D3FEND defense names
3. Computes cosine similarity between the technique embedding and all defense embeddings
4. Selects the top-k (default k=3) most similar defenses per attack

**IMPORTANT CONSTRAINT**: The learned mapping does NOT use deterministic ATTACK-DEFENSE pairs as supervision or labels. It uses ONLY the text fields (`attack_name`, `defense_name`) to compute embeddings and similarities. No model is trained to predict `defense_id` from `attack_id` or from deterministic pairs. The learned mapping is based PURELY on semantic similarity in embedding space, not on copying ontology links.

This approach leverages semantic similarity in text embeddings to discover potential ATT&CK→D3FEND relationships heuristically, without requiring explicit training on labeled pairs.

### DAC Definition

**Defense-Attack Consistency (DAC)** measures how well the learned mapping aligns with the deterministic (gold standard) mapping.

For each `attack_id`:
- **D_det** = set of defenses from the deterministic mapping
- **D_learn** = set of defenses from the learned mapping (rank ≤ k)
- **DAC** = |D_det ∩ D_learn| / |D_det|

DAC ranges from 0.0 to 1.0:
- **1.0** = perfect alignment (all deterministic defenses are found in learned mapping)
- **0.0** = no overlap (none of the deterministic defenses are found)

## How to Run

### Step 1: Build Learned Mapping

Generate the learned embedding mapping from the deterministic lookup:

```bash
python -m aicra.mappings.embedding_learned_mapping
```

This script:
- Loads `data/mappings/deterministic_attack_defense_lookup.csv` (ONLY to extract unique attack/defense names)
- Extracts unique attack and defense names (text fields only)
- Embeds them using sentence-transformers
- Computes cosine similarity between all attack-defense pairs
- Generates learned mapping with top-k defenses per attack (based purely on semantic similarity)
- Saves outputs to:
  - `data/mappings/learned_embedding_attack_defense_mapping.csv`
  - `data/mappings/learned_embedding_attack_defense_mapping.parquet`

**Note**: The deterministic lookup is used ONLY as a source of unique attack and defense names. The deterministic ATTACK-DEFENSE pairs themselves are NOT used as supervision.

### Step 2: Evaluate DAC

Compare the learned mapping against the deterministic mapping:

```bash
python -m aicra.metrics.dac_embedding_eval
```

This script:
- Loads both deterministic and learned mappings
- Computes DAC per attack
- Saves results to:
  - `results/dac_embedding_comparison.csv`

The output CSV contains:
- `attack_id`: ATT&CK technique ID
- `n_det_defenses`: Number of defenses in deterministic mapping
- `n_learn_defenses`: Number of defenses in learned mapping
- `n_overlap`: Number of overlapping defenses
- `dac`: DAC score for this attack (overlap / det_defenses)

## Output Files

### Learned Mapping
- **CSV**: `data/mappings/learned_embedding_attack_defense_mapping.csv`
  - Columns: `attack_id`, `defense_id`, `similarity_score`, `rank`, `method`
- **Parquet**: `data/mappings/learned_embedding_attack_defense_mapping.parquet`
  - Same structure as CSV, optimized for programmatic access

### DAC Results
- **CSV**: `results/dac_embedding_comparison.csv`
  - Columns: `attack_id`, `n_det_defenses`, `n_learn_defenses`, `n_overlap`, `dac`

## Interpretation

- **High DAC (close to 1.0)**: The learned mapping successfully identifies most or all of the deterministic defenses for that attack, indicating strong semantic alignment.
- **Low DAC (close to 0.0)**: The learned mapping misses most deterministic defenses, suggesting the embedding-based approach may not capture the relationship well for that attack.

The average DAC across all attacks provides an overall measure of how well the embedding-based approach performs compared to the deterministic mapping.










