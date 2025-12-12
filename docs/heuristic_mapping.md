# Heuristic ATT&CK–D3FEND Mapping (H3 Baseline)

## Overview

The heuristic mapping module provides a text-similarity-based learned baseline for comparing against the deterministic ATT&CK→D3FEND lookup table in the H3 experiment. This mapping uses semantic similarity between ATT&CK technique descriptions and D3FEND control descriptions to automatically generate technique-control pairs.

## Purpose

- **H3 Experiment Baseline**: Provides a learned/heuristic mapping to compare against the deterministic lookup
- **Text Similarity**: Uses sentence transformers to compute semantic similarity between attack and defense descriptions
- **Automatic Mapping**: Generates mappings without requiring manual ontology relationships

## Key Features

- Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings
- Falls back to TF-IDF if sentence-transformers is unavailable
- Configurable top-k controls per technique
- Minimum similarity threshold filtering
- Deterministic behavior with seed control

## Usage

### Basic Usage

```bash
python -m aicra.mappings.heuristic_mapping \
    --attack data/ontology/attack_techniques.csv \
    --d3fend data/ontology/d3fend_controls.csv \
    --out data/mappings/learned_mapping.csv
```

### With Custom Parameters

```bash
python -m aicra.mappings.heuristic_mapping \
    --attack data/ontology/attack_techniques.csv \
    --d3fend data/ontology/d3fend_controls.csv \
    --out data/mappings/learned_mapping.csv \
    --top-k 3 \
    --min-similarity 0.40 \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --seed 42
```

### Auto-Discovery

If `--attack` and `--d3fend` are not provided, the module will automatically search for:
- `data/ontology/attack_techniques.csv` or `data/mitre/raw/enterprise-attack.json`
- `data/ontology/d3fend_controls.csv` or `data/mitre/raw/d3fend.csv`

## Input Format

### ATT&CK Techniques CSV

Expected columns:
- `technique_id`: ATT&CK technique ID (e.g., "T1486")
- `name`: Technique name
- `description`: Technique description

### D3FEND Controls CSV

Expected columns:
- `control_id`: D3FEND control ID (e.g., "D3-RA")
- `name`: Control name
- `description`: Control description

## Output Format

The module generates `data/mappings/learned_mapping.csv` with the following columns:

- `technique_id`: ATT&CK technique ID
- `control_id`: D3FEND control ID
- `similarity_score`: Cosine similarity score (0-1)

This format aligns with the H3 evaluation pipeline expectations (`aicra.experiments.h3_evaluation`).

## Algorithm

1. **Load Data**: Load ATT&CK techniques and D3FEND controls with descriptions
2. **Build Text**: Combine name and description: `name + ". " + description`
3. **Compute Embeddings**: Use sentence-transformers to embed all texts
4. **Compute Similarity**: Calculate cosine similarity matrix between techniques and controls
5. **Select Top-K**: For each technique, select top-k controls with highest similarity
6. **Filter by Threshold**: Drop any match with similarity < min_similarity
7. **Output**: Save to CSV with columns: technique_id, control_id, similarity_score

## Configuration

The `HeuristicMappingConfig` dataclass supports:

- `top_k` (default: 3): Number of controls per technique
- `min_similarity` (default: 0.40): Minimum similarity threshold
- `model_name` (default: "sentence-transformers/all-MiniLM-L6-v2"): Sentence transformer model
- `seed` (default: 42): Random seed for deterministic behavior

## Integration with H3 Experiment

The heuristic mapping serves as a baseline for comparing:

- **Mapping Coverage (%)**: Percentage of techniques with at least one control mapping
- **Defense–Attack Consistency (DAC %)**: Consistency metric between heuristic and deterministic mappings
- **Δ Precision (Actionable Positives)**: Difference in precision between heuristic and deterministic
- **Variance Reduction in Risk Scores**: Impact on risk score variance

The output file `data/mappings/learned_mapping.csv` is consumed by the canonical H3 evaluation pipeline (`aicra.experiments.h3_evaluation`) to compute these metrics. See `docs/h3_evaluation_README.md` for details on running the H3 evaluation.

## Dependencies

Required packages (already in requirements):
- `sentence-transformers>=2.2.0`
- `torch>=2.0.0` (CPU-only is fine)
- `scikit-learn>=1.3.0` (for cosine similarity fallback)

## Testing

Run tests with:

```bash
pytest tests/test_heuristic_mapping.py -v
```

The test suite includes:
- Basic mapping functionality
- Minimum similarity threshold filtering
- Top-k limiting
- Deterministic behavior verification
- CSV loading tests

