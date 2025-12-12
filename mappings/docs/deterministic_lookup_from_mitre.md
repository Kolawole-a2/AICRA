# Deterministic ATT&CK→D3FEND Lookup Table

## Purpose

This project builds a deterministic ATT&CK→D3FEND lookup table using full MITRE ATT&CK Enterprise and D3FEND data.

The lookup table will later be used in AICRA as a gold-standard for the Defense–Attack Consistency (DAC) metric. This deterministic mapping provides a ground truth baseline against which learned mappings and consistency metrics can be evaluated.

## Pipeline Overview

### `aicra.mitre.fetch_full_mitre`

Downloads ATT&CK STIX and D3FEND data from official MITRE sources:

- **ATT&CK Enterprise STIX**: Downloaded from the MITRE GitHub repository (`mitre-attack/attack-stix-data`)
- **D3FEND Catalog**: Downloaded from the MITRE D3FEND GitHub repository
- **D3FEND JSON**: Downloaded for ontology-based mapping extraction

Builds three core CSV files:

- `data/mitre/attack_catalog.csv` - All ATT&CK Enterprise techniques with IDs and names
- `data/mitre/defense_catalog.csv` - All D3FEND defensive techniques with IDs and names
- `data/mitre/attack_defense_edges.csv` - Relationships between ATT&CK techniques and D3FEND defenses

### `aicra.mappings.deterministic_builder`

Joins the catalogs and edges to produce a deterministic lookup table:

- Loads the three CSV files created by `fetch_full_mitre`
- Merges edges with attack catalog on `attack_id`
- Merges the result with defense catalog on `defense_id`
- Creates a normalized lookup table with columns:
  - `attack_id` - ATT&CK technique ID (e.g., "T1005")
  - `attack_name` - ATT&CK technique name
  - `defense_id` - D3FEND defense ID (e.g., "D3-AM")
  - `defense_name` - D3FEND defense name
  - `is_correct` - Always 1 (indicating this is a verified mapping)
  - `source` - Source of the mapping (e.g., "d3fend_json", "mitre_d3fend")

Outputs:

- `data/mappings/deterministic_attack_defense_lookup.csv` - Human-readable CSV format
- `data/mappings/deterministic_attack_defense_lookup.parquet` - Efficient binary format for programmatic use

## How to Run

### Step 1: Fetch full MITRE data and build catalogs

```bash
python -m aicra.mitre.fetch_full_mitre
```

This will:
- Download ATT&CK Enterprise STIX bundle to `data/mitre/raw/enterprise-attack.json`
- Download D3FEND catalog to `data/mitre/raw/d3fend.csv`
- Download D3FEND JSON to `data/mitre/raw/d3fend.json`
- Extract and build the three catalog CSV files

### Step 2: Build deterministic ATT&CK→D3FEND lookup table

```bash
python -m aicra.mappings.deterministic_builder
```

This will:
- Load the catalogs and edges created in Step 1
- Join them to create the deterministic lookup table
- Save both CSV and Parquet formats to `data/mappings/`

### Integration into AICRA

After running these commands, you can copy the generated lookup files:

```bash
cp data/mappings/deterministic_attack_defense_lookup.* /path/to/aicra/data/mappings/
```

The lookup table can then be used in AICRA as:
- A gold-standard baseline for DAC metric evaluation
- A reference for comparing learned mappings against deterministic mappings
- A validation dataset for consistency checking

## Requirements

- Python 3.10+
- Required packages:
  - `requests` - For downloading MITRE data
  - `pandas` - For data manipulation
  - `stix2` - For parsing ATT&CK STIX data
  - `pyarrow` - For Parquet file support (optional but recommended)

Install with:
```bash
pip install requests pandas stix2 pyarrow
```

