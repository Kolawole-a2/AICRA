# Deterministic ATT&CK→D3FEND Lookup and EMBER-2024 Mapping

## Purpose

This project builds a full deterministic ATT&CK→D3FEND lookup table from MITRE data, then maps EMBER-2024 samples onto that ontology. The resulting mappings serve as a deterministic baseline for:

- Defense–Attack Consistency (DAC) metric evaluation
- Comparing learned mappings against deterministic mappings
- Validation datasets for consistency checking

## Pipeline

### Step 1: Fetch Full MITRE Data

```bash
python -m aicra.mitre.fetch_full_mitre
```

Downloads and processes:
- MITRE ATT&CK Enterprise STIX bundle
- D3FEND catalog and ontology data

Outputs:
- `data/mitre/attack_catalog.csv` - ATT&CK techniques
- `data/mitre/defense_catalog.csv` - D3FEND defenses
- `data/mitre/attack_defense_edges.csv` - ATT&CK↔D3FEND relationships

### Step 2: Build Deterministic Lookup

```bash
python -m aicra.mappings.deterministic_builder
```

Joins the catalogs and edges to create a deterministic ATT&CK→D3FEND lookup table.

Outputs:
- `data/mappings/deterministic_attack_defense_lookup.csv`
- `data/mappings/deterministic_attack_defense_lookup.parquet`

### Step 3: Map EMBER-2024 Samples

```bash
python -m aicra.mappings.deterministic_sample_level
```

This module:
1. **Automatically locates** the EMBER-2024 dataset by searching:
   - `../` (parent directory)
   - `../AICRA/` (sibling AICRA project)
   - Recursively searches for files containing "ember" and "2024" with extensions `.parquet` or `.csv`
   - Prefers `.parquet` over `.csv` if multiple matches found

2. **Loads EMBER data** from the discovered file

3. **Enriches with ATT&CK mappings** using `data/mitre/family_attack_map.csv`:
   - Maps malware families to ATT&CK techniques
   - Adds `attack_id` and `confidence` columns

4. **Joins with deterministic lookup** to create sample-level mappings

5. **Saves results**:
   - `data/mappings/ember_deterministic_sample_mapping.csv`
   - `data/mappings/ember_deterministic_sample_mapping.parquet`

## EMBER Dataset Discovery

The `find_ember_2024()` function automatically searches for EMBER-2024 files in likely locations:

- Default search paths: `Path("..")`, `Path("..") / "AICRA"`
- Recursively walks these directories
- Looks for filenames containing:
  - "ember" (case-insensitive)
  - "2024"
  - Extensions: `.parquet` or `.csv`
- Prefers `.parquet` over `.csv` if multiple files found

The function logs the chosen file path when found, or a warning if nothing is found.

## Final Outputs

All final outputs are saved under `data/mappings/`:

1. **`deterministic_attack_defense_lookup.csv/.parquet`**
   - Full deterministic ATT&CK→D3FEND mapping from MITRE sources
   - Columns: `attack_id`, `attack_name`, `defense_id`, `defense_name`, `is_correct`, `source`

2. **`ember_deterministic_sample_mapping.csv/.parquet`**
   - Sample-level mapping of EMBER-2024 samples to ATT&CK→D3FEND
   - Columns: `sha256` (or sample_id), `family`, `attack_id`, `attack_name`, `defense_id`, `defense_name`, `is_correct`, `ransomware_weight`

## Integration into AICRA

These files can be copied directly into the main AICRA project's `data/mappings/` folder and used as:

- **Gold-standard baseline** for DAC metric evaluation
- **Reference dataset** for comparing learned mappings against deterministic mappings
- **Validation dataset** for consistency checking

No modifications are required - the files are ready to use as-is.

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



