# Deterministic ATT&CK→D3FEND Lookup Table for Ransomware

This project builds a **deterministic ATT&CK→D3FEND lookup table** specifically for ransomware attacks, using full MITRE ATT&CK Enterprise and D3FEND data. The lookup table provides a gold-standard mapping that covers all ransomware families in the EMBER-2024 dataset.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Usage](#usage)
- [Output Files](#output-files)
- [Validation](#validation)
- [File Descriptions](#file-descriptions)

## Overview

This standalone Python project:

1. **Downloads and parses** full MITRE ATT&CK Enterprise STIX and D3FEND data
2. **Builds catalogs** of ATT&CK techniques and D3FEND defenses
3. **Extracts relationships** between ATT&CK techniques and D3FEND defenses
4. **Creates a deterministic lookup table** for ransomware-specific attack-defense mappings
5. **Maps EMBER-2024 ransomware families** to ATT&CK techniques
6. **Validates coverage** to ensure all ransomware attacks are covered

The resulting lookup table is **ransomware-specific** and uses **technique-specific defense mappings** (not all-to-all), ensuring accuracy and relevance.

## Project Structure

```
mappings/
├── aicra/
│   ├── mitre/
│   │   ├── __init__.py
│   │   └── fetch_full_mitre.py          # Downloads and parses MITRE data
│   └── mappings/
│       ├── __init__.py
│       ├── deterministic_builder.py      # Builds deterministic lookup table
│       ├── ember_family_enrichment.py    # Maps families to ATT&CK techniques
│       └── deterministic_sample_level.py # Creates sample-level mappings
├── data/
│   ├── mitre/
│   │   ├── raw/                          # Raw downloaded MITRE data
│   │   │   ├── enterprise-attack.json
│   │   │   ├── d3fend.csv
│   │   │   └── d3fend.json
│   │   ├── attack_catalog.csv            # All ATT&CK techniques
│   │   ├── defense_catalog.csv           # All D3FEND defenses
│   │   ├── attack_defense_edges.csv      # ATT&CK↔D3FEND relationships
│   │   └── family_attack_map.csv         # Family→ATT&CK mappings (48 families)
│   └── mappings/
│       ├── deterministic_attack_defense_lookup.csv    # Final lookup table
│       ├── deterministic_attack_defense_lookup.parquet
│       ├── ember_deterministic_sample_mapping.csv      # Sample-level mappings
│       ├── ember_deterministic_sample_mapping.parquet
│       └── validation_report.txt         # Validation results
├── docs/
│   ├── deterministic_lookup_from_mitre.md
│   └── deterministic_lookup_and_ember_mapping.md
├── validate_ember_coverage.py            # Validation script
└── README.md                              # This file
```

## Requirements

- **Python 3.10+**
- Required packages:
  - `requests` - For downloading MITRE data
  - `pandas` - For data manipulation
  - `stix2` - For parsing STIX bundles
  - `pyarrow` - For Parquet file I/O
  - `numpy` - For numerical operations

## Installation

1. **Clone or navigate to the project directory**

2. **Install required packages:**
   ```bash
   pip install requests pandas stix2 pyarrow numpy
   ```

3. **Verify installation:**
   ```bash
   python -c "import requests, pandas, stix2, pyarrow, numpy; print('All packages installed successfully')"
   ```

## Pipeline Overview

The project follows a three-step pipeline:

```
Step 1: Fetch MITRE Data
  ↓
  Downloads ATT&CK STIX and D3FEND data
  Builds: attack_catalog.csv, defense_catalog.csv, attack_defense_edges.csv

Step 2: Build Deterministic Lookup
  ↓
  Joins catalogs and edges
  Filters to ransomware-specific techniques
  Creates technique-specific defense mappings
  Outputs: deterministic_attack_defense_lookup.csv/.parquet

Step 3: Map EMBER-2024 (Optional)
  ↓
  Locates EMBER-2024 dataset
  Enriches with family→ATT&CK mappings
  Joins with deterministic lookup
  Outputs: ember_deterministic_sample_mapping.csv/.parquet
```

## Usage

### Step 1: Fetch Full MITRE Data

Downloads and processes MITRE ATT&CK Enterprise STIX and D3FEND data:

```bash
python -m aicra.mitre.fetch_full_mitre
```

**What it does:**
- Downloads ATT&CK Enterprise STIX bundle from MITRE GitHub
- Downloads D3FEND catalog and ontology JSON
- Extracts ATT&CK techniques (attack-pattern objects)
- Extracts D3FEND defenses
- Builds ransomware-specific ATT&CK↔D3FEND edge mappings

**Outputs:**
- `data/mitre/attack_catalog.csv` - All ATT&CK techniques (835 techniques)
- `data/mitre/defense_catalog.csv` - All D3FEND defenses (248 defenses)
- `data/mitre/attack_defense_edges.csv` - Ransomware-specific edges (173 edges)

### Step 2: Build Deterministic Lookup Table

Creates the final deterministic ATT&CK→D3FEND lookup table:

```bash
python -m aicra.mappings.deterministic_builder
```

**What it does:**
- Loads attack catalog, defense catalog, and edges
- Merges to create complete mappings
- Filters to ransomware-related ATT&CK techniques only
- Ensures all defense names are populated
- Creates technique-specific defense mappings (not all-to-all)

**Outputs:**
- `data/mappings/deterministic_attack_defense_lookup.csv` - Human-readable CSV
- `data/mappings/deterministic_attack_defense_lookup.parquet` - Efficient binary format

**Key Features:**
- **Ransomware-specific**: Only includes techniques used by ransomware families
- **Technique-specific**: Each technique mapped to semantically relevant defenses
- **Complete defense names**: All defense names clearly indicated
- **173 mappings** covering 46 unique techniques and 9 unique defenses

### Step 3: Map EMBER-2024 Samples (Optional)

Maps EMBER-2024 ransomware samples to the deterministic lookup:

```bash
python -m aicra.mappings.deterministic_sample_level
```

**What it does:**
- Automatically locates EMBER-2024 dataset in AICRA project folder
- Loads EMBER data
- Enriches with family→ATT&CK mappings from `family_attack_map.csv`
- Joins with deterministic lookup to create sample-level mappings

**Outputs:**
- `data/mappings/ember_deterministic_sample_mapping.csv`
- `data/mappings/ember_deterministic_sample_mapping.parquet`

**Note:** This step requires the EMBER-2024 dataset to be accessible in a sibling AICRA project folder.

## Output Files

### Core Outputs

1. **`data/mappings/deterministic_attack_defense_lookup.csv`**
   - **Purpose**: Main deterministic lookup table
   - **Format**: CSV with columns:
     - `attack_id` - ATT&CK technique ID (e.g., "T1486")
     - `attack_name` - ATT&CK technique name
     - `defense_id` - D3FEND defense ID (e.g., "D3-RA")
     - `defense_name` - D3FEND defense name (clearly indicated)
     - `is_correct` - Always 1 (verified mapping)
     - `source` - Source of mapping ("d3fend_ransomware_specific")
   - **Statistics**: 173 rows, 46 unique techniques, 9 unique defenses

2. **`data/mappings/deterministic_attack_defense_lookup.parquet`**
   - Same data as CSV, in efficient binary format for programmatic use

3. **`data/mitre/family_attack_map.csv`**
   - **Purpose**: Maps 48 ransomware families to ATT&CK techniques
   - **Format**: CSV with columns:
     - `family_name` - Ransomware family name
     - `attack_id` - ATT&CK technique ID
     - `confidence` - Mapping confidence (high/medium)
   - **Statistics**: 120 mappings across 48 families

### Intermediate Files

- `data/mitre/attack_catalog.csv` - All ATT&CK Enterprise techniques
- `data/mitre/defense_catalog.csv` - All D3FEND defenses
- `data/mitre/attack_defense_edges.csv` - ATT&CK↔D3FEND relationships

## Validation

### Running Validation

Validate that the deterministic lookup table covers all ransomware attacks in EMBER-2024:

```bash
python validate_ember_coverage.py
```

### What It Validates

1. **Family Coverage**: All 48 ransomware families from EMBER are mapped
2. **Attack ID Coverage**: All attack IDs from family_map are in deterministic lookup
3. **Defense Name Completeness**: All defense names are populated
4. **Sample-level Coverage**: (If EMBER dataset accessible) Sample-level validation

### Validation Report

The script generates a detailed report at:
- `data/mappings/validation_report.txt`

**Example Validation Results:**
```
[PASS] VALIDATION PASSED

The deterministic lookup table covers all ransomware attacks in EMBER-2024:
  - All 48 ransomware families are mapped to ATT&CK techniques
  - All 11 attack IDs from family_map are covered in deterministic lookup
  - All defense names are clearly indicated
  - Technique-specific defense mappings ensure accuracy
```

### Validation Statistics

- **Family Coverage**: 48/48 (100%)
- **Attack ID Coverage**: 11/11 (100%)
- **Total Mappings**: 173
- **Unique Techniques**: 46
- **Unique Defenses**: 9
- **All Defense Names Populated**: Yes

## File Descriptions

### Core Modules

#### `aicra/mitre/fetch_full_mitre.py`
Downloads and processes MITRE data:
- Downloads ATT&CK Enterprise STIX bundle
- Downloads D3FEND catalog and JSON ontology
- Extracts ATT&CK techniques from STIX
- Extracts D3FEND defenses
- Builds ransomware-specific technique-defense edges

**Key Functions:**
- `download_file()` - Downloads files from URLs
- `build_attack_catalog()` - Extracts ATT&CK techniques from STIX
- `build_defense_catalog()` - Extracts D3FEND defenses
- `build_attack_defense_edges_from_json()` - Creates technique-specific mappings
- `build_all_mitre_csvs()` - Orchestrates the full process

#### `aicra/mappings/deterministic_builder.py`
Builds the deterministic lookup table:
- Loads attack catalog, defense catalog, and edges
- Merges to create complete mappings
- Filters to ransomware-specific techniques
- Ensures defense names are populated

**Key Functions:**
- `load_attack_catalog()` - Loads ATT&CK catalog
- `load_defense_catalog()` - Loads D3FEND catalog
- `load_attack_defense_edges()` - Loads edges
- `build_deterministic_lookup()` - Creates final lookup table
- `save_deterministic_lookup()` - Saves to CSV and Parquet

#### `aicra/mappings/ember_family_enrichment.py`
Maps malware families to ATT&CK techniques:
- Loads family→ATT&CK mapping
- Enriches EMBER DataFrame with attack IDs

**Key Functions:**
- `load_family_attack_map()` - Loads family mapping
- `enrich_ember_with_attacks()` - Adds attack IDs to EMBER data

#### `aicra/mappings/deterministic_sample_level.py`
Creates sample-level mappings:
- Locates EMBER-2024 dataset
- Loads and enriches EMBER data
- Joins with deterministic lookup

**Key Functions:**
- `find_ember_2024()` - Locates EMBER dataset
- `load_ember_2024()` - Loads EMBER data
- `build_sample_level_deterministic_mapping()` - Creates sample mappings

### Ransomware Techniques Covered

The deterministic lookup includes the following ransomware-related ATT&CK techniques:

**Primary Impact Techniques:**
- `T1486` - Data Encrypted for Impact (primary ransomware technique)
- `T1490` - Inhibit System Recovery
- `T1485` - Data Destruction
- `T1487` - Disk Structure Wipe
- `T1488` - Disk Content Wipe
- `T1489` - Service Stop

**Supporting Techniques:**
- `T1055` - Process Injection (and sub-techniques)
- `T1070` - Indicator Removal (and sub-techniques)
- `T1021` - Remote Services (and sub-techniques)
- `T1041` - Exfiltration Over C2 Channel
- `T1496` - Resource Hijacking (and sub-techniques)

### D3FEND Defenses Used

The lookup table maps to 9 semantically relevant D3FEND defenses:

1. **D3-RA** - Restore Access
2. **D3-DO** - Decoy Object
3. **D3-DE** - Decoy Environment
4. **D3-FA** - File Analysis
5. **D3-PM** - Platform Monitoring
6. **D3-UBA** - User Behavior Analysis
7. **D3-AI** - Asset Inventory
8. **D3-PE** - Process Eviction
9. **D3-AMED** - Access Mediation

## Ransomware Families Covered

The `family_attack_map.csv` covers all 48 ransomware families from EMBER-2024:

- abransom, aconti, akira, azovwiper, bccrypt, bianlian, blackmatter, clop, conti, cryakl, cryfile, crypmodng, crypren, cryptagent, cryptedloader, cryptic, cryptonote, cryptowall, cryptran, cryrar, detplock, dotransom, fakeransom, gandcrab, genkryptikagen, hive, hmblocker, ibashade, injectorcrypt, lockbit, lockergoga, mbrlock, mountlocker, polycrypt, pornoblocker, prestige, pscrypt, pycrypter, ragnarlocker, ransomrevil, ransomsnatch, revil, rhysida, shade, silentcryptominer, teslacrypt, virlock, wannacry

## Integration with AICRA

The generated lookup files can be copied into your main AICRA repository:

```bash
# Copy lookup table
cp data/mappings/deterministic_attack_defense_lookup.* /path/to/AICRA/data/mappings/

# Copy family mapping
cp data/mitre/family_attack_map.csv /path/to/AICRA/data/mitre/
```

These files serve as:
- **Gold-standard baseline** for Defense–Attack Consistency (DAC) metric
- **Deterministic reference** for comparing learned vs deterministic mappings
- **Validation dataset** for consistency checking

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing packages with `pip install <package>`

2. **FileNotFoundError for EMBER**: 
   - Ensure EMBER-2024 dataset is in a sibling AICRA project folder
   - Or manually specify the path in the code

3. **Download Failures**:
   - Check internet connection
   - Verify MITRE URLs are accessible
   - Raw files are cached in `data/mitre/raw/` after first download

4. **Encoding Errors**:
   - Ensure files are saved with UTF-8 encoding
   - On Windows, use `encoding="utf-8"` when opening files

## License

This project is part of the AICRA research framework. Please refer to the main AICRA repository for licensing information.

## Citation

If you use this deterministic lookup table in your research, please cite:

- MITRE ATT&CK: https://attack.mitre.org/
- MITRE D3FEND: https://d3fend.mitre.org/
- EMBER-2024 Dataset: [Original EMBER paper citation]

## Contact

For questions or issues, please refer to the main AICRA project repository.

---

**Last Updated**: 2025-12-04  
**Python Version**: 3.10+  
**Validation Status**: ✓ All ransomware attacks covered

