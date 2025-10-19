# AI Cyber Risk Advisor (AICRA)

[![CI](https://github.com/aicra/aicra/workflows/CI/badge.svg)](https://github.com/aicra/aicra/actions)
[![codecov](https://codecov.io/gh/aicra/aicra/branch/main/graph/badge.svg)](https://codecov.io/gh/aicra/aicra)
[![PyPI version](https://badge.fury.io/py/aicra.svg)](https://badge.fury.io/py/aicra)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Overview

AICRA (AI Cyber Risk Advisor) is a machine learning-based cyber risk assessment system focused on endpoint ransomware detection in banking environments. It provides reproducible, auditable risk assessments with cost-sensitive decision making.

### Key Features

- **Ransomware Detection**: LightGBM-based binary classification with calibration
- **Cost-Sensitive Thresholding**: Optimal decision thresholds based on business impact
- **Risk Register**: CSV/JSON output with MITRE ATT&CK to D3FEND mappings
- **Drift Detection**: Evidently-based data and performance drift monitoring
- **Reproducible Pipeline**: Deterministic builds with MLflow tracking
- **Quality Gates**: Pre-commit hooks, type checking, security auditing

### Artifacts

- Console metrics (AUROC, PR-AUC, Brier, ECE, Lift@10%, confusion matrix)
- Cost-sensitive threshold selection with configurable impact costs
- Cyber Risk Advisor Register with prescriptive controls
- Policy JSON for auditable decision thresholds
- Drift reports and reliability diagrams

### Automatic Result Archiving

AICRA automatically archives all key artifacts from each run to timestamped folders for easy tracking and reporting:

**Directory Structure:**
```
results/
├── run_2025-10-17_2030/          # Timestamped run folder
│   ├── metrics.json               # Performance metrics
│   ├── policy.json               # Decision thresholds
│   ├── risk_register.csv         # Risk assessment results
│   ├── roc.png                   # ROC curve plot
│   ├── pr.png                    # Precision-Recall curve
│   ├── reliability.png            # Reliability diagram
│   ├── confusion_at_ops.png      # Confusion matrix
│   ├── lift_curve.png            # Lift curve (if available)
│   ├── bagged_lightgbm.joblib    # Trained model
│   ├── calibrator.joblib         # Calibration model
│   └── impact.csv                # Impact table context
├── run_2025-10-17_2045_smoke/    # Smoke test run
└── versions_log.csv              # Summary of all runs
```

**Features:**
- **Automatic Archiving**: Every run (`aicra register`, `aicra smoke`, `aicra run-all`) automatically creates a timestamped folder
- **Version Tracking**: `versions_log.csv` maintains a summary of all experiments with metrics
- **Git-Ready**: Archived results can be safely committed to GitHub for collaboration
- **Manual Control**: Use `aicra archive-results` to manually archive current artifacts

**Example Usage:**
```bash
# Run pipeline (automatically archives results)
aicra run-all

# Manually archive current results
aicra archive-results --run-name "experiment_v2"

# List recent runs
aicra list-runs --limit 5

# View versions log
cat results/versions_log.csv
```

**Sample Metrics from versions_log.csv:**
```csv
timestamp,run_name,dataset,model,AUROC,PR-AUC,ECE,Lift@5,folder_path
2025-10-17 20:30:15,run_2025-10-17_2030,ember2024,bagged_lightgbm,0.8234,0.1567,0.0892,1.45,results/run_2025-10-17_2030
2025-10-17 20:45:22,run_2025-10-17_2045_smoke,synthetic,mock_model,0.6789,0.1234,0.1456,1.23,results/run_2025-10-17_2045_smoke
```

## Full EMBER Debug Mode

The AICRA pipeline includes a comprehensive debug mode for the full EMBER-2024 phase to diagnose and fix common causes of AUROC ≈ 0.50 on large datasets.

### Debug Mode Features

- **Data Loading Validation**: Verifies row counts, label balance, and feature integrity
- **Split Integrity Checks**: Detects data leakage and validates time-ordered or stratified splits
- **Feature Analysis**: Identifies and removes constant/near-constant features
- **LightGBM Retuning**: Optimizes parameters for large datasets with early stopping
- **Comprehensive Reporting**: Generates detailed debug reports with probable causes and recommendations

### Usage

```bash
# Run full EMBER-2024 with debug mode
aicra run-test --phase full --data-dir data/ember2024 --seed 42 --debug --time-split

# Debug mode parameters:
# --debug: Enable deep diagnostics and verbose logs
# --time-split: Use time-ordered split if timestamp column exists
# --data-dir: Path to EMBER-2024 JSONL files (default: data/ember2024)
# --seed: Random seed for reproducibility (default: 42)
```

### Debug Artifacts

When debug mode is enabled, the following artifacts are generated:

- **`artifacts/debug_full_report.json`**: Comprehensive debug report with metrics, causes, and recommendations
- **`artifacts/debug_full_report.md`**: Human-readable Markdown version of the debug report
- **`artifacts/debug_full_data_summary.json`**: Data loading validation results
- **`artifacts/debug_full_split_time.json`**: Time-ordered split validation (if `--time-split` used)
- **`artifacts/debug_full_split_stratified.json`**: Stratified split validation (default)
- **`artifacts/leakage_check_full.csv`**: Data leakage detection results
- **`artifacts/removed_features_full.csv`**: List of constant/near-constant features removed
- **`artifacts/feature_importance_full.csv`**: Top features by importance

### Troubleshooting Low AUROC

If AUROC ≈ 0.50, the debug report will identify probable causes:

1. **Single-class labels**: Only one class found in the dataset
2. **Extreme class imbalance**: Prevalence < 1% or > 99%
3. **Insufficient informative features**: Too many constant features after cleaning
4. **Data leakage**: Overlapping IDs between train and test sets
5. **Model underfitting**: Try reducing `min_data_in_leaf` or increasing `num_leaves`
6. **Model overfitting**: Try increasing `min_data_in_leaf` or reducing `num_leaves`

### LightGBM Parameter Tuning

Debug mode automatically optimizes LightGBM parameters for large datasets:

- **num_leaves**: 127 for >100 features, 64 otherwise
- **learning_rate**: 0.05
- **n_estimators**: 3000 with early stopping (200 rounds)
- **min_data_in_leaf**: 200 (adjustable based on dataset size)
- **feature_fraction**: 0.8
- **bagging_fraction**: 0.8

### Configuration

Debug parameters can be adjusted in `aicra/config.py`:

```python
# Debug configuration
max_unmapped_rate: float = 0.05  # Maximum allowed unmapped rate
mapping_cache_size: int = 1000   # LRU cache size for mapping operations
mapping_timeout_seconds: int = 30 # Timeout for mapping operations
```

### Example Debug Output

```
🔍 DEBUG: Validating data loading...
🔍 DEBUG: Validating split integrity...
🔍 DEBUG: Retuning LightGBM for large data...
🔍 DEBUG: Generating debug report...

⚠️  Full run AUROC low (0.523) — potential data/param issue.
📋 See artifacts/debug_full_report.json for details.
🔍 Probable causes:
  • Extreme class imbalance
  • Model underfitting - try reducing min_data_in_leaf or increasing num_leaves

📋 Debug report saved to: artifacts/debug_full_report.json
```

## Lookup Coverage & Unmapped Report

The AICRA pipeline includes comprehensive lookup coverage tracking and fail-fast mechanisms to ensure high-quality mapping between malware families, ATT&CK techniques, and D3FEND controls.

### Coverage Metrics

The system tracks three key coverage metrics:

1. **Alias-to-Family Coverage**: Percentage of raw malware family names successfully mapped to canonical families
2. **Family-to-Attack Coverage**: Percentage of canonical families with mapped ATT&CK techniques  
3. **Attack-to-D3FEND Coverage**: Percentage of ATT&CK techniques with mapped D3FEND controls

### Coverage Thresholds

- **Default Maximum Unmapped Rate**: 5% (`max_unmapped_rate = 0.05`)
- **Fail-Fast Behavior**: Pipeline exits with non-zero status if alias-to-family coverage falls below threshold
- **Configurable**: Thresholds can be adjusted in `aicra/config.py`

### Coverage Reports

For each test phase, the system generates:

- **`artifacts/mapping_coverage_{phase}.json`**: Detailed coverage metrics and statistics
- **`artifacts/unmapped_report_{phase}.csv`**: List of unmapped items with occurrence counts

### Validation Commands

```bash
# Validate lookup coverage for specific phase
aicra validate-lookups --phase small_ember
aicra validate-lookups --phase full

# Expand lookups from MITRE data
aicra expand-lookups --from-mitre /path/to/mitre/data --dry-run
aicra expand-lookups --from-mitre /path/to/mitre/data
```

### Curating Lookup Files

To improve coverage:

1. **Review Unmapped Reports**: Check `artifacts/unmapped_report_{phase}.csv` for frequently unmapped items
2. **Update Canonical Families**: Add new mappings to `data/lookups/canonical_families.yaml`
3. **Add ATT&CK Mappings**: Update `data/lookups/family_to_attack.yaml` with technique mappings
4. **Add D3FEND Controls**: Update `data/lookups/attack_to_d3fend.yaml` with control mappings
5. **Re-run Validation**: Use `aicra validate-lookups` to verify improvements

### Performance Features

- **Vectorized Processing**: Uses pandas for efficient batch operations
- **LRU Caching**: Caches normalized results for repeated lookups
- **Pre-compiled Patterns**: Regex patterns compiled once for performance
- **Memory Efficient**: Handles 100k+ samples with reasonable memory usage

### Example Coverage Report

```json
{
  "phase": "small_ember",
  "coverage_metrics": {
    "alias_to_family_coverage": 0.847,
    "family_to_attack_coverage": 0.923,
    "attack_to_d3fend_coverage": 0.891
  },
  "coverage_stats": {
    "alias_to_family": {
      "mapped": 8470,
      "total": 10000,
      "unmapped": ["unknown_family_1", "unknown_family_2", ...]
    }
  }
}
```

## Quickstart

### Installation

```bash
# Clone repository
git clone https://github.com/aicra/aicra.git
cd aicra

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Smoke Test (Pre-Expansion)

Before scaling to full datasets, run the smoke test to validate the entire pipeline:

```bash
# Run end-to-end smoke test
make smoke
# or
aicra smoke

# Dry run (no training, just validation)
aicra smoke --dry-run
```

**Smoke Test Purpose**: Validates the complete AICRA pipeline on tiny synthetic data to ensure all components work together before scaling to production datasets.

**PASS Criteria**:
- AUROC ≥ 0.70
- PR-AUC > 0.05 (above prevalence)
- Brier Score ≤ 0.25
- Expected Calibration Error (ECE) ≤ 0.15

## Automated Test Phases

AICRA supports three sequential test phases for comprehensive validation:

### Phase 1: Smoke Test (Synthetic Data)
Fast validation using synthetic data - no external dependencies required:

```bash
# Run smoke test with synthetic data
aicra run-test --phase smoke
```

**Behavior**: Uses LogisticRegression on synthetic data for quick validation
**Data**: Generated synthetic features and labels
**Artifacts**: `metrics_smoke.json`, `roc_smoke.png`, `pr_smoke.png`, `reliability_smoke.png`, `confusion_smoke.png`, `policy_smoke.json`, `risk_register_smoke.csv`

### Phase 2: Small EMBER-2024 (Real Data)
Medium-scale test using real EMBER-2024 data with sampling:

```bash
# Run small EMBER test with real data
aicra run-test --phase small_ember --data-dir data/ember2024 --sample-size 10000 --seed 42
```

**Behavior**: Uses LightGBM on sampled EMBER-2024 data
**Data**: Real EMBER-2024 JSONL files (sampled to ~10k rows)
**Requirements**: 
- `data/ember2024/` directory must exist
- Must contain `*.jsonl` files with features and labels
- Will FAIL FAST if real data is missing (no synthetic fallback)
**Artifacts**: `metrics_small_ember.json`, `roc_small_ember.png`, `pr_small_ember.png`, `reliability_small_ember.png`, `confusion_small_ember.png`, `policy_small_ember.json`, `risk_register_small_ember.csv`, `comparison_smoke_small_ember.json`

### Phase 3: Full EMBER-2024 (Real Data)
Full-scale test using complete EMBER-2024 dataset:

```bash
# Run full EMBER test with real data
aicra run-test --phase full --data-dir data/ember2024 --seed 42
```

**Behavior**: Uses LightGBM on complete EMBER-2024 dataset
**Data**: Real EMBER-2024 JSONL files (all available data)
**Requirements**:
- `data/ember2024/` directory must exist
- Must contain `*.jsonl` files with features and labels
- Will FAIL FAST if real data is missing (no synthetic fallback)
**Artifacts**: `metrics_full.json`, `roc_full.png`, `pr_full.png`, `reliability_full.png`, `confusion_full.png`, `policy_full.json`, `risk_register_full.csv`, `test_results_history.csv`, `phase_comparison.png`

### Data Requirements

**For small_ember and full phases, you MUST provide real EMBER-2024 data:**

```
data/ember2024/
├── train_features.jsonl    # Training features (one JSON object per line)
├── train_labels.jsonl      # Training labels (one JSON object per line)
├── test_features.jsonl    # Test features (one JSON object per line)
└── test_labels.jsonl      # Test labels (one JSON object per line)
```

**JSONL Format Example:**
```json
{"feature_0": 0.1, "feature_1": 0.2, "feature_2": 0.3, "family": "benign", "timestamp": "2024-01-01T00:00:00"}
{"feature_0": 0.4, "feature_1": 0.5, "feature_2": 0.6, "family": "lockbit", "timestamp": "2024-01-01T01:00:00"}
```

**Label Format Example:**
```json
{"label": 0}
{"label": 1}
```

**Error Handling:**
- If `data/ember2024/` directory is missing: Clear error message with instructions
- If no `*.jsonl` files found: Clear error message with expected structure
- If invalid JSON in files: Graceful handling with warnings for invalid lines
- **NO SYNTHETIC FALLBACK**: small_ember and full phases will always fail if real data is missing

### Comparison and History

The test runner automatically generates:
- **Phase Comparison**: `comparison_smoke_small_ember.json` comparing smoke vs small EMBER results
- **Test History**: `test_results_history.csv` tracking all phase results over time
- **Progression Plots**: `phase_comparison.png` showing AUROC and Lift@5% progression across phases
- **Data Summaries**: `data_summary_{phase}.json` with dataset statistics for each phase
- Lift@5% > 1.0 (or Lift@10% > 1.0)
- All required artifacts generated (metrics.json, plots, policy.json, register.csv)
- Register contains ≥10 rows with required columns (susceptibility, bucket, techniques, controls)

**Output**: PASS/FAIL summary with metrics, artifact paths, and exit code (0=PASS, 1=FAIL).

### Basic Usage

```bash
# Run complete pipeline
aicra run-all

# Individual commands
aicra train                    # Train model
aicra evaluate                # Evaluate performance
aicra calibrate               # Calibrate probabilities
aicra thresholds              # Compute optimal thresholds
aicra drift-check --new-data sample.csv  # Check for drift
aicra register                # Generate risk register
```

### Makefile Commands

```bash
# Development
make setup                    # Set up development environment
make lint                     # Run linting
make typecheck               # Run type checking
make test                    # Run tests
make coverage                # Run tests with coverage
make audit                   # Security audit

# Pipeline
make train                   # Train model
make evaluate                # Evaluate model
make calibrate               # Calibrate model
make thresholds              # Compute thresholds
make drift                   # Check drift
make register                # Generate register

# Complete pipeline
make all                     # Run everything
```

## Configuration

AICRA uses pydantic-settings for configuration management:

```bash
# Environment variables
export RANDOM_SEED=42
export COST_FP=5.0
export COST_FN=100.0
export IMPACT_DEFAULT=5000000.0
export DRIFT_THRESHOLD=0.05
```

Or create a `.env` file:

```env
RANDOM_SEED=42
COST_FP=5.0
COST_FN=100.0
IMPACT_DEFAULT=5000000.0
DRIFT_THRESHOLD=0.05
MLFLOW_TRACKING_URI=file:./mlruns
```

## Data Requirements

- **EMBER-2024 Dataset**: Static analysis features from PE executables
- **Format**: JSONL (JSON Lines) with feature vectors and metadata
- **Schema**: Validated against `schemas/input_schema.json`

## Model Architecture

- **Base Model**: LightGBM Classifier
- **Ensemble**: Bagged ensemble with 3 models (seeds: 17, 42, 73)
- **Calibration**: Isotonic regression for probability calibration
- **Validation**: Time-ordered split with out-of-family evaluation

## Quality Assurance

### Code Quality
- **Linting**: Ruff, Pylint
- **Formatting**: Black, isort
- **Type Checking**: mypy (strict mode)
- **Security**: pip-audit, detect-secrets

### Testing
- **Unit Tests**: pytest with coverage
- **Integration Tests**: End-to-end pipeline testing
- **Schema Validation**: Input data validation

### CI/CD
- **GitHub Actions**: Automated testing and quality gates
- **SonarQube**: Code quality analysis
- **Docker**: Containerized deployment

## Governance

- **Data License**: See `DATA_LICENSE.md`
- **Model Card**: See `model_card.md`
- **Schema**: `schemas/input_schema.json`
- **Security**: Regular security audits and dependency updates

## Development

### Project Structure

```
aicra/
├── cli.py              # Typer CLI interface
├── config.py           # Pydantic settings
├── core/               # Core functionality
│   ├── data.py         # Data handling
│   ├── evaluation.py   # Metrics and evaluation
│   └── calibration.py  # Probability calibration
├── models/             # ML models
│   └── lightgbm.py    # LightGBM implementation
├── pipelines/          # ML pipelines
│   ├── training.py     # Training pipeline
│   ├── evaluation.py   # Evaluation pipeline
│   ├── calibration.py  # Calibration pipeline
│   └── drift.py        # Drift detection
├── utils/              # Utilities
└── register.py         # Risk register generation
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality gates: `make ci`
5. Submit a pull request

### License

MIT License - see `LICENSE` file for details.

## Citation

```bibtex
@software{aicra2024,
  title={AICRA: AI Cyber Risk Advisor},
  author={AICRA Team},
  year={2024},
  url={https://github.com/aicra/aicra}
}
```

## Support

- **Documentation**: [docs.aicra.org](https://docs.aicra.org)
- **Issues**: [GitHub Issues](https://github.com/aicra/aicra/issues)
- **Discussions**: [GitHub Discussions](https://github.com/aicra/aicra/discussions)
- **Email**: [support@aicra.org](mailto:support@aicra.org)

