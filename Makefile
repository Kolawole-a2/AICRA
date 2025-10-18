# AICRA Makefile with deterministic builds and quality gates

# Environment variables for reproducibility
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Default values
EMBER_BASE_URL?=
LIMIT?=50000
OPS_Q?=0.2
IMPACT?=5000000
COVERAGE_THRESHOLD?=80

.PHONY: help setup lint typecheck test coverage audit train evaluate calibrate thresholds drift register smoke clean docker-build docker-run sonar

help: ## Show this help message
	@echo "AICRA - AI Cyber Risk Advisor"
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Set up development environment
	python -m pip install --upgrade pip
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "Development environment set up successfully"

lint: ## Run linting (ruff, pylint)
	ruff check aicra tests
	pylint aicra
	@echo "Linting completed"

typecheck: ## Run type checking (mypy)
	mypy aicra
	@echo "Type checking completed"

test: ## Run tests
	pytest tests/ -v

coverage: ## Run tests with coverage
	pytest tests/ --cov=aicra --cov-report=term-missing --cov-report=html --cov-fail-under=$(COVERAGE_THRESHOLD)

audit: ## Run security audit
	pip-audit --desc --format=json --output=pip-audit-report.json
	@echo "Security audit completed"

train: ## Train model
	aicra train

evaluate: ## Evaluate model
	aicra evaluate

calibrate: ## Calibrate model
	aicra calibrate

thresholds: ## Compute optimal thresholds
	aicra thresholds

drift: ## Check for drift
	aicra drift-check --new-data sample.csv

register: ## Generate risk register
	aicra register

smoke: ## Run end-to-end smoke test
	aicra smoke

archive-results: ## Manually archive current results to timestamped folder
	aicra archive-results

list-runs: ## List recent runs from versions log
	aicra list-runs

# Legacy targets for backward compatibility
dirs: ## Create necessary directories
	mkdir -p data/ember data/ember2024 artifacts notebooks logs mappings schemas .github/workflows

download: ## Download EMBER dataset
	python aicra/utils/download_ember.py --outdir data/ember --datasets train test --limit $(LIMIT) --force --base-url $(EMBER_BASE_URL)

extract: ## Extract features
	python aicra/utils/feature_extractor.py --input data/ember/train/ember_features.jsonl --out data/ember/train_pe.npz --limit $(LIMIT) --force

download_full: ## Download full EMBER-2024 dataset
	python - <<END
import os
import thrember
os.makedirs("data/ember2024", exist_ok=True)
thrember.download_dataset("data/ember2024")
END

extract_full: ## Extract full dataset features
	python aicra/utils/feature_extractor.py --input data/ember2024/train/ember_features.jsonl --out data/ember/train_pe.npz --limit $(LIMIT) --force

labels: ## Generate labels
	python - <<END
import pandas as pd, numpy as np, json
features = pd.read_json('data/ember2024/train/ember_features.jsonl', lines=True)
with open('mappings/attack_mapping.json') as f:
    attack_mapping = json.load(f)
ransomware_families = [fam for fam, techniques in attack_mapping.items() if 'T1486' in techniques]
features['is_ransomware'] = features['family'].apply(lambda f: 1 if f in ransomware_families else 0)
X = features.drop(columns=['family','is_ransomware'])
y = features['is_ransomware']
np.savez('data/ember/train_pe_labels.npz', X=X.values, y=y.values)
END

eval: evaluate ## Alias for evaluate

plots: ## Generate plots (legacy)
	aicra evaluate --generate-plots

logs: ## Process logs
	python aicra/utils/log_ingest.py --logs logs/ --out data/normalized_logs.npz --clean --merge

ci: ## Run CI pipeline
	@echo "Running CI pipeline..."
	$(MAKE) lint typecheck test coverage audit
	@echo "CI pipeline completed"

all: ## Run complete pipeline
	$(MAKE) dirs download_full extract_full labels train evaluate calibrate thresholds register
	@echo "AICRA build complete → Train → Evaluate → Optimize Threshold → Generate Register"

# Docker targets
docker-build: ## Build Docker image
	docker build -t aicra .

docker-run: ## Run Docker container
	docker run --rm -v $(PWD)/data:/app/data -v $(PWD)/artifacts:/app/artifacts aicra

# SonarQube targets
sonar: ## Run SonarQube analysis
	docker-compose up -d sonarqube
	@echo "SonarQube started at http://localhost:9000"
	@echo "Run: docker-compose exec sonarqube sonar-scanner"

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	rm -rf .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Clean completed"