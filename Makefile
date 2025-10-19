# AICRA Makefile
# Production-grade build automation

.PHONY: help setup lock format lint typecheck test audit run docker-build docker-test clean

# Default target
help:
	@echo "AICRA Build System"
	@echo "=================="
	@echo ""
	@echo "Available targets:"
	@echo "  setup       - Install dependencies"
	@echo "  lock        - Lock dependency versions"
	@echo "  format      - Format code with black and isort"
	@echo "  lint        - Run linters (ruff, pylint)"
	@echo "  typecheck   - Run type checker (mypy)"
	@echo "  test        - Run tests with coverage"
	@echo "  audit       - Run security audits (pip-audit, safety)"
	@echo "  run         - Run the application"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-test - Test Docker image"
	@echo "  clean       - Clean build artifacts"
	@echo ""

# Setup development environment
setup:
	pip install -e .
	pip install -r requirements/dev.txt
	pre-commit install

# Lock dependency versions
lock:
	pip-compile requirements/base.in -o requirements/base.txt
	pip-compile requirements/dev.in -o requirements/dev.txt
	pip-compile requirements/prod.in -o requirements/prod.txt

# Format code
format:
	black aicra/ tests/
	isort aicra/ tests/
	ruff check --fix aicra/ tests/

# Run linters
lint:
	ruff check aicra/ tests/
	pylint aicra/
	black --check aicra/ tests/
	isort --check-only aicra/ tests/

# Run type checker
typecheck:
	mypy aicra/

# Run tests with coverage
test:
	pytest tests/ --cov=aicra --cov-report=xml --cov-report=html --cov-fail-under=85

# Run security audits
audit:
	pip-audit
	safety check

# Run the application
run:
	python -m aicra --help

# Build Docker image
docker-build:
	docker build -t aicra:latest .

# Test Docker image
docker-test:
	docker run --rm aicra:latest python -m aicra --help

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf coverage.xml
	rm -rf artifacts/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete