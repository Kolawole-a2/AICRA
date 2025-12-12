"""
Test H1 Classification Experiment

This test module validates the H1 experiment results and ensures:
- Metrics are in valid ranges (AUROC/PR-AUC between 0 and 1)
- JSON outputs contain expected keys
- H1 hypothesis target is met (AUROC >= 0.95)
- All required output files are generated
"""

import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest


def load_h1_results(results_path: Path) -> Dict[str, Any]:
    """Load H1 evaluation results from JSON file."""
    if not results_path.exists():
        pytest.skip(f"H1 results file not found: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_h1_results_file_exists():
    """Test that H1_full_results.json exists."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    assert results_path.exists(), f"H1 results file not found at {results_path}"


def test_h1_summary_file_exists():
    """Test that H1_summary.md exists."""
    summary_path = Path("results/H1_classification/H1_summary.md")
    assert summary_path.exists(), f"H1 summary file not found at {summary_path}"


def test_h1_json_structure():
    """Test that H1_full_results.json has the expected structure."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    # Check top-level keys
    assert "hypothesis" in results, "Missing 'hypothesis' key"
    assert "hypothesis_statement" in results, "Missing 'hypothesis_statement' key"
    assert "metrics" in results, "Missing 'metrics' key"
    
    # Check metrics structure
    metrics = results["metrics"]
    required_metrics = [
        "auroc", "pr_auc", "brier_score", "ece", "precision", "recall", "f1",
        "operational_threshold", "n_train_samples", "n_test_samples"
    ]
    for metric in required_metrics:
        assert metric in metrics, f"Missing required metric: {metric}"


def test_h1_auroc_range():
    """Test that AUROC is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    auroc = results["metrics"]["auroc"]
    assert 0.0 <= auroc <= 1.0, f"AUROC must be between 0 and 1, got {auroc}"


def test_h1_auroc_target():
    """Test that H1 hypothesis target is met: AUROC >= 0.95."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    auroc = results["metrics"]["auroc"]
    assert auroc >= 0.95, (
        f"H1 Hypothesis FAILED: AUROC must be >= 0.95, got {auroc}\n"
        f"This indicates the model did not meet the reliability target."
    )


def test_h1_pr_auc_range():
    """Test that PR-AUC is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    pr_auc = results["metrics"]["pr_auc"]
    assert 0.0 <= pr_auc <= 1.0, f"PR-AUC must be between 0 and 1, got {pr_auc}"


def test_h1_brier_score_range():
    """Test that Brier score is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    brier = results["metrics"]["brier_score"]
    assert 0.0 <= brier <= 1.0, f"Brier score must be between 0 and 1, got {brier}"


def test_h1_ece_range():
    """Test that ECE is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    ece = results["metrics"]["ece"]
    assert 0.0 <= ece <= 1.0, f"ECE must be between 0 and 1, got {ece}"


def test_h1_precision_range():
    """Test that precision is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    precision = results["metrics"]["precision"]
    assert 0.0 <= precision <= 1.0, f"Precision must be between 0 and 1, got {precision}"


def test_h1_recall_range():
    """Test that recall is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    recall = results["metrics"]["recall"]
    assert 0.0 <= recall <= 1.0, f"Recall must be between 0 and 1, got {recall}"


def test_h1_f1_range():
    """Test that F1 is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    f1 = results["metrics"]["f1"]
    assert 0.0 <= f1 <= 1.0, f"F1 must be between 0 and 1, got {f1}"


def test_h1_operational_threshold_range():
    """Test that operational threshold is between 0 and 1."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    threshold = results["metrics"]["operational_threshold"]
    assert 0.0 <= threshold <= 1.0, f"Operational threshold must be between 0 and 1, got {threshold}"


def test_h1_lift_metrics():
    """Test that Lift@k metrics are positive."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    metrics = results["metrics"]
    for k in [1, 5, 10]:
        lift_key = f"lift_at_{k}pct"
        if lift_key in metrics:
            lift = metrics[lift_key]
            # Lift can be any positive number (>= 0)
            assert lift >= 0.0, f"Lift@{k}% must be >= 0, got {lift}"


def test_h1_sample_counts():
    """Test that sample counts are positive integers."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    metrics = results["metrics"]
    n_train = metrics["n_train_samples"]
    n_test = metrics["n_test_samples"]
    
    assert n_train > 0, f"n_train_samples must be > 0, got {n_train}"
    assert n_test > 0, f"n_test_samples must be > 0, got {n_test}"
    assert isinstance(n_train, (int, float)), f"n_train_samples must be numeric, got {type(n_train)}"
    assert isinstance(n_test, (int, float)), f"n_test_samples must be numeric, got {type(n_test)}"


def test_h1_model_type():
    """Test that model_type is specified."""
    results_path = Path("results/H1_classification/H1_full_results.json")
    results = load_h1_results(results_path)
    
    model_type = results["metrics"].get("model_type")
    assert model_type is not None, "model_type must be specified"
    assert isinstance(model_type, str), f"model_type must be a string, got {type(model_type)}"
    assert model_type in ["lgbm", "ffnn"], f"model_type must be 'lgbm' or 'ffnn', got {model_type}"


def test_h1_backward_compatibility():
    """Test that backward-compatible metrics.json also exists."""
    metrics_path = Path("results/H1_classification/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        assert isinstance(metrics, dict), "metrics.json must be a dictionary"
        assert "auroc" in metrics, "metrics.json must contain 'auroc' key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
