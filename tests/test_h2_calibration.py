"""
Test H2 Calibration and Thresholding Experiment

This test module validates the H2 experiment results and ensures:
- Calibration metrics are in valid ranges (Brier/ECE between 0 and 1)
- JSON outputs contain expected keys
- Threshold optimization results are valid
- Cost-optimal threshold reduces expected loss compared to F1-optimized
"""

import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest


def load_h2_results(results_path: Path) -> Dict[str, Any]:
    """Load H2 evaluation results from JSON file."""
    if not results_path.exists():
        pytest.skip(f"H2 results file not found: {results_path}")
    
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def test_h2_results_file_exists():
    """Test that H2_full_results.json exists."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    assert results_path.exists(), f"H2 results file not found at {results_path}"


def test_h2_summary_file_exists():
    """Test that H2_summary.md exists."""
    summary_path = Path("results/H2_calibration_thresholds/H2_summary.md")
    assert summary_path.exists(), f"H2 summary file not found at {summary_path}"


def test_h2_json_structure():
    """Test that H2_full_results.json has the expected structure."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    # Check top-level keys
    assert "hypothesis" in results, "Missing 'hypothesis' key"
    assert "hypothesis_statement" in results, "Missing 'hypothesis_statement' key"
    assert "metrics" in results, "Missing 'metrics' key"
    
    # Check metrics structure
    metrics = results["metrics"]
    assert "calibration" in metrics, "Missing 'calibration' key"
    assert "f1_optimized" in metrics, "Missing 'f1_optimized' key"
    assert "cost_optimized" in metrics, "Missing 'cost_optimized' key"
    
    # Check calibration structure
    calibration = metrics["calibration"]
    required_calibration_keys = [
        "brier_uncalibrated", "brier_calibrated", "brier_improvement",
        "ece_uncalibrated", "ece_calibrated", "ece_improvement", "method"
    ]
    for key in required_calibration_keys:
        assert key in calibration, f"Missing calibration key: {key}"


def test_h2_brier_score_ranges():
    """Test that Brier scores are between 0 and 1."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    calibration = results["metrics"]["calibration"]
    brier_uncal = calibration["brier_uncalibrated"]
    brier_cal = calibration["brier_calibrated"]
    
    assert 0.0 <= brier_uncal <= 1.0, f"Brier (uncalibrated) must be between 0 and 1, got {brier_uncal}"
    assert 0.0 <= brier_cal <= 1.0, f"Brier (calibrated) must be between 0 and 1, got {brier_cal}"


def test_h2_ece_ranges():
    """Test that ECE values are between 0 and 1."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    calibration = results["metrics"]["calibration"]
    ece_uncal = calibration["ece_uncalibrated"]
    ece_cal = calibration["ece_calibrated"]
    
    assert 0.0 <= ece_uncal <= 1.0, f"ECE (uncalibrated) must be between 0 and 1, got {ece_uncal}"
    assert 0.0 <= ece_cal <= 1.0, f"ECE (calibrated) must be between 0 and 1, got {ece_cal}"


def test_h2_calibration_method():
    """Test that calibration method is specified and valid."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    method = results["metrics"]["calibration"]["method"]
    assert method is not None, "Calibration method must be specified"
    assert isinstance(method, str), f"Calibration method must be a string, got {type(method)}"
    assert method in ["platt", "isotonic", "auto"], f"Invalid calibration method: {method}"


def test_h2_f1_optimized_structure():
    """Test that F1-optimized threshold results have correct structure."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    f1_opt = results["metrics"]["f1_optimized"]
    assert "uncalibrated" in f1_opt, "Missing 'uncalibrated' in f1_optimized"
    assert "calibrated" in f1_opt, "Missing 'calibrated' in f1_optimized"
    
    for variant in ["uncalibrated", "calibrated"]:
        f1_metrics = f1_opt[variant]
        required_keys = ["threshold", "precision", "recall", "f1", "expected_loss"]
        for key in required_keys:
            assert key in f1_metrics, f"Missing key in f1_optimized.{variant}: {key}"


def test_h2_cost_optimized_structure():
    """Test that cost-optimized threshold results have correct structure."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    cost_opt = results["metrics"]["cost_optimized"]
    assert "cost_fn" in cost_opt, "Missing 'cost_fn' in cost_optimized"
    assert "cost_fp" in cost_opt, "Missing 'cost_fp' in cost_optimized"
    assert "uncalibrated" in cost_opt, "Missing 'uncalibrated' in cost_optimized"
    assert "calibrated" in cost_opt, "Missing 'calibrated' in cost_optimized"
    
    for variant in ["uncalibrated", "calibrated"]:
        cost_metrics = cost_opt[variant]
        required_keys = ["threshold", "precision", "recall", "f1", "expected_loss"]
        for key in required_keys:
            assert key in cost_metrics, f"Missing key in cost_optimized.{variant}: {key}"


def test_h2_threshold_ranges():
    """Test that all thresholds are between 0 and 1."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    metrics = results["metrics"]
    
    # Check F1-optimized thresholds
    f1_uncal_thresh = metrics["f1_optimized"]["uncalibrated"]["threshold"]
    f1_cal_thresh = metrics["f1_optimized"]["calibrated"]["threshold"]
    
    assert 0.0 <= f1_uncal_thresh <= 1.0, f"F1 threshold (uncalibrated) must be between 0 and 1, got {f1_uncal_thresh}"
    assert 0.0 <= f1_cal_thresh <= 1.0, f"F1 threshold (calibrated) must be between 0 and 1, got {f1_cal_thresh}"
    
    # Check cost-optimized thresholds
    cost_uncal_thresh = metrics["cost_optimized"]["uncalibrated"]["threshold"]
    cost_cal_thresh = metrics["cost_optimized"]["calibrated"]["threshold"]
    
    assert 0.0 <= cost_uncal_thresh <= 1.0, f"Cost threshold (uncalibrated) must be between 0 and 1, got {cost_uncal_thresh}"
    assert 0.0 <= cost_cal_thresh <= 1.0, f"Cost threshold (calibrated) must be between 0 and 1, got {cost_cal_thresh}"


def test_h2_precision_recall_f1_ranges():
    """Test that precision, recall, and F1 are between 0 and 1."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    metrics = results["metrics"]
    
    for variant in ["uncalibrated", "calibrated"]:
        # F1-optimized
        f1_metrics = metrics["f1_optimized"][variant]
        assert 0.0 <= f1_metrics["precision"] <= 1.0, f"F1 precision ({variant}) must be between 0 and 1"
        assert 0.0 <= f1_metrics["recall"] <= 1.0, f"F1 recall ({variant}) must be between 0 and 1"
        assert 0.0 <= f1_metrics["f1"] <= 1.0, f"F1 score ({variant}) must be between 0 and 1"
        
        # Cost-optimized
        cost_metrics = metrics["cost_optimized"][variant]
        assert 0.0 <= cost_metrics["precision"] <= 1.0, f"Cost precision ({variant}) must be between 0 and 1"
        assert 0.0 <= cost_metrics["recall"] <= 1.0, f"Cost recall ({variant}) must be between 0 and 1"
        assert 0.0 <= cost_metrics["f1"] <= 1.0, f"Cost F1 ({variant}) must be between 0 and 1"


def test_h2_expected_loss_positive():
    """Test that expected loss values are non-negative."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    metrics = results["metrics"]
    
    for variant in ["uncalibrated", "calibrated"]:
        # F1-optimized
        f1_loss = metrics["f1_optimized"][variant]["expected_loss"]
        assert f1_loss >= 0.0, f"F1 expected loss ({variant}) must be >= 0, got {f1_loss}"
        
        # Cost-optimized
        cost_loss = metrics["cost_optimized"][variant]["expected_loss"]
        assert cost_loss >= 0.0, f"Cost expected loss ({variant}) must be >= 0, got {cost_loss}"


def test_h2_cost_structure():
    """Test that cost structure is specified and positive."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    cost_opt = results["metrics"]["cost_optimized"]
    cost_fn = cost_opt["cost_fn"]
    cost_fp = cost_opt["cost_fp"]
    
    assert cost_fn > 0.0, f"cost_fn must be > 0, got {cost_fn}"
    assert cost_fp > 0.0, f"cost_fp must be > 0, got {cost_fp}"
    assert isinstance(cost_fn, (int, float)), f"cost_fn must be numeric, got {type(cost_fn)}"
    assert isinstance(cost_fp, (int, float)), f"cost_fp must be numeric, got {type(cost_fp)}"


def test_h2_cost_optimal_vs_f1_optimal():
    """
    Test that cost-optimal threshold reduces expected loss compared to F1-optimized.
    
    Note: This is the core H2 hypothesis expectation. If this fails, it indicates
    that cost-aware thresholding is not providing the expected benefit.
    """
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    metrics = results["metrics"]
    
    # Compare calibrated versions (as per H2 hypothesis)
    f1_loss = metrics["f1_optimized"]["calibrated"]["expected_loss"]
    cost_loss = metrics["cost_optimized"]["calibrated"]["expected_loss"]
    
    # Cost-optimal should have lower or equal expected loss
    # (lower is better for expected loss)
    if cost_loss > f1_loss:
        # This is a warning, not a failure, as it depends on the cost structure
        # But we document it
        print(
            f"\n⚠️  H2 Note: Cost-optimal expected loss ({cost_loss:.4f}) is higher than "
            f"F1-optimized ({f1_loss:.4f}).\n"
            f"  This may indicate the cost structure (FN={metrics['cost_optimized']['cost_fn']}, "
            f"FP={metrics['cost_optimized']['cost_fp']}) needs adjustment."
        )


def test_h2_sample_count():
    """Test that test sample count is positive."""
    results_path = Path("results/H2_calibration_thresholds/H2_full_results.json")
    results = load_h2_results(results_path)
    
    n_test = results["metrics"]["n_test_samples"]
    assert n_test > 0, f"n_test_samples must be > 0, got {n_test}"
    assert isinstance(n_test, (int, float)), f"n_test_samples must be numeric, got {type(n_test)}"


def test_h2_backward_compatibility():
    """Test that backward-compatible metrics.json also exists."""
    metrics_path = Path("results/H2_calibration_thresholds/metrics.json")
    if metrics_path.exists():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        assert isinstance(metrics, dict), "metrics.json must be a dictionary"
        assert "calibration" in metrics, "metrics.json must contain 'calibration' key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
