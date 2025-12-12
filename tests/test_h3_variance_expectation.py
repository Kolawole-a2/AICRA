"""
Test H3 expectation: Deterministic variance_reduction must be significantly higher than learned.

This test enforces a key H3 hypothesis expectation:
- Deterministic mapping should achieve higher variance reduction than learned mapping
- The difference must be statistically significant (p < 0.05)

The test loads results from the H3 evaluation and validates these expectations.
"""

import json
import math
from pathlib import Path
import pytest


def load_h3_results(results_path: Path) -> dict:
    """Load H3 evaluation results from JSON file."""
    if not results_path.exists():
        pytest.skip(f"H3 results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def test_h3_variance_reduction_expectation():
    """
    Test that deterministic variance_reduction is significantly higher than learned.
    
    This test validates a core H3 expectation:
    1. Deterministic mapping achieves higher mean variance reduction than learned
    2. The difference is statistically significant (p < 0.05)
    
    If this test fails, it indicates that either:
    - The deterministic mapping is not performing better than learned (data-driven finding)
    - There is an issue with the evaluation pipeline or data quality
    """
    # Load H3 results
    results_path = Path("results/H3_full_evaluation/H3_full_results.json")
    results = load_h3_results(results_path)
    
    # Extract aggregated metrics
    aggregated = results.get("aggregated_metrics", {})
    det_metrics = aggregated.get("deterministic", {})
    lrn_metrics = aggregated.get("learned", {})
    stat_tests = aggregated.get("statistical_tests", {})
    
    # Extract variance_reduction means
    det_variance = det_metrics.get("variance_reduction", {})
    lrn_variance = lrn_metrics.get("variance_reduction", {})
    
    det_mean = det_variance.get("mean")
    lrn_mean = lrn_variance.get("mean")
    
    # Extract statistical test p-value
    variance_test = stat_tests.get("variance_reduction", {})
    ttest = variance_test.get("ttest", {})
    p_ttest = ttest.get("pvalue")
    
    # Validate that required metrics exist
    assert det_mean is not None, "Deterministic variance_reduction mean not found in results"
    assert lrn_mean is not None, "Learned variance_reduction mean not found in results"
    assert p_ttest is not None, "Variance_reduction t-test p-value not found in results"
    
    # Small epsilon for floating-point comparison
    eps = 1e-8
    
    # Assertion 1: Deterministic must have strictly higher mean variance reduction
    assert det_mean > lrn_mean + eps, (
        f"H3 Expectation FAILED: Deterministic variance_reduction must be higher than learned.\n"
        f"  Deterministic mean: {det_mean}\n"
        f"  Learned mean: {lrn_mean}\n"
        f"  Difference: {det_mean - lrn_mean}\n"
        f"  This indicates the deterministic mapping is not achieving better variance reduction."
    )
    
    # Assertion 2: Difference must be statistically significant at alpha = 0.05
    # Handle NaN case (can occur if variance is constant across splits)
    if math.isnan(p_ttest):
        pytest.fail(
            f"H3 Expectation FAILED: Variance_reduction t-test p-value is NaN (undefined).\n"
            f"  This typically occurs when variance_reduction is constant across all splits.\n"
            f"  Deterministic mean: {det_mean}\n"
            f"  Learned mean: {lrn_mean}\n"
            f"  This suggests insufficient variation in the data or an issue with the evaluation pipeline."
        )
    
    assert p_ttest < 0.05, (
        f"H3 Expectation FAILED: Variance_reduction difference is not statistically significant.\n"
        f"  Expected p-value < 0.05, but got p = {p_ttest}\n"
        f"  Deterministic mean: {det_mean}\n"
        f"  Learned mean: {lrn_mean}\n"
        f"  Difference: {det_mean - lrn_mean}\n"
        f"  While deterministic is higher, the difference is not statistically significant at alpha=0.05."
    )
    
    # If we get here, the test passed
    print(f"\n✓ H3 Variance Reduction Expectation PASSED")
    print(f"  Deterministic variance_reduction: {det_mean:.6f}")
    print(f"  Learned variance_reduction: {lrn_mean:.6f}")
    print(f"  Difference: {det_mean - lrn_mean:.6f}")
    print(f"  T-test p-value: {p_ttest:.6f} (< 0.05)")


if __name__ == "__main__":
    # Allow running directly with: python -m pytest tests/test_h3_variance_expectation.py
    pytest.main([__file__, "-v"])
