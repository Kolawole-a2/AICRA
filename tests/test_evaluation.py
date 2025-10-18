"""Tests for evaluation functionality."""

import numpy as np
from sklearn.datasets import make_classification

from aicra.core.evaluation import (
    Metrics,
    compute_lift_at_k,
    cost_sensitive_threshold,
    evaluate_probs,
    expected_calibration_error,
)


def test_evaluate_probs() -> None:
    """Test evaluation metrics computation."""
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Create synthetic probabilities
    y_prob = np.random.random(1000)
    threshold = 0.5

    # Evaluate
    metrics = evaluate_probs(y, y_prob, threshold)

    assert isinstance(metrics, Metrics)
    assert 0 <= metrics.auroc <= 1
    assert 0 <= metrics.pr_auc <= 1
    assert 0 <= metrics.brier <= 1
    assert 0 <= metrics.ece <= 1
    assert metrics.lift_at_k >= 0
    assert 0 <= metrics.threshold <= 1
    assert len(metrics.confusion) == 4


def test_cost_sensitive_threshold() -> None:
    """Test cost-sensitive threshold computation."""
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Create synthetic probabilities
    y_prob = np.random.random(1000)

    # Test with different costs
    cost_fn = 100.0
    cost_fp = 5.0

    threshold = cost_sensitive_threshold(y, y_prob, cost_fn, cost_fp)

    assert 0 <= threshold <= 1
    assert isinstance(threshold, float)


def test_expected_calibration_error() -> None:
    """Test expected calibration error computation."""
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Create synthetic probabilities
    y_prob = np.random.random(1000)

    ece = expected_calibration_error(y, y_prob)

    assert 0 <= ece <= 1
    assert isinstance(ece, float)


def test_compute_lift_at_k() -> None:
    """Test lift at k computation."""
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # Create synthetic probabilities
    y_prob = np.random.random(1000)

    lift = compute_lift_at_k(y, y_prob, k=0.1)

    assert lift >= 0
    assert isinstance(lift, float)


def test_evaluate_probs_perfect_classifier() -> None:
    """Test evaluation with perfect classifier."""
    # Create perfect classifier scenario
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = 0.5

    metrics = evaluate_probs(y_true, y_prob, threshold)

    # Should have high performance
    assert metrics.auroc > 0.8
    assert metrics.pr_auc > 0.8
    assert metrics.brier < 0.5


def test_evaluate_probs_random_classifier() -> None:
    """Test evaluation with random classifier."""
    # Create random classifier scenario
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.4, 0.6, 0.3, 0.7])
    threshold = 0.5

    metrics = evaluate_probs(y_true, y_prob, threshold)

    # Should have moderate performance
    assert 0.3 <= metrics.auroc <= 0.7
    assert 0.3 <= metrics.pr_auc <= 0.8
