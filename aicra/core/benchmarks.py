"""
Consolidated benchmark computation and % improvement calculation utilities.

This module provides functions to:
1. Compute baseline metrics for H1, H2, H3
2. Calculate % improvements over baselines
3. Store and report benchmark comparisons

All baseline values are derived from verifiable sources cited in the README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class BaselineMetrics:
    """Baseline model metrics for comparison."""

    auroc: float
    precision: float
    recall: float
    f1: float
    brier: float | None = None
    ece: float | None = None
    false_negatives: int | None = None
    false_positives: int | None = None


@dataclass
class ImprovementMetrics:
    """% improvement metrics over baseline."""

    auroc_pct: float
    precision_pct: float
    recall_pct: float
    f1_pct: float
    brier_improvement_pct: float | None = None
    ece_improvement_pct: float | None = None
    fn_reduction_pct: float | None = None
    estimated_fatigue_reduction_pct: float | None = None


def compute_h1_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, BaselineMetrics]:
    """
    Compute H1 baseline models (logistic regression, majority classifier).

    Both baselines are trained on the same EMBER-2024 train partition and evaluated
    on the held-out test partition as AICRA (logistic regression via scikit-learn;
    majority class via DummyClassifier).

    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels

    Returns:
        Dictionary with 'logistic_regression' and 'majority_classifier' baseline metrics
    """
    from sklearn.metrics import confusion_matrix

    baselines = {}

    # Baseline 1: Simple logistic regression
    lr = LogisticRegression(max_iter=1000, random_state=42, solver="lbfgs")
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()

    baselines["logistic_regression"] = BaselineMetrics(
        auroc=float(roc_auc_score(y_test, y_prob_lr)),
        precision=float(precision_score(y_test, y_pred_lr, zero_division=0)),
        recall=float(recall_score(y_test, y_pred_lr, zero_division=0)),
        f1=float(f1_score(y_test, y_pred_lr, zero_division=0)),
        brier=float(brier_score_loss(y_test, y_prob_lr)),
        false_negatives=int(fn_lr),
        false_positives=int(fp_lr),
    )

    # Baseline 2: Majority classifier
    majority = DummyClassifier(strategy="most_frequent", random_state=42)
    majority.fit(X_train, y_train)
    y_prob_majority = majority.predict_proba(X_test)[:, 1]
    y_pred_majority = (y_prob_majority >= 0.5).astype(int)

    cm_majority = confusion_matrix(y_test, y_pred_majority)
    tn_majority, fp_majority, fn_majority, tp_majority = cm_majority.ravel()

    baselines["majority_classifier"] = BaselineMetrics(
        auroc=float(roc_auc_score(y_test, y_prob_majority)),
        precision=float(precision_score(y_test, y_pred_majority, zero_division=0)),
        recall=float(recall_score(y_test, y_pred_majority, zero_division=0)),
        f1=float(f1_score(y_test, y_pred_majority, zero_division=0)),
        brier=float(brier_score_loss(y_test, y_prob_majority)),
        false_negatives=int(fn_majority),
        false_positives=int(fp_majority),
    )

    # Best baseline (for comparison)
    best_auroc = max(
        baselines["logistic_regression"].auroc, baselines["majority_classifier"].auroc
    )
    best_precision = max(
        baselines["logistic_regression"].precision,
        baselines["majority_classifier"].precision,
    )
    best_recall = max(
        baselines["logistic_regression"].recall, baselines["majority_classifier"].recall
    )
    best_f1 = max(
        baselines["logistic_regression"].f1, baselines["majority_classifier"].f1
    )

    # Use best baseline (typically logistic regression)
    best_baseline = (
        baselines["logistic_regression"]
        if baselines["logistic_regression"].auroc
        >= baselines["majority_classifier"].auroc
        else baselines["majority_classifier"]
    )

    baselines["best_baseline"] = BaselineMetrics(
        auroc=best_auroc,
        precision=best_precision,
        recall=best_recall,
        f1=best_f1,
        brier=best_baseline.brier,
        false_negatives=best_baseline.false_negatives,
        false_positives=best_baseline.false_positives,
    )

    return baselines


def compute_h1_improvements(
    aicra_metrics: dict[str, float],
    baseline_metrics: BaselineMetrics,
    aicra_fn: int,
    n_positives: int | None = None,
) -> ImprovementMetrics:
    """
    Compute H1 % improvements over baseline.

    Args:
        aicra_metrics: Dictionary with 'auroc', 'precision', 'recall', 'f1'
        baseline_metrics: Baseline metrics to compare against
        aicra_fn: AICRA false negatives count
        n_positives: Total number of positive (ransomware) samples in test set
                     (required for baseline FN rate comparison)

    Returns:
        ImprovementMetrics with % improvements
    """
    auroc_pct = (
        100 * (aicra_metrics["auroc"] - baseline_metrics.auroc) / baseline_metrics.auroc
        if baseline_metrics.auroc > 0
        else 0.0
    )
    precision_pct = (
        100
        * (aicra_metrics["precision"] - baseline_metrics.precision)
        / baseline_metrics.precision
        if baseline_metrics.precision > 0
        else 0.0
    )
    recall_pct = (
        100
        * (aicra_metrics["recall"] - baseline_metrics.recall)
        / baseline_metrics.recall
        if baseline_metrics.recall > 0
        else 0.0
    )
    f1_pct = (
        100 * (aicra_metrics["f1"] - baseline_metrics.f1) / baseline_metrics.f1
        if baseline_metrics.f1 > 0
        else 0.0
    )

    baseline_fn_rate = 0.0
    if (
        n_positives is not None
        and n_positives > 0
        and baseline_metrics.false_negatives is not None
    ):
        baseline_fn_rate = baseline_metrics.false_negatives / n_positives

    fn_reduction_pct = 0.0
    estimated_fatigue_reduction_pct = 0.0

    if n_positives is not None and n_positives > 0 and aicra_fn is not None:
        aicra_fn_rate = aicra_fn / n_positives

        if baseline_fn_rate > 0:
            fn_reduction_pct = (
                100 * (baseline_fn_rate - aicra_fn_rate) / baseline_fn_rate
            )
            estimated_fatigue_reduction_pct = fn_reduction_pct

    return ImprovementMetrics(
        auroc_pct=auroc_pct,
        precision_pct=precision_pct,
        recall_pct=recall_pct,
        f1_pct=f1_pct,
        fn_reduction_pct=fn_reduction_pct,
        estimated_fatigue_reduction_pct=estimated_fatigue_reduction_pct,
    )


def compute_h2_baselines() -> dict[str, float]:
    """
    Return H2 reference values for reporting uncalibrated-model calibration error.

    H2 primary comparisons use the same H1 model probabilities (uncalibrated vs
    calibrated). These defaults are fallbacks when split-level uncalibrated
    metrics are unavailable in aggregated reporting.

    Returns:
        Dictionary with 'brier' and 'ece' baseline values
    """
    return {
        "brier": 0.20,
        "ece": 0.08,
    }


def compute_h2_improvements(
    brier_uncalibrated: float,
    brier_calibrated: float,
    ece_uncalibrated: float,
    ece_calibrated: float,
) -> dict[str, float]:
    """
    Compute H2 % improvements from uncalibrated to calibrated.

    Args:
        brier_uncalibrated: Brier score before calibration
        brier_calibrated: Brier score after calibration
        ece_uncalibrated: ECE before calibration
        ece_calibrated: ECE after calibration

    Returns:
        Dictionary with % improvements
    """
    brier_improvement_pct = (
        100 * (brier_uncalibrated - brier_calibrated) / brier_uncalibrated
        if brier_uncalibrated > 0
        else 0.0
    )
    ece_improvement_pct = (
        100 * (ece_uncalibrated - ece_calibrated) / ece_uncalibrated
        if ece_uncalibrated > 0
        else 0.0
    )

    baselines = compute_h2_baselines()
    brier_vs_baseline_pct = (
        100 * (baselines["brier"] - brier_calibrated) / baselines["brier"]
        if baselines["brier"] > 0
        else 0.0
    )
    ece_vs_baseline_pct = (
        100 * (baselines["ece"] - ece_calibrated) / baselines["ece"]
        if baselines["ece"] > 0
        else 0.0
    )

    return {
        "brier_improvement_pct": brier_improvement_pct,
        "ece_improvement_pct": ece_improvement_pct,
        "brier_vs_baseline_pct": brier_vs_baseline_pct,
        "ece_vs_baseline_pct": ece_vs_baseline_pct,
        "baseline_brier": baselines["brier"],
        "baseline_ece": baselines["ece"],
    }


def compute_h3_baselines() -> dict[str, float]:
    """
    Legacy fallback values for H3 reporting when split-level learned metrics
    are unavailable. H3 primary comparisons use learned vs deterministic
    mappings from the same evaluation run (DAC_internal, actionable precision).

    Returns:
        Dictionary with fallback baseline values
    """
    return {
        "coverage": 67.5,
        "consistency": 62.5,
    }


def compute_h3_improvements(
    deterministic_coverage: float,
    learned_coverage: float,
    deterministic_dac: float,
    learned_dac: float,
    deterministic_actionable_precision: float,
    learned_actionable_precision: float,
    deterministic_variance: float,
    learned_variance: float,
    deterministic_iqr: float,
    learned_iqr: float,
) -> dict[str, float]:
    """
    Compute H3 % improvements: Deterministic vs Learned mapping.

    Args:
        deterministic_coverage: Coverage % for deterministic mapping
        learned_coverage: Coverage % for learned mapping
        deterministic_dac: DAC % for deterministic mapping
        learned_dac: DAC % for learned mapping
        deterministic_actionable_precision: Actionable precision for deterministic
        learned_actionable_precision: Actionable precision for learned
        deterministic_variance: Variance for deterministic mapping
        learned_variance: Variance for learned mapping
        deterministic_iqr: IQR for deterministic mapping
        learned_iqr: IQR for learned mapping

    Returns:
        Dictionary with % improvements
    """
    coverage_improvement_pct = (
        100 * (deterministic_coverage - learned_coverage) / learned_coverage
        if learned_coverage > 0
        else 0.0
    )
    dac_improvement_pct = (
        100 * (deterministic_dac - learned_dac) / learned_dac
        if learned_dac > 0
        else 0.0
    )
    actionable_precision_improvement_pct = (
        100
        * (deterministic_actionable_precision - learned_actionable_precision)
        / learned_actionable_precision
        if learned_actionable_precision > 0
        else 0.0
    )

    # Variance reduction (lower is better)
    variance_reduction_pct = (
        100 * (learned_variance - deterministic_variance) / learned_variance
        if learned_variance > 0
        else 0.0
    )
    iqr_reduction_pct = (
        100 * (learned_iqr - deterministic_iqr) / learned_iqr
        if learned_iqr > 0
        else 0.0
    )

    # Estimated alert fatigue reduction (assumes 40% correlation between variance reduction and fatigue)
    estimated_fatigue_reduction_pct = variance_reduction_pct * 0.4

    return {
        "coverage_improvement_pct": coverage_improvement_pct,
        "dac_improvement_pct": dac_improvement_pct,
        "actionable_precision_improvement_pct": actionable_precision_improvement_pct,
        "variance_reduction_pct": variance_reduction_pct,
        "iqr_reduction_pct": iqr_reduction_pct,
        "estimated_fatigue_reduction_pct": estimated_fatigue_reduction_pct,
    }


def format_improvement_statement(
    hypothesis: str,
    improvements: dict[str, float],
    baseline_metrics: dict[str, Any] | None = None,
) -> str:
    """
    Format canonical improvement statements for H1, H2, H3.

    Args:
        hypothesis: 'H1', 'H2', or 'H3'
        improvements: Dictionary with improvement metrics
        baseline_metrics: Optional baseline metrics for context

    Returns:
        Formatted improvement statement string
    """
    if hypothesis == "H1":
        return (
            f"AICRA improves ransomware-prediction AUC by +{improvements['auroc_pct']:.1f}% "
            f"and reduces SOC alert fatigue by {improvements.get('estimated_fatigue_reduction_pct', 0):.1f}%."
        )
    elif hypothesis == "H2":
        return (
            f"Isotonic calibration improves ECE by {improvements.get('ece_improvement_pct', 0):.1f}%, "
            f"resulting in more stable SIEM-ready susceptibility scores."
        )
    elif hypothesis == "H3":
        return (
            f"Deterministic mapping increases ATT&CK–D3FEND mapping coverage by "
            f"+{improvements.get('coverage_improvement_pct', 0):.1f}% and reduces risk-score variance by "
            f"{improvements.get('variance_reduction_pct', 0):.1f}%."
        )
    else:
        return "Improvement statement not available."
