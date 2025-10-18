from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class Metrics:
    auroc: float
    pr_auc: float
    brier: float
    ece: float
    lift_at_k: float
    threshold: float
    confusion: tuple[int, int, int, int]


def expected_calibration_error(y_true: np.ndarray[Any, np.dtype[np.integer]], y_prob: np.ndarray[Any, np.dtype[np.floating]], n_bins: int = 10) -> float:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    weights = np.ones_like(prob_true) / len(prob_true)
    return float(np.sum(weights * np.abs(prob_true - prob_pred)))


def compute_lift_at_k(y_true: np.ndarray[Any, np.dtype[np.integer]], y_prob: np.ndarray[Any, np.dtype[np.floating]], k: float = 0.1) -> float:
    n = len(y_true)
    top_k = max(1, int(n * k))
    order = np.argsort(-y_prob)
    y_top = y_true[order][:top_k]
    precision_at_k = y_top.mean()
    base_rate = y_true.mean()
    return float(precision_at_k / (base_rate + 1e-12))


def evaluate_probs(y_true: np.ndarray[Any, np.dtype[np.integer]], y_prob: np.ndarray[Any, np.dtype[np.floating]], threshold: float) -> Metrics:
    auroc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    lift = compute_lift_at_k(y_true, y_prob, k=0.1)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return Metrics(auroc, pr_auc, brier, ece, lift, threshold, (tn, fp, fn, tp))


def cost_sensitive_threshold(
    y_true: np.ndarray[Any, np.dtype[np.integer]], y_prob: np.ndarray[Any, np.dtype[np.floating]], cost_fn: float, cost_fp: float
) -> float:
    thresholds = np.linspace(0.01, 0.99, 199)
    best_t = 0.5
    best_cost = float("inf")
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fn * cost_fn + fp * cost_fp
        if cost < best_cost:
            best_cost = cost
            best_t = float(t)
    return best_t
