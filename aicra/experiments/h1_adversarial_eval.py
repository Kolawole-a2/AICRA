"""
H1 Adversarial Robustness Evaluation: Feature-Level Perturbations and Mimicry Attacks

Evaluates model robustness against:
1. Feature-level perturbations (noise injection)
2. Mimicry attacks (shifting ransomware features toward benign distributions)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

from ..core.serialization import safe_joblib_load

logger = logging.getLogger(__name__)


def perturb_features(
    features: np.ndarray,
    perturbation_type: str = "gaussian",
    strength: float = 0.1,
    feature_ranges: dict | None = None,
) -> np.ndarray:
    """
    Add perturbations to features within plausible ranges.

    Args:
        features: Feature matrix (n_samples, n_features)
        perturbation_type: "gaussian", "uniform", or "mimicry"
        strength: Perturbation strength (0.0 to 1.0)
        feature_ranges: Dict mapping feature indices to (min, max) ranges

    Returns:
        Perturbed features
    """
    n_samples, n_features = features.shape
    perturbed = features.copy()

    if perturbation_type == "gaussian":
        noise = np.random.normal(0, strength, features.shape)
        perturbed = features + noise
    elif perturbation_type == "uniform":
        noise = np.random.uniform(-strength, strength, features.shape)
        perturbed = features + noise
    elif perturbation_type == "mimicry":
        # Shift toward benign distribution (mean=0 for benign samples)
        benign_mean = np.zeros(n_features)  # Simplified - would use actual benign mean
        shift = (benign_mean - features) * strength
        perturbed = features + shift
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    # Clip to valid ranges if provided
    if feature_ranges:
        for idx, (min_val, max_val) in feature_ranges.items():
            perturbed[:, idx] = np.clip(perturbed[:, idx], min_val, max_val)
    else:
        # Default: clip to [0, 1] for normalized features
        perturbed = np.clip(perturbed, 0, 1)

    return perturbed


def evaluate_robustness(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    perturbation_strengths: list[float] | None = None,
    perturbation_types: list[str] | None = None,
) -> dict:
    """
    Evaluate model robustness under various perturbations.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        perturbation_strengths: List of perturbation strengths to test
        perturbation_types: List of perturbation types

    Returns:
        Dictionary with robustness metrics
    """
    logger.info("=" * 80)
    logger.info("Adversarial Robustness Evaluation")
    logger.info("=" * 80)

    if perturbation_strengths is None:
        perturbation_strengths = [0.01, 0.05, 0.1, 0.2]
    if perturbation_types is None:
        perturbation_types = ["gaussian", "uniform", "mimicry"]

    # Baseline metrics (no perturbation)
    y_prob_baseline = model.predict_proba(pd.DataFrame(X_test))
    if y_prob_baseline.ndim > 1:
        y_prob_baseline = y_prob_baseline[:, 1]

    auroc_baseline = roc_auc_score(y_test, y_prob_baseline)
    y_pred_baseline = (y_prob_baseline >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_baseline, labels=[0, 1]).ravel()

    results = {
        "baseline": {
            "auroc": float(auroc_baseline),
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        },
        "perturbations": {},
    }

    # Test each perturbation type and strength
    for ptype in perturbation_types:
        results["perturbations"][ptype] = {}

        for strength in perturbation_strengths:
            logger.info(f"Testing {ptype} perturbation, strength={strength}")

            # Perturb features
            X_perturbed = perturb_features(X_test, ptype, strength)

            # Generate predictions
            y_prob_pert = model.predict_proba(pd.DataFrame(X_perturbed))
            if y_prob_pert.ndim > 1:
                y_prob_pert = y_prob_pert[:, 1]

            # Compute metrics
            auroc_pert = roc_auc_score(y_test, y_prob_pert)
            y_pred_pert = (y_prob_pert >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y_test, y_pred_pert, labels=[0, 1]
            ).ravel()

            # Classification changes
            label_flips = (y_pred_baseline != y_pred_pert).sum()
            label_flip_pct = (label_flips / len(y_test)) * 100.0

            # Focus on ransomware samples (y_test == 1)
            ransomware_mask = y_test == 1
            if ransomware_mask.sum() > 0:
                ransomware_flips = (
                    y_pred_baseline[ransomware_mask] != y_pred_pert[ransomware_mask]
                ).sum()
                ransomware_flip_pct = (ransomware_flips / ransomware_mask.sum()) * 100.0
            else:
                ransomware_flips = 0
                ransomware_flip_pct = 0.0

            results["perturbations"][ptype][f"strength_{strength}"] = {
                "auroc": float(auroc_pert),
                "auroc_drop": float(auroc_baseline - auroc_pert),
                "auroc_drop_pct": (
                    float((auroc_baseline - auroc_pert) / auroc_baseline * 100)
                    if auroc_baseline > 0
                    else 0.0
                ),
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
                "label_flips": int(label_flips),
                "label_flip_pct": float(label_flip_pct),
                "ransomware_flips": int(ransomware_flips),
                "ransomware_flip_pct": float(ransomware_flip_pct),
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp),
                },
            }

            logger.info(
                f"  AUROC: {auroc_pert:.4f} (drop: {auroc_baseline - auroc_pert:.4f})"
            )
            logger.info(f"  Label flips: {label_flips} ({label_flip_pct:.2f}%)")
            logger.info(
                f"  Ransomware flips: {ransomware_flips} ({ransomware_flip_pct:.2f}%)"
            )

    return results


def evaluate_mimicry_attack(
    model, X_ransomware: np.ndarray, X_benign: np.ndarray, mimicry_strength: float = 0.5
) -> dict:
    """
    Evaluate mimicry attack: shift ransomware features toward benign distribution.

    Args:
        model: Trained model
        X_ransomware: Ransomware feature matrix
        X_benign: Benign feature matrix (reference distribution)
        mimicry_strength: Strength of mimicry (0.0 = no change, 1.0 = full shift to benign)

    Returns:
        Dictionary with mimicry attack results
    """
    logger.info("=" * 80)
    logger.info("Mimicry Attack Evaluation")
    logger.info("=" * 80)

    # Compute benign distribution statistics
    benign_mean = X_benign.mean(axis=0)

    # Baseline predictions (unperturbed ransomware)
    y_prob_baseline = model.predict_proba(pd.DataFrame(X_ransomware))
    if y_prob_baseline.ndim > 1:
        y_prob_baseline = y_prob_baseline[:, 1]

    n_ransomware = len(X_ransomware)
    y_true = np.ones(n_ransomware)  # All are ransomware

    # Apply mimicry: shift toward benign distribution
    X_mimicry = X_ransomware.copy()
    for i in range(n_ransomware):
        # Interpolate between ransomware sample and benign mean
        X_mimicry[i] = (1 - mimicry_strength) * X_ransomware[
            i
        ] + mimicry_strength * benign_mean

    # Predictions on mimicry samples
    y_prob_mimicry = model.predict_proba(pd.DataFrame(X_mimicry))
    if y_prob_mimicry.ndim > 1:
        y_prob_mimicry = y_prob_mimicry[:, 1]

    # Metrics
    auroc_baseline = roc_auc_score(y_true, y_prob_baseline)
    auroc_mimicry = roc_auc_score(y_true, y_prob_mimicry)

    # Classification changes
    y_pred_baseline = (y_prob_baseline >= 0.5).astype(int)
    y_pred_mimicry = (y_prob_mimicry >= 0.5).astype(int)

    evasions = (y_pred_baseline == 1) & (y_pred_mimicry == 0)  # Ransomware → Benign
    n_evasions = evasions.sum()
    evasion_rate = (n_evasions / n_ransomware) * 100.0

    # Risk score reduction
    risk_score_reduction = (y_prob_baseline - y_prob_mimicry).mean()
    risk_score_reduction_pct = (
        (risk_score_reduction / y_prob_baseline.mean() * 100)
        if y_prob_baseline.mean() > 0
        else 0.0
    )

    return {
        "mimicry_strength": float(mimicry_strength),
        "baseline": {
            "auroc": float(auroc_baseline),
            "mean_risk_score": float(y_prob_baseline.mean()),
        },
        "mimicry": {
            "auroc": float(auroc_mimicry),
            "mean_risk_score": float(y_prob_mimicry.mean()),
            "auroc_drop": float(auroc_baseline - auroc_mimicry),
        },
        "evasion": {
            "n_evasions": int(n_evasions),
            "evasion_rate_pct": float(evasion_rate),
            "risk_score_reduction": float(risk_score_reduction),
            "risk_score_reduction_pct": float(risk_score_reduction_pct),
        },
    }


def main():
    """Main entry point for adversarial evaluation."""
    import argparse

    parser = argparse.ArgumentParser(description="H1 Adversarial Robustness Evaluation")
    parser.add_argument(
        "--model", type=Path, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--test-data", type=Path, help="Path to test data (CSV or JSONL)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/H1_adversarial"),
        help="Output directory",
    )
    parser.add_argument(
        "--perturbation-strengths",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.1, 0.2],
    )
    parser.add_argument("--mimicry-strength", type=float, default=0.5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load model
    model = safe_joblib_load(args.model)

    # Load test data
    from ..core.data import load_ember_2024

    _, test_data = load_ember_2024()
    X_test = test_data.features.values
    y_test = test_data.labels.values

    # Split into ransomware and benign
    ransomware_mask = y_test == 1
    X_ransomware = X_test[ransomware_mask]
    X_benign = X_test[~ransomware_mask]

    # Run evaluations
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Robustness evaluation
    robustness_results = evaluate_robustness(
        model, X_test, y_test, perturbation_strengths=args.perturbation_strengths
    )

    # Mimicry attack
    mimicry_results = evaluate_mimicry_attack(
        model, X_ransomware, X_benign, mimicry_strength=args.mimicry_strength
    )

    # Save results
    with open(output_dir / "robustness_results.json", "w") as f:
        json.dump(robustness_results, f, indent=2)

    with open(output_dir / "mimicry_results.json", "w") as f:
        json.dump(mimicry_results, f, indent=2)

    logger.info("Adversarial evaluation complete")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
