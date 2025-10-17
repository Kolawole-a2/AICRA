"""Calibration pipeline for AICRA models."""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import brier_score_loss

from ..config import Settings
from ..core.calibration import Calibrator
from ..core.data import Dataset


class CalibrationPipeline:
    """Calibration pipeline with reliability diagrams."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self,
        train_data: Dataset,
        val_data: Dataset,
        y_prob_train: np.ndarray[Any, np.dtype[np.floating]],
        y_prob_val: np.ndarray[Any, np.dtype[np.floating]],
        method: Literal["platt", "isotonic", "auto"] = "auto",
    ) -> Calibrator:
        """Train calibrator and generate reliability plot."""
        
        # Select best method if auto
        if method == "auto":
            method = self._select_best_calibration_method(y_prob_train, train_data.labels.values)
            mlflow.log_param("selected_calibration_method", method)

        # Train calibrator
        calibrator = self._create_calibrator(method)
        calibrator.fit(y_prob_train, train_data.labels.values)

        # Generate calibrated probabilities
        y_prob_calibrated = calibrator.transform(y_prob_val)

        # Generate reliability plot
        self._plot_reliability_diagram(
            val_data.labels.values,
            y_prob_val,
            y_prob_calibrated,
            method
        )

        # Compute calibration metrics
        brier_uncalibrated = brier_score_loss(val_data.labels.values, y_prob_val)
        brier_calibrated = brier_score_loss(val_data.labels.values, y_prob_calibrated)
        
        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_param("calibration_method", method)
            mlflow.log_metrics({
                "brier_uncalibrated": brier_uncalibrated,
                "brier_calibrated": brier_calibrated,
                "brier_improvement": brier_uncalibrated - brier_calibrated,
            })
            mlflow.log_artifacts(str(self.settings.artifacts_dir))

        return calibrator
    
    def _select_best_calibration_method(
        self, 
        y_prob: np.ndarray[Any, np.dtype[np.floating]], 
        y_true: np.ndarray[Any, np.dtype[np.integer]]
    ) -> Literal["platt", "isotonic"]:
        """Select best calibration method via CV Brier score."""
        
        # Create dummy classifier wrapper for sklearn's CalibratedClassifierCV
        class DummyClassifier:
            def __init__(self, y_prob):
                self.y_prob = y_prob
            
            def predict_proba(self, X):
                return np.column_stack([1 - self.y_prob, self.y_prob])
        
        dummy_clf = DummyClassifier(y_prob)
        
        # Test Platt scaling
        platt_calibrator = CalibratedClassifierCV(dummy_clf, method='sigmoid', cv=3)
        platt_scores = cross_val_score(
            platt_calibrator, 
            np.arange(len(y_prob)).reshape(-1, 1), 
            y_true, 
            scoring='neg_brier_score', 
            cv=3
        )
        platt_brier = -np.mean(platt_scores)
        
        # Test isotonic regression
        isotonic_calibrator = CalibratedClassifierCV(dummy_clf, method='isotonic', cv=3)
        isotonic_scores = cross_val_score(
            isotonic_calibrator, 
            np.arange(len(y_prob)).reshape(-1, 1), 
            y_true, 
            scoring='neg_brier_score', 
            cv=3
        )
        isotonic_brier = -np.mean(isotonic_scores)
        
        # Return method with lower Brier score
        return "isotonic" if isotonic_brier < platt_brier else "platt"
    
    def _create_calibrator(self, method: Literal["platt", "isotonic"]) -> Calibrator:
        """Create calibrator based on method."""
        if method == "platt":
            return PlattCalibrator()
        elif method == "isotonic":
            return IsotonicCalibrator()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

    def _plot_reliability_diagram(
        self,
        y_true: np.ndarray[Any, np.dtype[np.integer]],
        y_prob_uncalibrated: np.ndarray[Any, np.dtype[np.floating]],
        y_prob_calibrated: np.ndarray[Any, np.dtype[np.floating]],
        method: str
    ) -> None:
        """Plot reliability diagram comparing uncalibrated vs calibrated probabilities."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Uncalibrated
        prob_true, prob_pred = calibration_curve(y_true, y_prob_uncalibrated, n_bins=10)
        ax1.plot(prob_pred, prob_true, 'o-', label='Model')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Mean Predicted Probability')
        ax1.set_ylabel('Fraction of Positives')
        ax1.set_title('Uncalibrated Probabilities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Calibrated
        prob_true_cal, prob_pred_cal = calibration_curve(y_true, y_prob_calibrated, n_bins=10)
        ax2.plot(prob_pred_cal, prob_true_cal, 'o-', label=f'{method.title()} Calibrated')
        ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax2.set_xlabel('Mean Predicted Probability')
        ax2.set_ylabel('Fraction of Positives')
        ax2.set_title(f'{method.title()} Calibrated Probabilities')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.settings.artifacts_dir / 'reliability.png', dpi=300, bbox_inches='tight')
        plt.close()


class PlattCalibrator(Calibrator):
    """Platt scaling calibrator."""
    
    def __init__(self):
        self.lr = LogisticRegression()
        self.is_fitted = False
    
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Fit Platt scaling."""
        # Platt scaling: sigmoid(1 / (1 + exp(A * logit(p) + B)))
        # We use logistic regression on logit(p) to find A and B
        logits = np.log(y_prob / (1 - y_prob + 1e-15))  # Add small epsilon to avoid log(0)
        self.lr.fit(logits.reshape(-1, 1), y_true)
        self.is_fitted = True
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform probabilities using Platt scaling."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        logits = np.log(y_prob / (1 - y_prob + 1e-15))
        calibrated_probs = self.lr.predict_proba(logits.reshape(-1, 1))[:, 1]
        return calibrated_probs


class IsotonicCalibrator(Calibrator):
    """Isotonic regression calibrator."""
    
    def __init__(self):
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = False
    
    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> None:
        """Fit isotonic regression."""
        self.isotonic.fit(y_prob, y_true)
        self.is_fitted = True
    
    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Transform probabilities using isotonic regression."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        return self.isotonic.transform(y_prob)
