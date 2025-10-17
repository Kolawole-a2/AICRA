from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..config import get_settings


@dataclass
class BaggedLightGBM:
    models: list[LGBMClassifier]
    per_seed_metrics: dict[str, Any] = None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray[Any, np.dtype[np.floating]]:
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(np.vstack(probs), axis=0)
    
    def predict_proba_per_seed(self, X: pd.DataFrame) -> list[np.ndarray[Any, np.dtype[np.floating]]]:
        """Get predictions from each individual model for analysis."""
        return [m.predict_proba(X)[:, 1] for m in self.models]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> BaggedLightGBM:
        return joblib.load(path)


def train_bagged_lightgbm(X: pd.DataFrame, y: pd.Series) -> BaggedLightGBM:
    settings = get_settings()
    models: list[LGBMClassifier] = []
    per_seed_metrics = {}
    
    for i, seed in enumerate(settings.random_seeds):
        model = LGBMClassifier(
            objective="binary",
            learning_rate=settings.learning_rate,
            num_leaves=settings.num_leaves,
            n_estimators=settings.n_estimators,
            subsample=settings.subsample,
            colsample_bytree=settings.colsample_bytree,
            random_state=seed,
            class_weight=settings.class_weight,
            boosting_type="gbdt",
            # GOSS off per constraints
        )
        model.fit(X, y)
        models.append(model)
        
        # Log per-seed metrics
        per_seed_metrics[f"seed_{i}"] = {
            "seed": seed,
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "feature_importance_mean": np.mean(model.feature_importances_),
            "feature_importance_std": np.std(model.feature_importances_),
        }
    
    # Log ensemble metrics
    mlflow.log_metrics({
        "ensemble_size": len(models),
        "total_features": X.shape[1],
        "total_samples": X.shape[0],
    })
    
    return BaggedLightGBM(models=models, per_seed_metrics=per_seed_metrics)
