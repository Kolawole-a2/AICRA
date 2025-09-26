from __future__ import annotations

from dataclasses import dataclass
from typing import List

import joblib
import numpy as np
from lightgbm import LGBMClassifier

from .config import TRAINING


@dataclass
class BaggedLightGBM:
    models: List[LGBMClassifier]

    def predict_proba(self, X):
        probs = [m.predict_proba(X)[:, 1] for m in self.models]
        return np.mean(np.vstack(probs), axis=0)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def train_bagged_lightgbm(X, y) -> BaggedLightGBM:
    models: List[LGBMClassifier] = []
    for seed in TRAINING.random_seeds:
        model = LGBMClassifier(
            objective="binary",
            learning_rate=TRAINING.learning_rate,
            num_leaves=TRAINING.num_leaves,
            n_estimators=TRAINING.n_estimators,
            subsample=TRAINING.subsample,
            colsample_bytree=TRAINING.colsample_bytree,
            random_state=seed,
            class_weight=TRAINING.class_weight,
            boosting_type="gbdt",
            # GOSS off per constraints
        )
        model.fit(X, y)
        models.append(model)
    return BaggedLightGBM(models=models)
