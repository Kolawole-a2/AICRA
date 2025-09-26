from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    method: str
    model: object

    def fit(self, probs: np.ndarray, y: np.ndarray):
        if self.method == "isotonic":
            self.model.fit(probs, y)
        else:
            self.model.fit(probs.reshape(-1, 1), y)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if self.method == "isotonic":
            return self.model.transform(probs)
        return self.model.predict_proba(probs.reshape(-1, 1))[:, 1]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def create_calibrator(method: str = "platt") -> Calibrator:
    if method == "isotonic":
        return Calibrator(method=method, model=IsotonicRegression(out_of_bounds="clip"))
    return Calibrator(method="platt", model=LogisticRegression(max_iter=1000))
