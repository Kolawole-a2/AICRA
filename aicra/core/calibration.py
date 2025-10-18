from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class Calibrator:
    method: str
    model: IsotonicRegression | LogisticRegression

    def fit(self, probs: np.ndarray[Any, np.dtype[np.floating]], y: np.ndarray[Any, np.dtype[np.integer]]) -> Calibrator:
        if self.method == "isotonic":
            self.model.fit(probs, y)
        else:
            self.model.fit(probs.reshape(-1, 1), y)
        return self

    def transform(self, probs: np.ndarray[Any, np.dtype[np.floating]]) -> np.ndarray[Any, np.dtype[np.floating]]:
        if self.method == "isotonic":
            return self.model.transform(probs)
        return self.model.predict_proba(probs.reshape(-1, 1))[:, 1]

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> Calibrator:
        return joblib.load(path)


def create_calibrator(method: str = "platt") -> Calibrator:
    if method == "isotonic":
        return Calibrator(method=method, model=IsotonicRegression(out_of_bounds="clip"))
    return Calibrator(method="platt", model=LogisticRegression(max_iter=1000))
