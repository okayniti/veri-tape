"""Calibrator classes, kept in their own importable module (not a __main__
script) so joblib pickles reference a stable module path -- pickling a class
defined in whatever module happened to be run as `__main__` makes it
unloadable from any other entry point."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self):
        self.lr = LogisticRegression()

    def fit(self, raw_proba: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        eps = 1e-6
        logit = np.log(np.clip(raw_proba, eps, 1 - eps) / np.clip(1 - raw_proba, eps, 1 - eps)).reshape(-1, 1)
        self.lr.fit(logit, y)
        return self

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        eps = 1e-6
        logit = np.log(np.clip(raw_proba, eps, 1 - eps) / np.clip(1 - raw_proba, eps, 1 - eps)).reshape(-1, 1)
        return self.lr.predict_proba(logit)[:, 1]


class IsotonicCalibrator:
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")

    def fit(self, raw_proba: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso.fit(raw_proba, y)
        return self

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        return self.iso.predict(raw_proba)
