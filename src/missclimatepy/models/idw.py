
from __future__ import annotations
import numpy as np

class IDWRegressor:
    """
    Inverse Distance Weighting using only (lat, lon, elev) and ignoring calendar terms
    in the distance. Assumes X columns = [lat, lon, elev, doy_sin, doy_cos, year]
    but distance is computed over the first three.
    """
    def __init__(self, power: float = 2.0, eps: float = 1e-12):
        self.power = power
        self.eps = eps
        self.X_ref = None
        self.y_ref = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_ref = X[:, :3]  # lat, lon, elev
        self.y_ref = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.X_ref is None:
            raise RuntimeError("Fit before predict")
        Q = X[:, :3]
        d2 = np.sum((Q[:, None, :] - self.X_ref[None, :, :])**2, axis=2)
        w = 1.0 / (d2 + self.eps)**(self.power / 2.0)
        w /= w.sum(axis=1, keepdims=True)
        yhat = (w * self.y_ref[None, :]).sum(axis=1)
        return yhat
