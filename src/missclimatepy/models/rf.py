
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestRegressor as SKRF

class RFRegressor:
    def __init__(self, n_estimators=200, max_depth=None, n_jobs=-1, random_state=42):
        self.model = SKRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
