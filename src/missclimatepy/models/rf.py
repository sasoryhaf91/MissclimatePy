# src/missclimatepy/models/rf.py
from __future__ import annotations
from typing import Any, Optional
from sklearn.ensemble import RandomForestRegressor

def build_rf(
    n_estimators: int = 300,
    random_state: int = 42,
    n_jobs: int = -1,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = "sqrt",
    bootstrap: bool = True,
    **kwargs: Any,
) -> RandomForestRegressor:
    params = dict(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
    )
    params.update(kwargs)  # allow extra overrides
    return RandomForestRegressor(**params)
