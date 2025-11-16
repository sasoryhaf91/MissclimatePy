# src/missclimatepy/__init__.py
"""
missclimatepy
=============

Minimal and reproducible framework for climate data imputation using only
spatial coordinates (latitude, longitude, altitude) and calendar features.

Public API
----------
- MissClimateImputer : High-level imputer class for fitting and imputing gaps.
- impute_dataset     : Single-variable long-format imputation engine.
- RFParams           : Dataclass carrying RandomForest hyperparameters.
- evaluate_stations  : Station-wise evaluation with KNN spatial pooling and
                       optional stratified inclusion of target observations.
- __version__        : Package version string.

Example
-------
>>> import pandas as pd
>>> from missclimatepy import MissClimateImputer, evaluate_stations, RFParams
>>>
>>> df = pd.DataFrame({
...     "station": ["S001"]*3 + ["S002"]*3,
...     "date": pd.to_datetime(["1991-01-01","1991-01-02","1991-01-03"]*2),
...     "latitude": [19.5]*6,
...     "longitude": [-99.1]*6,
...     "altitude": [2300]*6,
...     "tmin": [8.0, None, 7.5, 9.0, None, 8.2],
... })
>>>
>>> imp = MissClimateImputer(
...     engine="rf",
...     target="tmin",
...     k_neighbors=5,
...     min_obs_per_station=10,
...     n_estimators=100,
...     n_jobs=-1,
... )
>>> df_filled = imp.fit_transform(df)
>>> report = evaluate_stations(
...     df,
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     k_neighbors=5, include_target_pct=10.0,
...     rf_params=RFParams(n_estimators=100, random_state=42),
...     show_progress=False,
... )

License
-------
MIT License © 2025 Hugo Antonio Fernández
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("missclimatepy")
except Exception:
    __version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
from .api import MissClimateImputer
from .evaluate import RFParams, evaluate_stations
from .imputate import impute_dataset

__all__ = [
    "MissClimateImputer",
    "RFParams",
    "evaluate_stations",
    "impute_dataset",
    "__version__",
]
