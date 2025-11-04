# SPDX-License-Identifier: MIT
"""
missclimatepy
=============

A lightweight, reproducible toolkit for imputing gaps in daily
climatological time series using spatial coordinates (lat, lon, altitude)
and calendar features. The package exposes:

- `MissClimateImputer`: a simple high-level API to impute a single target
  variable (e.g., precipitation, tmin, tmax, evaporation) in long-format data.
- `evaluate_all_stations_fast`: station-wise evaluation that trains one model
  per station using either all other stations or only its K nearest neighbors,
  with optional controlled inclusion (1–95%) of the target station rows.
- `RFParams`: a small dataclass for RandomForest hyperparameters.

The package is column-name agnostic: you pass your own column names for
station id, date, latitude, longitude, altitude, and target.

Example
-------
>>> from missclimatepy import MissClimateImputer, evaluate_all_stations_fast, RFParams
>>> # See README or quickstart for full examples.
"""

from .api import MissClimateImputer
from .evaluate import evaluate_all_stations_fast, RFParams

__all__ = [
    "MissClimateImputer",
    "evaluate_all_stations_fast",
    "RFParams",
]

# Version is kept here for a single source of truth; update on release.
__version__ = "0.1.0"
