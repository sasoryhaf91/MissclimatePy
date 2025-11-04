# src/missclimatepy/__init__.py
"""
missclimatepy
=============

A minimal and reproducible local framework for imputing climate time series
using only spatial (lat, lon, altitude) and temporal predictors.

This package exposes a small, stable API for JOSS:
- run_quickstart / QuickstartConfig
- evaluate_all_stations_fast / RFParams
- neighbor_distances
"""

from .api import (
    run_quickstart,
    QuickstartConfig,
    evaluate_all_stations_fast,
    RFParams,
    neighbor_distances,
)

__all__ = [
    "run_quickstart",
    "QuickstartConfig",
    "evaluate_all_stations_fast",
    "RFParams",
    "neighbor_distances",
]

__version__ = "0.1.0"
