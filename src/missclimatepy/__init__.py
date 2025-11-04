"""
missclimatepy
=============

Minimal, production-ready tools for station-wise climate data imputation
via neighbor-restricted Random Forests with optional target inclusion.

Key public API
--------------
- evaluate_all_stations_fast(...)
- neighbor_distances(...)
- run_quickstart(...)

Cite
----
Please cite the JOSS paper (once accepted) and this package’s DOI on Zenodo.
"""

from .metrics import safe_metrics, aggregate_and_score
from .neighbors import neighbor_distances
from .evaluate import evaluate_all_stations_fast
from .quickstart import run_quickstart, QuickstartConfig

__all__ = [
    "safe_metrics",
    "aggregate_and_score",
    "neighbor_distances",
    "evaluate_all_stations_fast",
    "run_quickstart",
    "QuickstartConfig",
]

__version__ = "0.3.0"
