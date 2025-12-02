# SPDX-License-Identifier: MIT
"""
missclimatepy.api
=================

Convenience API for MissClimatePy.

This module exposes *minimal* but convenient entry points that wrap the
core functionality implemented in the internal submodules:

- :func:`evaluate_xyzt` → station-wise model evaluation on XYZT features.
- :func:`impute_xyzt`   → network-wide imputation of a target series.
- :func:`build_neighbor_map` → spatial KNN neighbor map in (lat, lon).

It also re-exports commonly used utilities for diagnostics:

- Masking helpers (coverage, gap profiles, random masking).
- Metric helpers (MAE/RMSE/R2/KGE, aggregated metrics).
- Visualization helpers (missingness matrix, parity plots, etc.).

Typical usage (Kaggle-style)
----------------------------

>>> import pandas as pd
>>> import missclimatepy as mcp
>>> from missclimatepy import api
>>>
>>> df = pd.read_csv("smn_mx_daily.csv", parse_dates=["date"])
>>>
>>> report, preds = api.evaluate_xyzt(
...     data=df,
...     id_col="station", date_col="date",
...     lat_col="lat", lon_col="lon", alt_col="alt",
...     target_col="tmin",
...     start="1991-01-01", end="2020-12-31",
...     model_kind="rf",
...     model_params={"n_estimators": 200, "random_state": 42},
...     k_neighbors=20,
...     include_target_pct=10.0,
...     min_station_rows=3650,
...     include_mcm_baseline=True,
...     include_kge=True,
... )
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd

from .evaluate import evaluate_stations
from .impute import impute_dataset
from .neighbors import build_neighbor_map

from .metrics import (
    compute_metrics,
    aggregate_and_compute,
)

from .masking import (
    percent_missing_between,
    gap_profile_by_station,
    missing_matrix,
    describe_missing,
    apply_random_mask_by_station,
)

from .viz import (
    plot_missing_matrix,
    plot_metrics_distribution,
    plot_parity_scatter,
    plot_time_series_overlay,
    plot_spatial_scatter,
    plot_gap_histogram,
    plot_imputed_series,
    plot_imputation_coverage,
)

from .models import (
    SUPPORTED_MODELS,
    make_model,
)


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------


def evaluate_xyzt(*args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Station-wise evaluation on XYZT features.

    This is a thin wrapper around :func:`missclimatepy.evaluate_stations`
    that simply forwards all positional and keyword arguments.

    Returns
    -------
    (report, predictions) : (DataFrame, DataFrame)
        Exactly the same output as :func:`evaluate_stations`.
    """
    return evaluate_stations(*args, **kwargs)


def impute_xyzt(*args, **kwargs) -> pd.DataFrame:
    """
    Network-wide imputation on XYZT features.

    This is a thin wrapper around :func:`missclimatepy.impute_dataset`
    that simply forwards all positional and keyword arguments.

    Returns
    -------
    DataFrame
        Long-format output with columns:

        [station, date, latitude, longitude, altitude, <target>, source]
    """
    return impute_dataset(*args, **kwargs)


# Optional convenience aliases matching top-level style
def evaluate(*args, **kwargs):
    """
    Alias of :func:`evaluate_xyzt` for users who prefer a shorter name.
    """
    return evaluate_xyzt(*args, **kwargs)


def impute(*args, **kwargs):
    """
    Alias of :func:`impute_xyzt` for users who prefer a shorter name.
    """
    return impute_xyzt(*args, **kwargs)


__all__ = [
    # High-level wrappers
    "evaluate_xyzt",
    "impute_xyzt",
    "evaluate",
    "impute",
    # Neighbors
    "build_neighbor_map",
    # Metrics
    "compute_metrics",
    "aggregate_and_compute",
    # Masking
    "percent_missing_between",
    "gap_profile_by_station",
    "missing_matrix",
    "describe_missing",
    "apply_random_mask_by_station",
    # Visualization
    "plot_missing_matrix",
    "plot_metrics_distribution",
    "plot_parity_scatter",
    "plot_time_series_overlay",
    "plot_spatial_scatter",
    "plot_gap_histogram",
    "plot_imputed_series",
    "plot_imputation_coverage",
    # Models
    "SUPPORTED_MODELS",
    "make_model",
]
