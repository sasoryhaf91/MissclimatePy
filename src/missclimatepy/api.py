# SPDX-License-Identifier: MIT
"""
missclimatepy.api
=================

High-level public interface for MissClimatePy.

This module exposes *minimal* but convenient entry points that wrap the
core functionality implemented in the internal submodules:

- :func:`evaluate_xyzt` → station-wise model evaluation.
- :func:`impute_xyzt`   → local imputation of a target series.
- :func:`build_neighbor_map` → spatial KNN neighbor map in (lat, lon).

It also re-exports the model factory utilities:

- :func:`make_model` and :data:`SUPPORTED_MODELS` from :mod:`missclimatepy.models`.

and a small set of plotting helpers from :mod:`missclimatepy.viz`.

The goal is to provide a clean, discoverable surface API while keeping the
internals modular and testable.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from .neighbors import build_neighbor_map
from .evaluate import evaluate_stations
from .impute import impute_dataset
from .models import SUPPORTED_MODELS, make_model
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


# --------------------------------------------------------------------------- #
# High-level evaluation wrapper
# --------------------------------------------------------------------------- #


def evaluate_xyzt(
    data: pd.DataFrame,
    **kwargs: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Station-wise evaluation of XYZT-style models.

    This is a thin, user-facing wrapper around
    :func:`missclimatepy.evaluate.evaluate_stations`. It forwards all keyword
    arguments to that function, so the full set of options (station selection,
    neighbourhood configuration, model kind & hyperparameters, logging, etc.)
    is available without duplicating the signature here.

    Parameters
    ----------
    data : DataFrame
        Long-format table containing station id, date, coordinates and the
        target variable to be modelled.
    **kwargs :
        Any keyword arguments accepted by
        :func:`missclimatepy.evaluate.evaluate_stations`, such as:

        - ``id_col``, ``date_col``, ``lat_col``, ``lon_col``, ``alt_col``,
          ``target_col``;
        - ``start``, ``end``;
        - ``add_cyclic``, ``feature_cols``;
        - station-selection options (``prefix``, ``station_ids``, ``regex``,
          ``custom_filter``, ``min_station_rows``);
        - neighbourhood options (``k_neighbors``, ``neighbor_map``);
        - leakage control (``include_target_pct``, ``include_target_seed``);
        - model configuration (``model_kind``, ``model_params``);
        - output and logging options.

    Returns
    -------
    report : DataFrame
        Per-station metrics at daily, monthly and yearly scales.
    preds : DataFrame
        Per-row predictions with observed and modelled values.
    """
    return evaluate_stations(data=data, **kwargs)


# --------------------------------------------------------------------------- #
# High-level imputation wrapper
# --------------------------------------------------------------------------- #


def impute_xyzt(
    data: pd.DataFrame,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Impute daily station records using XYZT-style models.

    This is a thin wrapper around
    :func:`missclimatepy.impute.impute_dataset`. It forwards all keyword
    arguments to that function, allowing users to control:

    - the schema (id/date/coordinate/target columns),
    - the temporal window,
    - station selection and Minimum Data Requirement (MDR),
    - neighbourhood configuration (KNN map or all-stations),
    - model kind and hyperparameters, and
    - optional on-disk persistence (CSV/Parquet, with or without partitions).

    Parameters
    ----------
    data : DataFrame
        Long-format table with station id, date, coordinates and the target
        variable to be imputed (containing NaNs).
    **kwargs :
        Any keyword arguments accepted by
        :func:`missclimatepy.impute.impute_dataset`.

    Returns
    -------
    DataFrame
        A long-format table with the minimal schema:

        ``[station, date, latitude, longitude, altitude, <target>, source]``

        where ``source`` is ``"observed"`` or ``"imputed"``.
    """
    return impute_dataset(data=data, **kwargs)


# --------------------------------------------------------------------------- #
# Re-exports for convenience
# --------------------------------------------------------------------------- #

# Spatial neighbour utilities
__all__ = [
    "evaluate_xyzt",
    "impute_xyzt",
    "build_neighbor_map",
    "make_model",
    "SUPPORTED_MODELS",
    # plotting helpers
    "plot_missing_matrix",
    "plot_metrics_distribution",
    "plot_parity_scatter",
    "plot_time_series_overlay",
    "plot_spatial_scatter",
    "plot_gap_histogram",
    "plot_imputed_series",
    "plot_imputation_coverage",
]
