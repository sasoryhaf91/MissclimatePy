# SPDX-License-Identifier: MIT
"""
missclimatepy
=============

Spatial–temporal imputation for daily climate station records using
only XYZT-style features:

    X = (x, y, z, t) = (longitude, latitude, elevation, calendar features)

Main high-level entry points
----------------------------

- :func:`evaluate` – station-wise evaluation of XYZT models on a daily
  climate network, with flexible station selection, KNN neighborhoods,
  controlled leakage (include_target_pct), and a full metric set
  (MAE/RMSE/R2/KGE at daily and aggregated scales), plus comparison
  against a Mean Climatology Model (MCM) baseline.

- :func:`impute` – network-wide imputation of missing values using the
  same XYZT feature design and model options, returning a long-format
  dataframe with a ``source`` flag:

    * "observed" – original non-missing values
    * "imputed"  – values filled by the model
    * "missing"  – still-missing values (if any remain)

Core submodules
---------------

- :mod:`missclimatepy.features` – datetime helpers and calendar features
- :mod:`missclimatepy.metrics`  – metric computation (MAE, RMSE, R2, KGE)
- :mod:`missclimatepy.models`   – model factory for RF / ET / KNN / MLP / etc.
- :mod:`missclimatepy.neighbors` – spatial KNN neighbor maps
- :mod:`missclimatepy.masking`  – missingness summaries and random masking
- :mod:`missclimatepy.viz`      – plotting helpers for diagnostics
- :mod:`missclimatepy.evaluate` – core station-wise evaluator
- :mod:`missclimatepy.impute`   – core network-wide imputer
"""

from __future__ import annotations

# Core high-level functions
from .evaluate import evaluate_stations
from .impute import impute_dataset

# Neighbors
from .neighbors import build_neighbor_map

# Metrics
from .metrics import (
    compute_metrics,
    aggregate_and_compute,
)

# Masking
from .masking import (
    percent_missing_between,
    gap_profile_by_station,
    missing_matrix,
    describe_missing,
    apply_random_mask_by_station,
)

# Visualization
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

# Models
from .models import (
    SUPPORTED_MODELS,
    make_model,
)

# Features / utilities
from .features import (
    ensure_datetime_naive,
    add_calendar_features,
    add_cyclic_doy,
    validate_required_columns,
)


# Public, user-facing short aliases (for Kaggle / examples)
def evaluate(*args, **kwargs):
    """
    Thin wrapper around :func:`evaluate_stations`.

    See :func:`missclimatepy.evaluate_stations` for full documentation.
    """
    return evaluate_stations(*args, **kwargs)


def impute(*args, **kwargs):
    """
    Thin wrapper around :func:`impute_dataset`.

    See :func:`missclimatepy.impute_dataset` for full documentation.
    """
    return impute_dataset(*args, **kwargs)


__all__ = [
    # High-level aliases
    "evaluate",
    "impute",
    # Core implementations
    "evaluate_stations",
    "impute_dataset",
    # Models
    "SUPPORTED_MODELS",
    "make_model",
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
    # Features / utilities
    "ensure_datetime_naive",
    "add_calendar_features",
    "add_cyclic_doy",
    "validate_required_columns",
]


# Sync this with pyproject.toml if you bump the version
__version__ = "0.1.0"
