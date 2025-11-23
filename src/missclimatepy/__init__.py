# SPDX-License-Identifier: MIT
"""
missclimatepy
=============

Spatial–temporal imputation tools for daily climate station records.

Core goal
---------

MissClimatePy is a focused toolkit for **imputing daily climate series**
(e.g. precipitation, minimum/maximum temperature) in station networks.

The main idea is to provide a **robust imputer** that:

- works on long-format daily tables (station × date),
- uses only **space–time predictors**:
  latitude, longitude, altitude (x, y, z) and calendar features
  (year, month, day-of-year, optional harmonics),
- can fill **all gaps** in a station, even when the station has very sparse
  information, by relying on a global model plus a climatological baseline,
- supports **minimum data requirement (MDR) experiments**:
  you can explore how station-level coverage affects imputation quality and
  search for practical thresholds of “acceptable” data availability.

Configurable backends
---------------------

Both high-level routines expose a configurable backend via ``model_kind``:

- ``"rf"``     : RandomForestRegressor
- ``"knn"``    : KNeighborsRegressor
- ``"linear"`` : LinearRegression
- ``"mlp"``    : MLPRegressor
- ``"svd"``    : TruncatedSVD + LinearRegression (pipeline)
- ``"mcm"``    : Mean Climatology Model (pure temporal baseline, no ML)

Hyper-parameters can be tuned via:

- ``rf_params``     : specific dataclass/dict for random forest
  (e.g. ``n_estimators``, ``max_depth``, ``max_features``,
  ``random_state``).
- ``model_params``  : extra kwargs passed to the chosen backend
  (e.g. ``n_neighbors`` for KNN, ``hidden_layer_sizes`` for MLP).

Public API
----------

High-level imputation / evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :func:`impute_dataset`:
    Fill missing values in a long-format daily dataset using a configurable
    backend. Returns an imputed table that can be used directly in analyses
    or visualized with the plotting helpers.

- :func:`evaluate_stations`:
    Station-wise evaluation of a chosen backend. Produces multi-scale metrics
    (daily, monthly, annual), including KGE and a Mean Climatology Model (MCM)
    baseline, which is particularly useful in MDR-style experiments.

Metrics and baselines
^^^^^^^^^^^^^^^^^^^^^

- :func:`compute_kge`
- :func:`compute_mcm_baseline`
- :func:`multiscale_metrics`

Missingness exploration
^^^^^^^^^^^^^^^^^^^^^^^

- :func:`percent_missing_between`
- :func:`gap_profile_by_station`
- :func:`missing_matrix`
- :func:`describe_missing`
- :func:`apply_random_mask_by_station`

Visualization helpers
^^^^^^^^^^^^^^^^^^^^^

- :func:`plot_missing_matrix`
- :func:`plot_metrics_distribution`
- :func:`plot_parity_scatter`
- :func:`plot_time_series_overlay`
- :func:`plot_spatial_scatter`
- :func:`plot_gap_histogram`
- :func:`plot_imputed_series`
- :func:`plot_imputation_coverage`
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# ---------------------------------------------------------------------------
# Package version
# ---------------------------------------------------------------------------

try:  # pragma: no cover - during development the package may not be installed
    __version__ = version("missclimatepy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# ---------------------------------------------------------------------------
# Public API imports
# ---------------------------------------------------------------------------

from .evaluate import RFParams, evaluate_stations
from .impute import impute_dataset

from .metrics import (
    compute_kge,
    compute_mcm_baseline,
    multiscale_metrics,
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

# Optional short aliases for convenience / backwards-compatibility
# (one-shot helpers mirroring the old API naming)
def impute(*args, **kwargs):
    """Thin alias for :func:`impute_dataset`."""
    return impute_dataset(*args, **kwargs)


def evaluate(*args, **kwargs):
    """Thin alias for :func:`evaluate_stations`."""
    return evaluate_stations(*args, **kwargs)


__all__ = [
    "__version__",
    # Core modeling
    "RFParams",
    "impute_dataset",
    "evaluate_stations",
    "impute",         # convenience alias
    "evaluate",       # convenience alias
    # Metrics / baseline
    "compute_kge",
    "compute_mcm_baseline",
    "multiscale_metrics",
    # Missingness exploration
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
]
