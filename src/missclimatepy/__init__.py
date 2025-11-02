# src/missclimatepy/__init__.py
"""
Public entry points for MissclimatePy.

- Keeps backward compatibility with legacy constructor arguments used in tests
  (model, n_estimators, max_depth, min_samples_leaf, rf_params, ...).
- Makes optional modules (mdr, viz) safe to import in CI (skip if missing).
"""

from __future__ import annotations

from .api import MissClimateImputer as _MissClimateImputer
from .quickstart import run_quickstart
from .prepare import (
    enforce_schema,
    filter_period,
    missing_summary,
    select_stations,
)
from .neighbors import neighbor_distances
from .evaluate import evaluate_per_station

# Optional imports (do not fail CI if extra deps are not present)
try:
    from .mdr import mdr_grid_search  # type: ignore
    _HAS_MDR = True
except Exception:  # pragma: no cover
    mdr_grid_search = None  # type: ignore
    _HAS_MDR = False

try:
    from .viz import plot_station_series, plot_metrics_distribution  # type: ignore
    _HAS_VIZ = True
except Exception:  # pragma: no cover
    plot_station_series = None  # type: ignore
    plot_metrics_distribution = None  # type: ignore
    _HAS_VIZ = False


__version__ = "0.1.1"


def MissClimateImputer(  # type: ignore
    engine: str = "rf",
    target: str = "tmin",
    k_neighbors: int = 20,
    min_obs_per_station: int = 50,
    n_jobs: int = -1,
    random_state: int = 42,
    # ---- backward-compat (tests / legacy) ----
    model: str | None = None,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    min_samples_leaf: int | None = None,
    rf_params: dict | None = None,
    model_params: dict | None = None,
):
    """
    Backward-compatible constructor wrapper.

    Accepts both the new interface:
        MissClimateImputer(engine="rf", target="tmin", k_neighbors=20, ...)

    and the legacy arguments used in tests:
        MissClimateImputer(model="rf", n_estimators=100, max_depth=25, ...)

    Parameters
    ----------
    engine : str
        Engine name (currently 'rf'). If `model` is provided, it overrides this.
    target : str
        Target variable to impute (e.g., 'tmin').
    k_neighbors : int
        Number of neighbors used by the local engine.
    min_obs_per_station : int
        Minimum visible observations required at the station for training
        (use 0 to allow strict LOSO).
    n_jobs : int
        Parallel jobs for the engine (if supported).
    random_state : int
        Random seed.
    model, n_estimators, max_depth, min_samples_leaf, rf_params :
        Legacy/compatibility arguments; they are merged into `rf_params`.
    model_params : dict | None
        Extra engine parameters (kept for forward compatibility).

    Returns
    -------
    missclimatepy.api.MissClimateImputer
        Real implementation instance.
    """
    # Legacy alias: model -> engine
    if model is not None:
        engine = model

    # Merge legacy scalar hyperparams into rf_params
    merged_rf = dict(rf_params or {})
    if n_estimators is not None:
        merged_rf["n_estimators"] = n_estimators
    if max_depth is not None:
        merged_rf["max_depth"] = max_depth
    if min_samples_leaf is not None:
        merged_rf["min_samples_leaf"] = min_samples_leaf

    # Build instance from the real class
    return _MissClimateImputer(
        engine=engine,
        target=target,
        k_neighbors=k_neighbors,
        min_obs_per_station=min_obs_per_station,
        n_jobs=n_jobs,
        random_state=random_state,
        rf_params=merged_rf,
        model_params=model_params,
    )


# ---- Public API ----
__all__ = [
    "MissClimateImputer",
    "run_quickstart",
    "enforce_schema",
    "filter_period",
    "missing_summary",
    "select_stations",
    "neighbor_distances",
    "evaluate_per_station",
]

# Add optional names only if available (prevents CI import errors)
if _HAS_MDR:
    __all__.append("mdr_grid_search")
if _HAS_VIZ:
    __all__ += ["plot_station_series", "plot_metrics_distribution"]

