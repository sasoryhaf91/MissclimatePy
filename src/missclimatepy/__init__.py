# src/missclimatepy/__init__.py

from .api import MissClimateImputer as _MissClimateImputer
from .quickstart import run_quickstart
from .prepare import enforce_schema, filter_period, missing_summary, select_stations
from .neighbors import neighbor_distances
from .evaluate import evaluate_per_station

# --- Imports opcionales (no romper tests si no están disponibles) ---
try:
    from .mdr import mdr_grid_search  # puede no existir aún
except Exception:  # pragma: no cover
    mdr_grid_search = None  # type: ignore

try:
    from .viz import plot_station_series, plot_metrics_distribution
except Exception:  # pragma: no cover
    plot_station_series = None  # type: ignore
    plot_metrics_distribution = None  # type: ignore

__version__ = "0.1.0"

# Construimos __all__ sólo con lo disponible
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

if mdr_grid_search is not None:
    __all__.append("mdr_grid_search")
if plot_station_series is not None:
    __all__.append("plot_station_series")
if plot_metrics_distribution is not None:
    __all__.append("plot_metrics_distribution")


def MissClimateImputer(  # type: ignore
    engine="rf",
    target="tmin",
    k_neighbors=20,
    min_obs_per_station=50,
    n_jobs=-1,
    random_state=42,
    # ---- retro-compat (tests / legacy) ----
    model=None,
    n_estimators=None,
    max_depth=None,
    min_samples_leaf=None,
    rf_params=None,
    model_params=None,
):
    """
    Back-compat constructor wrapper (Py3.9-safe).
    Acepta la interfaz nueva (engine="rf", rf_params={...}) y también la legada
    usada en los tests (model="rf", n_estimators=..., max_depth=..., etc.).
    """
    # Legacy alias: model -> engine
    if model is not None:
        engine = model

    # Merge legacy scalar hyperparams into rf_params
    merged_rf_params = dict(rf_params or {})
    if n_estimators is not None:
        merged_rf_params["n_estimators"] = n_estimators
    if max_depth is not None:
        merged_rf_params["max_depth"] = max_depth
    if min_samples_leaf is not None:
        merged_rf_params["min_samples_leaf"] = min_samples_leaf

    return _MissClimateImputer(
        engine=engine,
        target=target,
        k_neighbors=k_neighbors,
        min_obs_per_station=min_obs_per_station,
        n_jobs=n_jobs,
        random_state=random_state,
        rf_params=merged_rf_params,
        model_params=model_params,
    )
