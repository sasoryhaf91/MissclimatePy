# src/missclimatepy/__init__.py
from .api import MissClimateImputer
from .quickstart import run_quickstart
from .prepare import enforce_schema, filter_period, missing_summary, select_stations
from .neighbors import neighbor_distances
from .evaluate import evaluate_per_station
from .mdr import mdr_grid_search
from .viz import plot_station_series, plot_metrics_distribution

__all__ = [
    "MissClimateImputer",
    "run_quickstart",
    "enforce_schema", "filter_period", "missing_summary", "select_stations",
    "neighbor_distances",
    "evaluate_per_station",
    "mdr_grid_search",
    "plot_station_series", "plot_metrics_distribution",
]

__version__ = "0.1.0"




