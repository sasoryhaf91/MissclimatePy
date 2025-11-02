# src/missclimatepy/__init__.py
from .api import MissClimateImputer
from .quickstart import run_quickstart
from .prepare import enforce_schema, filter_period, missing_summary, select_stations
from .neighbors import neighbor_distances
from .evaluate import evaluate_per_station
from .mdr import inclusion_sweep, recommend_min_inclusion
from .viz import plot_inclusion_curves, plot_inclusion_aggregate

def mdr_grid_search(
    df,
    target: str,
    include_pcts=(0.0, 0.04, 0.10, 0.20, 0.40, 0.60, 0.80),
    period=None,
    stations=None,
    min_obs: int = 60,
    engine: str = "rf",
    model_params: dict | None = None,
    thresholds: dict | None = None,
):
    sweep_df = inclusion_sweep(
        df=df,
        target=target,
        include_pcts=include_pcts,
        period=period,
        stations=stations,
        min_obs=min_obs,
        engine=engine,
        model_params=model_params,
    )
    rec_df = recommend_min_inclusion(
        sweep_df,
        thresholds=thresholds or {"R2": 0.5, "RMSE": 2.0},
    )
    return sweep_df, rec_df

def plot_station_series(
    sweep_df,
    station: str,
    metrics=("RMSE", "R2"),
    out_png: str | None = None,
    title_extra: str = "",
):
    return plot_inclusion_curves(
        sweep_df=sweep_df,
        station=str(station),
        metrics=metrics,
        out_png=out_png,
        title_extra=title_extra,
    )

def plot_metrics_distribution(
    sweep_df,
    metric: str = "RMSE",
    out_png: str | None = None,
    quantiles=(0.1, 0.5, 0.9),
    title: str = "Aggregate inclusion curves",
):
    return plot_inclusion_aggregate(
        sweep_df=sweep_df,
        metric=metric,
        out_png=out_png,
        quantiles=quantiles,
        title=title,
    )

__all__ = [
    "MissClimateImputer",
    "run_quickstart",
    "enforce_schema", "filter_period", "missing_summary", "select_stations",
    "neighbor_distances",
    "evaluate_per_station",
    "inclusion_sweep", "recommend_min_inclusion",
    "plot_inclusion_curves", "plot_inclusion_aggregate",
    "mdr_grid_search",
    "plot_station_series", "plot_metrics_distribution",
]

__version__ = "0.1.1"






