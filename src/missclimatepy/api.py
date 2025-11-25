# SPDX-License-Identifier: MIT
"""
missclimatepy.api
=================

High-level, public interface for MissClimatePy.

This module provides convenient entry points that wrap the core
functionality implemented in the internal submodules:

- :func:`evaluate` / :func:`evaluate_xyzt`
    Station-wise model evaluation using XYZT-style features.

- :func:`impute` / :func:`impute_xyzt`
    Local imputation of a single target variable per station, producing
    complete series (observed + imputed) on a daily grid.

- :func:`build_neighbor_map`
    Spatial KNN neighbor map in (lat, lon) degrees.

It also re-exports the main visualization helpers from
:mod:`missclimatepy.viz`, so that users can access them directly via
``missclimatepy.api`` if they prefer a single public entry point.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
)

import pandas as pd

from .neighbors import build_neighbor_map
from .evaluate import evaluate_stations, RFParams
from .impute import impute_dataset

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
# Public XYZT wrappers (evaluation / imputation)
# --------------------------------------------------------------------------- #


def evaluate_xyzt(
    data: pd.DataFrame,
    *,
    # column names (generic schema)
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # temporal window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # feature config
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection (OR semantics)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    min_station_rows: Optional[int] = None,
    # neighborhood & target leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,  # 0 = exclude; 1..95 = include %
    include_target_seed: int = 42,
    # model & metrics
    model_kind: str = "rf",
    model_params: Optional[Mapping[str, Any]] = None,
    agg_for_metrics: str = "sum",
    # logging / UX
    show_progress: bool = False,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    # outputs
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate an XYZT-style model per station.

    This is a thin, typed wrapper around
    :func:`missclimatepy.evaluate.evaluate_stations`. See that function
    for detailed semantics and implementation notes.

    Parameters
    ----------
    data : DataFrame
        Long-format table with at least:
        [id_col, date_col, lat_col, lon_col, alt_col, target_col].
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, coordinates and target.

    Other parameters
    ----------------
    All remaining parameters are passed verbatim to
    :func:`evaluate_stations`.

    Returns
    -------
    (report, preds)
        Identical to :func:`evaluate_stations`:

        - report : one row per station with metrics and metadata.
        - preds  : per-row predictions with coordinates and observed/modelled
          values.
    """
    return evaluate_stations(
        data,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=start,
        end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
        min_station_rows=min_station_rows,
        k_neighbors=k_neighbors,
        neighbor_map=neighbor_map,
        include_target_pct=include_target_pct,
        include_target_seed=include_target_seed,
        model_kind=model_kind,
        model_params=model_params,
        agg_for_metrics=agg_for_metrics,
        show_progress=show_progress,
        log_csv=log_csv,
        flush_every=flush_every,
        save_table_path=save_table_path,
        parquet_compression=parquet_compression,
    )


def impute_xyzt(
    data: pd.DataFrame,
    *,
    # schema
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection (optional, OR semantics)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # neighbors
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    # MDR (minimum observed rows within the window)
    min_station_rows: Optional[int] = None,
    # how much of the target station history the model can see
    include_target_pct: Optional[float] = None,
    # model (RandomForest hyperparameters)
    rf_params: Optional[Union[RFParams, Dict[str, Any]]] = None,
    # logging
    show_progress: bool = False,
    # persistence
    save_path: Optional[str] = None,
    save_format: Literal["csv", "parquet", "auto"] = "auto",
    save_index: bool = False,
    save_partitions: bool = False,
) -> pd.DataFrame:
    """
    Impute a single target variable for the selected stations.

    This is a thin, typed wrapper around
    :func:`missclimatepy.impute.impute_dataset`. See that function
    for detailed semantics and implementation notes.

    Parameters
    ----------
    data : DataFrame
        Long-format input with at least:
        [id_col, date_col, lat_col, lon_col, alt_col, target_col].
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, coordinates and target.

    Other parameters
    ----------------
    All remaining parameters are passed verbatim to
    :func:`impute_dataset`.

    Returns
    -------
    DataFrame
        Minimal long-format table with columns:

        [station, date, latitude, longitude, altitude, <target_col>, source]

        where ``source ∈ {"observed", "imputed"}``.
    """
    return impute_dataset(
        data,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=start,
        end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
        k_neighbors=k_neighbors,
        neighbor_map=neighbor_map,
        min_station_rows=min_station_rows,
        include_target_pct=include_target_pct,
        rf_params=rf_params,
        show_progress=show_progress,
        save_path=save_path,
        save_format=save_format,
        save_index=save_index,
        save_partitions=save_partitions,
    )


# Canonical short names for the public API
#: Canonical evaluation entry point.
evaluate = evaluate_xyzt

#: Canonical imputation entry point.
impute = impute_xyzt


__all__ = [
    # Core XYZT API
    "evaluate_xyzt",
    "evaluate",
    "impute_xyzt",
    "impute",
    "build_neighbor_map",
    # Visualization helpers (re-exported for convenience)
    "plot_missing_matrix",
    "plot_metrics_distribution",
    "plot_parity_scatter",
    "plot_time_series_overlay",
    "plot_spatial_scatter",
    "plot_gap_histogram",
    "plot_imputed_series",
    "plot_imputation_coverage",
]
