# SPDX-License-Identifier: MIT
"""
missclimatepy.api
=================

User-facing convenience API for missclimatepy.

This module exposes:
- Thin wrappers to **compute neighbors** and to **evaluate** station-wise
  Random-Forest models with *controlled inclusion* of the target station.
- A "quickstart" helper that mirrors :func:`missclimatepy.quickstart.run_quickstart`.

The goal is to keep common workflows one-import away:

Examples
--------
>>> import pandas as pd
>>> from missclimatepy.api import (
...     compute_neighbors, evaluate, RFParams, quickstart,
... )

Load data and compute neighbors:
>>> df = pd.read_csv("/path/to/datos.csv")
>>> neigh_tbl, neigh_map = compute_neighbors(
...     df,
...     station_col="station", lat_col="latitude", lon_col="longitude",
...     k_neighbors=20, include_self=False,
... )

Evaluate (daily metrics reported; monthly/annual also computed):
>>> report = evaluate(
...     df,
...     station_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     start="1991-01-01", end="2020-12-31",
...     k_neighbors=20,
...     include_target_pct=30.0,        # 0 = exclude, 1..95 = include %
...     min_station_rows=9125,
...     rf_params=RFParams(n_estimators=15, max_depth=30, random_state=42, n_jobs=-1),
...     neighbor_map=neigh_map,         # optional: reuse precomputed neighbors
...     show_progress=True,
... )
>>> report.head()

Or use the batteries-included quickstart:
>>> report2 = quickstart(
...     data_path="/path/to/datos.csv",
...     station_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target="tmin",
...     period=("1991-01-01", "2020-12-31"),
...     stations=["2038","2124","29007"],
...     k_neighbors=20,
...     include_target_pct=30.0,
...     min_station_rows=9125,
...     rf_params=RFParams(n_estimators=15, max_depth=30, random_state=42, n_jobs=-1),
...     show_progress=True,
... )
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .neighbors import (
    neighbor_distances,
    build_neighbor_map,  # maps {station -> [neighbors...]}
)
from .evaluate import evaluate_all_stations_fast, RFParams
from .quickstart import run_quickstart, QuickstartConfig

StationId = Union[str, int]


# --------------------------------------------------------------------------- #
# Small validators
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in DataFrame: {missing}. "
            f"Available: {list(df.columns)[:10]}..."
        )


# --------------------------------------------------------------------------- #
# Public helpers
# --------------------------------------------------------------------------- #
def compute_neighbors(
    data: pd.DataFrame,
    *,
    station_col: str,
    lat_col: str,
    lon_col: str,
    k_neighbors: int = 20,
    include_self: bool = False,
) -> Tuple[pd.DataFrame, Dict[StationId, List[StationId]]]:
    """
    Compute the **K nearest neighbors** per station using great-circle
    (Haversine) distance over station centroids and return both the **long
    table** and a **neighbor map**.

    Parameters
    ----------
    data : DataFrame
        Full input table; only unique (station, lat, lon) pairs are used.
    station_col, lat_col, lon_col : str
        Column names for station id and coordinates (in degrees).
    k_neighbors : int, default 20
        Number of neighbors per station to return.
    include_self : bool, default False
        If True, allow a station to appear as its own neighbor (rank #1, distance 0).

    Returns
    -------
    (neighbors_table, neighbor_map) : (DataFrame, dict)
        *neighbors_table* has columns: ``[station, neighbor, rank, distance_km]``.
        *neighbor_map* is ``{station -> [neighbor1, neighbor2, ...]}``.

    Notes
    -----
    - Complexity is O(n^2). For very large station networks, consider chunking.
    """
    _require_columns(data, [station_col, lat_col, lon_col])

    # Build a minimal frame with user-provided names -> expected names
    centroids = (
        data[[station_col, lat_col, lon_col]]
        .dropna()
        .drop_duplicates()
        .rename(
            columns={
                station_col: "station",
                lat_col: "latitude",
                lon_col: "longitude",
            }
        )
    )

    tbl = neighbor_distances(
        stations=centroids,
        k_neighbors=k_neighbors,
        include_self=include_self,
    )
    nmap = build_neighbor_map(tbl)

    return tbl, nmap


def evaluate(
    data: pd.DataFrame,
    *,
    # columns
    station_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # period
    start: Optional[str] = None,
    end: Optional[str] = None,
    # selection
    station_ids: Optional[Iterable[StationId]] = None,
    min_station_rows: Optional[int] = None,
    # neighbors & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[StationId, List[StationId]]] = None,
    include_target_pct: float = 0.0,  # 0 = exclude; 1..95 = include %
    include_target_seed: int = 42,
    # model
    rf_params: Optional[RFParams] = None,
    agg_for_metrics: str = "sum",
    # UX
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Evaluate **per-station** models trained on **K nearest neighbors** with
    optional *controlled inclusion* of the target station rows.

    This is a thin wrapper around :func:`missclimatepy.evaluate.evaluate_all_stations_fast`.

    Parameters
    ----------
    data : DataFrame
        Long table with at least the provided columns.
    station_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names in *data* for station id, date, coordinates, altitude,
        and the target variable to reconstruct (e.g., "prec", "tmin").
    start, end : str or None
        Optional date window (inclusive) to clip the analysis.
    station_ids : iterable or None
        Restrict the run to these stations (ids can be str or int).
    min_station_rows : int or None
        Require at least this number of valid rows per station (after feature
        screening) to be evaluated.
    k_neighbors : int or None, default 20
        Build a neighbor map on-the-fly with K neighbors (ignored if
        *neighbor_map* is provided).
    neighbor_map : dict or None
        Precomputed neighbor map: ``{station -> [neighbors...]}``. If provided,
        it takes precedence and *k_neighbors* is ignored.
    include_target_pct : float, default 0.0
        Percentage (0..95) of valid rows from the target station to include in
        training. **0** = exclude target (LOSO-like). Use a small value like 5–30
        when you want mild bias toward the target’s local regime.
    include_target_seed : int, default 42
        Random seed for sampling target rows when *include_target_pct* > 0.
    rf_params : RFParams or None
        Random-Forest hyperparameters. If *None*, a lean default is used.
    agg_for_metrics : {"sum","mean","median"}, default "sum"
        Aggregation when computing monthly/annual metrics.
    show_progress : bool, default False
        Print per-station progress.

    Returns
    -------
    DataFrame
        One row per station with daily/monthly/annual metrics and metadata.

    Notes
    -----
    - Daily metrics columns end in ``_d``, monthly ``_m``, annual ``_y``.
    - Memory use is lean: features are built once and reused per-station.
    """
    _require_columns(
        data,
        [station_col, date_col, lat_col, lon_col, alt_col, target_col],
    )

    # Normalize user column names into the internal expected names
    df = data.rename(
        columns={
            station_col: "station",
            date_col: "date",
            lat_col: "latitude",
            lon_col: "longitude",
            alt_col: "altitude",
            target_col: "target_tmp__",
        }
    )

    # Pass through to core evaluator
    res = evaluate_all_stations_fast(
        df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="target_tmp__",
        rf_params=(asdict(rf_params) if isinstance(rf_params, RFParams) else rf_params),
        agg_for_metrics=agg_for_metrics,
        start=start,
        end=end,
        add_cyclic=False,
        feature_cols=None,
        prefix=None,
        station_ids=list(station_ids) if station_ids is not None else None,
        regex=None,
        custom_filter=None,
        k_neighbors=k_neighbors if neighbor_map is None else None,
        neighbor_map=neighbor_map,
        log_csv=None,
        flush_every=20,
        show_progress=show_progress,
        include_target_pct=float(include_target_pct),
        include_target_seed=int(include_target_seed),
        min_station_rows=min_station_rows,
        save_table_path=None,
        parquet_compression="snappy",
    )
    return res


def quickstart(
    *,
    data_path: str,
    station_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target: str,
    period: Tuple[str, str],
    stations: Optional[Iterable[StationId]] = None,
    k_neighbors: Optional[int] = 20,
    include_target_pct: float = 0.0,
    min_station_rows: Optional[int] = None,
    rf_params: Optional[RFParams] = None,
    outputs_dir: Optional[str] = None,
    plots_dir: Optional[str] = None,
    title_tag: Optional[str] = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Batteries-included helper for first-time users and reproducible runs.
    It mirrors :func:`missclimatepy.quickstart.run_quickstart`, but keeps the
    signature close to the generic API (explicit column names).

    Parameters
    ----------
    data_path : str
        Path to the CSV file with the long-format table.
    station_col, date_col, lat_col, lon_col, alt_col : str
        Column names in the CSV.
    target : str
        Name of the target variable (e.g., "prec", "tmin").
    period : (start, end)
        Date window inclusive, e.g., ``("1991-01-01","2020-12-31")``.
    stations : iterable or None
        Optional subset of station ids to evaluate.
    k_neighbors, include_target_pct, min_station_rows, rf_params :
        Same meaning as in :func:`evaluate`.
    outputs_dir, plots_dir, title_tag : str or None
        Optional output folders and a tag for plot/report titles.
    show_progress : bool
        Print progress.

    Returns
    -------
    DataFrame
        Evaluation report produced by the quickstart pipeline.
    """
    cfg = QuickstartConfig(
        data_path=data_path,
        # explicit schema
        station_col=station_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target=target,
        period=period,
        stations=None if stations is None else list(stations),
        k_neighbors=k_neighbors,
        include_target_pct=float(include_target_pct),
        min_station_rows=min_station_rows,
        rf_params=(asdict(rf_params) if isinstance(rf_params, RFParams) else rf_params),
        outputs_dir=outputs_dir,
        plots_dir=plots_dir or "plots",
        title_tag=title_tag or "",
        show_progress=show_progress,
    )
    return run_quickstart(cfg)


# Re-export key types for convenience from the API entry point
__all__ = [
    "RFParams",
    "compute_neighbors",
    "evaluate",
    "quickstart",
]
