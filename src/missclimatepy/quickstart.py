# src/missclimatepy/quickstart.py
# SPDX-License-Identifier: MIT
"""
missclimatepy.quickstart
========================

Small, opinionated façade around :func:`evaluate_all_stations_fast` to let users
run a full evaluation with a minimal set of arguments.

It intentionally avoids schema normalization: the user tells us their column
names (station, date, latitude, longitude, altitude, target).

Example
-------
>>> from missclimatepy.quickstart import run_quickstart
>>> report = run_quickstart(
...     data_path="/path/to/data.csv",
...     target="tmin",
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     period=("1991-01-01","2020-12-31"),
...     station_ids=["2038","2124","29007"],
...     k_neighbors=20,
...     include_target_pct=30.0,          # 0..95
...     min_station_rows=9125,
...     show_progress=True,
... )
>>> report.head()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd

from .evaluate import evaluate_all_stations_fast, RFParams


@dataclass
class QuickstartConfig:
    """
    Container for quickstart parameters.

    Notes
    -----
    - `rf_params` uses `default_factory` to avoid mutable-default errors.
    - `data_path` may be a str (CSV or Parquet) **or** a DataFrame (already loaded).
    """
    # input
    data_path: Union[str, pd.DataFrame]

    # column names (generic schema; user-provided)
    id_col: str = "station"
    date_col: str = "date"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    alt_col: str = "altitude"
    target: str = "tmin"

    # period
    period: Optional[Tuple[Optional[str], Optional[str]]] = None  # (start, end)

    # selection
    station_ids: Optional[Sequence[Union[str, int]]] = None
    prefix: Optional[Iterable[str]] = None
    regex: Optional[str] = None
    custom_filter: Optional[callable] = None
    min_station_rows: Optional[int] = None

    # features
    add_cyclic: bool = False
    feature_cols: Optional[List[str]] = None

    # neighborhood & leakage
    k_neighbors: Optional[int] = 20
    include_target_pct: float = 0.0  # 0..95
    include_target_seed: int = 42

    # model
    rf_params: RFParams = field(default_factory=RFParams)

    # metrics/agg
    agg_for_metrics: str = "sum"  # "sum"|"mean"|"median"

    # UX / logging
    show_progress: bool = False
    log_csv: Optional[str] = None
    flush_every: int = 20

    # output
    save_table_path: Optional[str] = None
    parquet_compression: str = "snappy"


def _load_data(obj: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """Load CSV/Parquet path or pass through DataFrame."""
    if isinstance(obj, pd.DataFrame):
        return obj
    path = str(obj).lower()
    if path.endswith(".parquet") or path.endswith(".pq"):
        return pd.read_parquet(obj)
    # default to CSV (handles .csv.gz too)
    return pd.read_csv(obj)


def _run(cfg: QuickstartConfig) -> pd.DataFrame:
    """Internal runner that wires QuickstartConfig into evaluate_all_stations_fast."""
    df = _load_data(cfg.data_path)

    start, end = (None, None)
    if cfg.period is not None:
        start, end = cfg.period

    report = evaluate_all_stations_fast(
        df,
        # schema
        id_col=cfg.id_col,
        date_col=cfg.date_col,
        lat_col=cfg.lat_col,
        lon_col=cfg.lon_col,
        alt_col=cfg.alt_col,
        target_col=cfg.target,
        # period
        start=start,
        end=end,
        # features
        add_cyclic=cfg.add_cyclic,
        feature_cols=cfg.feature_cols,
        # selection
        prefix=cfg.prefix,
        station_ids=cfg.station_ids,
        regex=cfg.regex,
        custom_filter=cfg.custom_filter,
        min_station_rows=cfg.min_station_rows,
        # neighborhood & leakage
        k_neighbors=cfg.k_neighbors,
        include_target_pct=cfg.include_target_pct,
        include_target_seed=cfg.include_target_seed,
        # model & metrics
        rf_params=cfg.rf_params,
        agg_for_metrics=cfg.agg_for_metrics,
        # UX
        show_progress=cfg.show_progress,
        log_csv=cfg.log_csv,
        flush_every=cfg.flush_every,
        # output
        save_table_path=cfg.save_table_path,
        parquet_compression=cfg.parquet_compression,
    )
    return report


def run_quickstart(**kwargs) -> pd.DataFrame:
    """
    Convenience entry point. Accepts any :class:`QuickstartConfig` field as a keyword
    and returns the evaluation report (one row per station).

    Returns
    -------
    pandas.DataFrame
        Sorted by daily RMSE ascending.
    """
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)


__all__ = ["QuickstartConfig", "run_quickstart"]
