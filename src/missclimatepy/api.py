# src/missclimatepy/api.py
"""
Public API re-exports and compatibility shims.

Users can:
- run_quickstart / QuickstartConfig
- evaluate_all_stations_fast / RFParams
- neighbor_distances / build_neighbor_map
- MissClimateImputer  (back-compat thin wrapper)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union, List
import pandas as pd

from .quickstart import run_quickstart, QuickstartConfig
from .evaluate import evaluate_all_stations_fast, RFParams
from .neighbors import neighbor_distances, build_neighbor_map


class MissClimateImputer:
    """
    Back-compat thin wrapper used in legacy tests.

    It does NOT hold a persistent fitted model per station;
    instead, it runs the evaluation pipeline and exposes the report.
    """

    def __init__(
        self,
        *,
        id_col: str = "station",
        date_col: str = "date",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude",
        target_col: str = "tmin",
        k_neighbors: Optional[int] = 20,
        include_target_pct: float = 0.0,
        add_cyclic: bool = False,
        feature_cols: Optional[List[str]] = None,
        min_station_rows: Optional[int] = None,
        rf_params: Optional[Union[RFParams, dict]] = None,
    ) -> None:
        self.id_col = id_col
        self.date_col = date_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.alt_col = alt_col
        self.target_col = target_col
        self.k_neighbors = k_neighbors
        self.include_target_pct = include_target_pct
        self.add_cyclic = add_cyclic
        self.feature_cols = feature_cols
        self.min_station_rows = min_station_rows
        self.rf_params = rf_params if rf_params is not None else RFParams()

        self.report_: Optional[pd.DataFrame] = None

    def fit(
        self,
        data: pd.DataFrame,
        *,
        start: Optional[str] = None,
        end: Optional[str] = None,
        station_ids: Optional[Sequence[Union[str, int]]] = None,
        show_progress: bool = False,
    ) -> "MissClimateImputer":
        """Run the per-station evaluation and store the report."""
        self.report_ = evaluate_all_stations_fast(
            data,
            id_col=self.id_col,
            date_col=self.date_col,
            lat_col=self.lat_col,
            lon_col=self.lon_col,
            alt_col=self.alt_col,
            target_col=self.target_col,
            start=start,
            end=end,
            add_cyclic=self.add_cyclic,
            feature_cols=self.feature_cols,
            station_ids=station_ids,
            min_station_rows=self.min_station_rows,
            k_neighbors=self.k_neighbors,
            include_target_pct=self.include_target_pct,
            rf_params=self.rf_params,
            show_progress=show_progress,
        )
        return self

    def report(self) -> pd.DataFrame:
        if self.report_ is None:
            raise RuntimeError("Call .fit(data, ...) before requesting the report.")
        return self.report_


__all__ = [
    "run_quickstart",
    "QuickstartConfig",
    "evaluate_all_stations_fast",
    "RFParams",
    "neighbor_distances",
    "build_neighbor_map",
    "MissClimateImputer",
]
