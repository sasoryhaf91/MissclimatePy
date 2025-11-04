# src/missclimatepy/quickstart.py
"""
Quickstart wrapper
------------------

One function to go from a CSV/DataFrame to a metrics report, exposing:
- column mapping,
- K neighbors,
- include_target_pct,
- RF hyperparameters.

This is what most users will call first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import pandas as pd

from .evaluate import evaluate_all_stations_fast, RFParams


@dataclass
class QuickstartConfig:
    # Data
    data_path: Optional[str] = None           # if None, use `data` argument
    data: Optional[pd.DataFrame] = None

    # Column names
    id_col: str = "station"
    date_col: str = "date"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    alt_col: str = "altitude"
    target_col: str = "tmin"

    # Period
    start: Optional[str] = None
    end: Optional[str] = None

    # Selection
    station_ids: Optional[Sequence[Union[str, int]]] = None
    min_station_rows: Optional[int] = None

    # Neighborhood & inclusion
    k_neighbors: Optional[int] = 20
    include_target_pct: float = 0.0
    include_target_seed: int = 42

    # Features / model
    add_cyclic: bool = False
    feature_cols: Optional[List[str]] = None
    rf_params: RFParams = RFParams(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)

    # Output
    show_progress: bool = True
    log_csv: Optional[str] = None
    flush_every: int = 20
    save_table_path: Optional[str] = None


def run_quickstart(**kwargs) -> pd.DataFrame:
    """
    Run a one-shot evaluation based on keyword args matching QuickstartConfig.
    """
    cfg = QuickstartConfig(**kwargs)

    if cfg.data is None:
        if not cfg.data_path:
            raise ValueError("Either 'data' (DataFrame) or 'data_path' must be provided.")
        ext = str(cfg.data_path).lower()
        if ext.endswith(".parquet"):
            df = pd.read_parquet(cfg.data_path)
        else:
            df = pd.read_csv(cfg.data_path)
    else:
        df = cfg.data

    report = evaluate_all_stations_fast(
        df,
        id_col=cfg.id_col, date_col=cfg.date_col,
        lat_col=cfg.lat_col, lon_col=cfg.lon_col, alt_col=cfg.alt_col,
        target_col=cfg.target_col,
        start=cfg.start, end=cfg.end,
        add_cyclic=cfg.add_cyclic, feature_cols=cfg.feature_cols,
        station_ids=cfg.station_ids,
        min_station_rows=cfg.min_station_rows,
        k_neighbors=cfg.k_neighbors,
        include_target_pct=cfg.include_target_pct,
        include_target_seed=cfg.include_target_seed,
        rf_params=cfg.rf_params,
        show_progress=cfg.show_progress,
        log_csv=cfg.log_csv,
        flush_every=cfg.flush_every,
        save_table_path=cfg.save_table_path,
    )
    return report


__all__ = ["QuickstartConfig", "run_quickstart"]
