# SPDX-License-Identifier: MIT
"""
Quickstart runner for missclimatepy.

- Normalizes altitude/elevation (prefers 'altitude').
- Optionally builds neighbor map (k_neighbors).
- Calls evaluate_all_stations_fast (daily metrics; low memory).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Dict, List

import pandas as pd

from .evaluate import evaluate_all_stations_fast
from .neighbors import build_neighbor_map


@dataclass
class QuickstartConfig:
    data_path: str
    target: str
    period: Tuple[str, str]
    stations: Optional[Iterable[str]] = None     # None -> evaluate all
    k_neighbors: Optional[int] = None            # None -> global training
    include_target_pct: float = 0.0              # 0=LOSO, 1..95 allowed
    min_station_rows: Optional[int] = None       # filter by valid rows (after feature NA drop)
    outputs_dir: str = "/kaggle/working/mcp"
    verbose: bool = True
    title_tag: str = ""


def _run(cfg: QuickstartConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)

    # Period clip early (save memory when CSV is huge)
    if "date" not in df.columns:
        raise ValueError("Input must contain column 'date'.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    lo, hi = pd.to_datetime(cfg.period[0]), pd.to_datetime(cfg.period[1])
    df = df[(df["date"] >= lo) & (df["date"] <= hi)].copy()

    # Basic columns existence
    base_cols = ["station", "latitude", "longitude"]
    for c in base_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in input CSV.")

    # Unify altitude/elevation (prefer 'altitude')
    if "altitude" not in df.columns and "elevation" in df.columns:
        df = df.rename(columns={"elevation": "altitude"})
    if "altitude" not in df.columns:
        raise ValueError("Input must contain 'altitude' (or 'elevation').")

    # Optional station subset
    station_ids = list(cfg.stations) if cfg.stations else None

    # Neighbor map (optional)
    neighbor_map: Dict[int, List[int]] = {}
    if cfg.k_neighbors is not None and cfg.k_neighbors > 0:
        neighbor_map = build_neighbor_map(
            df, id_col="station", lat_col="latitude", lon_col="longitude",
            k_neighbors=int(cfg.k_neighbors)
        )

    # Evaluate (daily metrics only; low memory)
    res = evaluate_all_stations_fast(
        df,
        id_col="station", date_col="date",
        lat_col="latitude", lon_col="longitude", alt_col="altitude",
        target_col=cfg.target,
        station_ids=station_ids,
        start=cfg.period[0], end=cfg.period[1],
        k_neighbors=cfg.k_neighbors,
        neighbor_map=neighbor_map or None,
        include_target_pct=cfg.include_target_pct,
        min_station_rows=cfg.min_station_rows,
        show_progress=True,
    )
    return res


def run_quickstart(**kwargs) -> pd.DataFrame:
    """Build a QuickstartConfig from kwargs and execute quickstart."""
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)
