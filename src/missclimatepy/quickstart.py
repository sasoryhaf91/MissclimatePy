# SPDX-License-Identifier: MIT
"""
Quickstart runner for missclimatepy.

You can control:
- k_neighbors (None -> global training; int -> restrict to K nearest)
- include_target_pct (0 = strict LOSO; clamp 1..95 if >0)
- RandomForest params: rf_n_estimators, rf_max_depth, rf_n_jobs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Iterable, Tuple, Dict, List

import pandas as pd

from .evaluate import evaluate_all_stations_fast
from .neighbors import build_neighbor_map


@dataclass
class QuickstartConfig:
    # required
    data_path: str
    target: str
    period: Tuple[str, str]

    # optional station subset
    stations: Optional[Iterable[str]] = None  # None -> all

    # neighbor control
    k_neighbors: Optional[int] = None         # None -> global, int -> local K-NN

    # leakage control
    include_target_pct: float = 0.0           # 0=LOSO; 1..95 allowed (clamped)

    # RF controls (the ones you asked for)
    rf_n_estimators: int = 15
    rf_max_depth: Optional[int] = 30          # None -> unlimited depth
    rf_n_jobs: int = -1                       # parallelism

    # misc
    min_station_rows: Optional[int] = None    # filter by valid rows
    outputs_dir: str = "/kaggle/working/mcp"
    verbose: bool = True
    title_tag: str = ""


def _run(cfg: QuickstartConfig) -> pd.DataFrame:
    # 1) load & clip period early (saves memory)
    df = pd.read_csv(cfg.data_path)
    if "date" not in df.columns:
        raise ValueError("Input must contain column 'date'.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    lo, hi = pd.to_datetime(cfg.period[0]), pd.to_datetime(cfg.period[1])
    df = df[(df["date"] >= lo) & (df["date"] <= hi)].copy()

    # 2) required base columns
    for col in ("station", "latitude", "longitude"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in input CSV.")

    # 3) altitude/elevation normalization
    if "altitude" not in df.columns and "elevation" in df.columns:
        df = df.rename(columns={"elevation": "altitude"})
    if "altitude" not in df.columns:
        raise ValueError("Input must contain 'altitude' (or 'elevation').")

    # 4) optional station subset
    station_ids = list(cfg.stations) if cfg.stations else None

    # 5) optional neighbor map
    neighbor_map: Dict[int, List[int]] = {}
    if cfg.k_neighbors is not None and int(cfg.k_neighbors) > 0:
        neighbor_map = build_neighbor_map(
            df, id_col="station", lat_col="latitude", lon_col="longitude",
            k_neighbors=int(cfg.k_neighbors)
        )

    # 6) RF parameters (only what you need)
    rf_params = dict(
        n_estimators=int(cfg.rf_n_estimators),
        max_depth=(None if cfg.rf_max_depth is None else int(cfg.rf_max_depth)),
        n_jobs=int(cfg.rf_n_jobs),
        random_state=42,
        bootstrap=True,
        max_samples=None,
    )

    # 7) evaluate (daily metrics; lean memory)
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
        rf_params=rf_params,
        show_progress=True,
    )
    return res


def run_quickstart(**kwargs) -> pd.DataFrame:
    """Build a QuickstartConfig from kwargs and execute quickstart."""
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)
