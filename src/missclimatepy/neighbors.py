# src/missclimatepy/neighbors.py
# SPDX-License-Identifier: MIT
"""
Spatial neighbor search utilities for MissclimatePy.

This module exposes composable helpers to find nearest stations using
great-circle distances (Haversine) on latitude/longitude, with optional
filters on maximum horizontal radius and maximum altitude difference.

Key functions
-------------
neighbor_distances : Return a tidy DataFrame with k nearest neighbors per station.
build_neighbor_map : Return a dict {station_id: [neighbor_ids...]} for fast lookup.

Design notes
------------
- Haversine distances require radians; we convert internally.
- Distances are returned in kilometers using a precise Earth radius.
- Self matches are removed by default.
- Optional altitude filtering allows restricting neighbors by |Δ altitude|.
- Deterministic, order-stable results for reproducibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# Mean Earth radius in kilometers (IUGG recommended value)
EARTH_RADIUS_KM: float = 6371.0088


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _latlon_to_radians(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Stack (lat, lon) and convert to radians for BallTree(haversine)."""
    return np.deg2rad(np.c_[lat, lon])


def _safe_unique_ordered(values: Sequence) -> List:
    """Return unique values preserving first occurrence order."""
    seen = set()
    out: List = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def neighbor_distances(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    altitude_col: Optional[str] = None,
    k_neighbors: int = 5,
    max_radius_km: Optional[float] = None,
    max_abs_altitude_diff: Optional[float] = None,
    include_self: bool = False,
) -> pd.DataFrame:
    """
    Compute k nearest neighbor stations for each station using Haversine distance.

    Parameters
    ----------
    df : DataFrame
        Station metadata with columns for id, latitude, longitude, and
        optionally altitude.
        Each row must correspond to a single station.
    id_col, lat_col, lon_col : str
        Column names for station id and geographic coordinates (in degrees).
    altitude_col : str or None
        Optional altitude column name (in meters). If provided together with
        `max_abs_altitude_diff`, neighbors whose absolute altitude difference
        exceeds the threshold will be excluded.
    k_neighbors : int
        Number of neighbors to return per station. If `include_self=False`
        (default), the function internally queries k+1 and drops self-matches
        to guarantee up to k neighbors.
    max_radius_km : float or None
        Optional maximum great-circle distance (km) for neighbors. Candidates
        outside this radius are discarded.
    max_abs_altitude_diff : float or None
        Optional maximum absolute altitude difference (same units as altitude
        column, typically meters). Candidates exceeding the threshold are
        discarded.
    include_self : bool
        If True, a station may appear as its own neighbor at distance 0.
        By default, self is removed.

    Returns
    -------
    DataFrame
        Tidy table with columns:
        - station         : station id (source)
        - neighbor        : neighbor station id (target)
        - distance_km     : great-circle distance in kilometers
        - rank            : 1..k rank by ascending distance (ties broken by id)
        - altitude_diff   : neighbor_altitude - station_altitude (if available)

        Rows with no available neighbors after filtering are omitted.

    Notes
    -----
    - Distances are computed with sklearn BallTree(haversine), which expects
      coordinates in radians and returns angular distance in radians. We convert
      to kilometers via multiplication by EARTH_RADIUS_KM.
    - The function preserves determinism by breaking equal-distance ties with
      neighbor id ordering.
    """
    _require_columns(df, [id_col, lat_col, lon_col])
    if k_neighbors <= 0:
        raise ValueError("k_neighbors must be a positive integer.")

    # Prepare arrays
    meta = df[[id_col, lat_col, lon_col] + ([altitude_col] if altitude_col else [])].copy()
    station_ids = meta[id_col].astype(object).to_numpy()
    lat = pd.to_numeric(meta[lat_col], errors="coerce").to_numpy()
    lon = pd.to_numeric(meta[lon_col], errors="coerce").to_numpy()

    valid_mask = (~np.isnan(lat)) & (~np.isnan(lon))
    if not np.all(valid_mask):
        # Drop rows with invalid coordinates
        meta = meta.loc[valid_mask].reset_index(drop=True)
        station_ids = station_ids[valid_mask]
        lat = lat[valid_mask]
        lon = lon[valid_mask]

    coords_rad = _latlon_to_radians(lat, lon)
    tree = BallTree(coords_rad, metric="haversine")

    # If excluding self, we query k+1 to ensure k non-self neighbors when possible
    query_k = k_neighbors if include_self else min(k_neighbors + 1, coords_rad.shape[0])

    # Query all stations at once
    distances_rad, indices = tree.query(coords_rad, k=query_k)
    distances_km = distances_rad * EARTH_RADIUS_KM

    # Optional altitude vector
    alt = None
    if altitude_col and altitude_col in meta.columns:
        alt = pd.to_numeric(meta[altitude_col], errors="coerce").to_numpy()

    rows: List[Dict] = []
    n = coords_rad.shape[0]

    for i in range(n):
        sid = station_ids[i]
        # Build candidate list (neighbor id, distance_km, altitude_diff?)
        cand_ids = station_ids[indices[i]]
        cand_dist = distances_km[i]

        # Optionally remove self
        keep = np.ones_like(cand_dist, dtype=bool)
        if not include_self:
            keep &= cand_ids != sid

        # Apply maximum radius
        if max_radius_km is not None:
            keep &= cand_dist <= max_radius_km

        # Apply altitude difference filter
        if alt is not None and max_abs_altitude_diff is not None:
            # Δalt = alt_j - alt_i
            delta_alt = alt[indices[i]] - alt[i]
            keep &= np.abs(delta_alt) <= max_abs_altitude_diff
        else:
            delta_alt = None  # type: ignore[assignment]

        if not np.any(keep):
            continue

        kept_ids = cand_ids[keep]
        kept_dist = cand_dist[keep]
        kept_delta = (delta_alt[keep] if delta_alt is not None else None)

        # Sort by (distance, neighbor id) for determinism
        order = np.lexsort((kept_ids, kept_dist))
        kept_ids = kept_ids[order]
        kept_dist = kept_dist[order]
        if kept_delta is not None:
            kept_delta = kept_delta[order]

        # Cap to k_neighbors
        cap = min(k_neighbors, kept_ids.size)
        for rank, (nid, dkm) in enumerate(zip(kept_ids[:cap], kept_dist[:cap]), start=1):
            record = {
                "station": sid,
                "neighbor": nid,
                "distance_km": float(dkm),
                "rank": int(rank),
            }
            if kept_delta is not None:
                record["altitude_diff"] = float(kept_delta[rank - 1])
            rows.append(record)

    if not rows:
        # No neighbors found under filters
        return pd.DataFrame(columns=["station", "neighbor", "distance_km", "rank"] + (["altitude_diff"] if altitude_col else []))

    out = pd.DataFrame(rows)

    # Enforce column order and dtypes
    columns = ["station", "neighbor", "distance_km", "rank"]
    if "altitude_diff" in out.columns:
        columns.append("altitude_diff")
    out = out[columns].sort_values(["station", "rank", "neighbor"]).reset_index(drop=True)
    out["rank"] = out["rank"].astype(int)
    out["distance_km"] = out["distance_km"].astype(float)
    if "altitude_diff" in out.columns:
        out["altitude_diff"] = out["altitude_diff"].astype(float)
    return out


def build_neighbor_map(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    altitude_col: Optional[str] = None,
    k_neighbors: int = 5,
    max_radius_km: Optional[float] = None,
    max_abs_altitude_diff: Optional[float] = None,
    include_self: bool = False,
) -> Dict[object, List[object]]:
    """
    Build a dictionary mapping each station id to an ordered list of neighbor ids.

    This is a thin wrapper around `neighbor_distances` that returns data in a
    structure convenient for per-station modeling loops.

    Parameters
    ----------
    df : DataFrame
        Station metadata table; one row per station.
    id_col, lat_col, lon_col : str
        Column names for station id and geographic coordinates.
    altitude_col : str or None
        Optional altitude column used only for filtering by `max_abs_altitude_diff`.
    k_neighbors : int
        Number of neighbors to keep per station.
    max_radius_km : float or None
        Optional max great-circle distance in kilometers.
    max_abs_altitude_diff : float or None
        Optional max |Δ altitude| threshold (same units as altitude column).
    include_self : bool
        Whether to allow the station to appear as its own neighbor.

    Returns
    -------
    dict
        { station_id: [neighbor_id_1, neighbor_id_2, ..., neighbor_id_k] }
        Only stations with at least one neighbor under the filters are present.

    Notes
    -----
    - The neighbor list is ordered by ascending Haversine distance, with
      deterministic tie-breakers on neighbor id.
    - If fewer than `k_neighbors` neighbors satisfy the filters, the list
      can be shorter.
    """
    table = neighbor_distances(
        df,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        altitude_col=altitude_col,
        k_neighbors=k_neighbors,
        max_radius_km=max_radius_km,
        max_abs_altitude_diff=max_abs_altitude_diff,
        include_self=include_self,
    )

    mapping: Dict[object, List[object]] = {}
    if table.empty:
        return mapping

    for sid, sub in table.groupby("station"):
        # Already sorted by rank, then neighbor id
        mapping[sid] = sub["neighbor"].tolist()

    return mapping


__all__ = [
    "neighbor_distances",
    "build_neighbor_map",
    "EARTH_RADIUS_KM",
]
