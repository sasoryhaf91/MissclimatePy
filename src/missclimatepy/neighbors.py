# src/missclimatepy/neighbors.py
"""
Neighbor utilities (Haversine KNN)
----------------------------------

This module exposes a single public function:

    neighbor_distances(stations, k_neighbors=20, radius_km=6371.0088, include_self=False)

It computes K nearest neighbors per station using great-circle distances.
Only 'station', 'latitude', and 'longitude' are required. 'altitude' can be
present but is not used for distance here (kept purely local and fast).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List

__all__ = ["neighbor_distances"]


def _to_radians(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x.astype(float))


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray, radius_km: float = 6371.0088) -> np.ndarray:
    """
    Pairwise Haversine distance (km). Returns an (n,n) matrix.
    """
    lat_r = _to_radians(lat)
    lon_r = _to_radians(lon)

    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return radius_km * c


def neighbor_distances(
    stations: pd.DataFrame,
    *,
    k_neighbors: int = 20,
    radius_km: float = 6371.0088,
    include_self: bool = False,
) -> pd.DataFrame:
    """
    Compute K nearest neighbors per station (Haversine distance).

    Parameters
    ----------
    stations : DataFrame
        Must contain at least: 'station', 'latitude', 'longitude'.
        Duplicates by (station, latitude, longitude) are dropped.
    k_neighbors : int, default 20
        Number of neighbors to return per station (clipped to dataset size).
    radius_km : float, default 6371.0088
        Earth radius in kilometers.
    include_self : bool, default False
        If True, a station may appear as its own neighbor with distance 0.

    Returns
    -------
    DataFrame
        Columns: ['station', 'neighbor', 'rank', 'distance_km'] with rank 1..K.
    """
    required = {"station", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"neighbor_distances: missing columns {sorted(missing)}")

    # Unique coordinate per station
    df = (
        stations[["station", "latitude", "longitude"]]
        .drop_duplicates(subset=["station", "latitude", "longitude"])
        .reset_index(drop=True)
    )
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    # Pairwise distances
    D = _haversine_matrix(df["latitude"].to_numpy(), df["longitude"].to_numpy(), radius_km=radius_km)

    # Exclude diagonal if requested
    if not include_self:
        np.fill_diagonal(D, np.inf)

    # Effective K
    k_eff = int(min(max(k_neighbors, 0), n if include_self else max(0, n - 1)))
    if k_eff == 0:
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    # Indices of the K smallest distances per row
    nbr_idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]

    rows: List[tuple] = []
    stations_arr = df["station"].astype(str).to_numpy()
    for i in range(n):
        idxs = nbr_idx[i]
        dists = D[i, idxs]
        order = np.argsort(dists)
        idxs = idxs[order]
        dists = dists[order]
        src = stations_arr[i]
        for r, (j, d) in enumerate(zip(idxs, dists), start=1):
            rows.append((src, stations_arr[j], r, float(d)))

    return pd.DataFrame(rows, columns=["station", "neighbor", "rank", "distance_km"])
