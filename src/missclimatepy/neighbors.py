from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict

__all__ = ["neighbor_distances"]

def _to_radians(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x.astype(float))

def _haversine_matrix(lat: np.ndarray, lon: np.ndarray, radius_km: float = 6371.0088) -> np.ndarray:
    """
    Full pairwise Haversine matrix (km). O(n^2) – chunk externally if needed.
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
    station_col: str = "station",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    k_neighbors: int = 20,
    include_self: bool = False,
) -> pd.DataFrame:
    """
    Compute K nearest neighbors per station using Haversine distance.

    Returns a tidy table:
        [station, neighbor, rank, distance_km]

    Notes
    -----
    - Duplicates by (station, lat, lon) are removed before computing distances.
    - Complexity is O(n^2); suitable for ~a few thousands stations in one shot.
    """
    required = {station_col, lat_col, lon_col}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"neighbor_distances: missing columns {sorted(missing)}")

    df = (
        stations[[c for c in [station_col, lat_col, lon_col] if c in stations.columns]]
        .drop_duplicates(subset=[station_col, lat_col, lon_col])
        .reset_index(drop=True)
        .copy()
    )
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=[station_col, "neighbor", "rank", "distance_km"])

    D = _haversine_matrix(df[lat_col].to_numpy(), df[lon_col].to_numpy())
    if not include_self:
        np.fill_diagonal(D, np.inf)

    k_eff = int(min(max(k_neighbors, 0), n if include_self else max(0, n - 1)))
    if k_eff == 0:
        return pd.DataFrame(columns=[station_col, "neighbor", "rank", "distance_km"])

    nbr_idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows = []
    ids = df[station_col].astype(str).to_numpy()
    for i in range(n):
        idxs = nbr_idx[i]
        dists = D[i, idxs]
        order = np.argsort(dists)
        for r, (j, d) in enumerate(zip(idxs[order], dists[order]), start=1):
            rows.append((ids[i], ids[j], r, float(d)))

    out = pd.DataFrame(rows, columns=[station_col, "neighbor", "rank", "distance_km"])
    return out
