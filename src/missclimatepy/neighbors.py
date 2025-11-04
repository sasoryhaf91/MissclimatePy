# src/missclimatepy/neighbors.py
"""
Neighbor utilities (Haversine KNN)
----------------------------------

Public API
----------
- neighbor_distances(stations, k_neighbors=20, radius_km=6371.0088, include_self=False)
- build_neighbor_map(df, id_col, lat_col, lon_col, k, include_self=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Union

StationId = Union[str, int]

__all__ = ["neighbor_distances", "build_neighbor_map"]


def _to_radians(x: np.ndarray) -> np.ndarray:
    return np.deg2rad(x.astype(float))


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray, radius_km: float = 6371.0088) -> np.ndarray:
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
    Compute K nearest neighbors per station using Haversine distance.

    Parameters
    ----------
    stations : DataFrame with ['station','latitude','longitude'].
    k_neighbors : int
        Number of neighbors to return per station (clipped to dataset size).
    include_self : bool
        If False, self-pairs are removed; if True, self can appear with distance 0.

    Returns
    -------
    DataFrame with columns ['station','neighbor','rank','distance_km'].
    """
    req = {"station", "latitude", "longitude"}
    miss = req - set(stations.columns)
    if miss:
        raise ValueError(f"neighbor_distances: missing columns {sorted(miss)}")

    df = (
        stations[["station", "latitude", "longitude"]]
        .drop_duplicates(subset=["station", "latitude", "longitude"])
        .reset_index(drop=True)
    )
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    D = _haversine_matrix(df["latitude"].to_numpy(), df["longitude"].to_numpy(), radius_km=radius_km)
    if not include_self:
        np.fill_diagonal(D, np.inf)

    k_eff = int(min(max(k_neighbors, 0), n if include_self else max(0, n - 1)))
    if k_eff == 0:
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    nbr_idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]

    rows: List[tuple] = []
    sarr = df["station"].astype(str).to_numpy()
    for i in range(n):
        idxs = nbr_idx[i]
        dists = D[i, idxs]
        order = np.argsort(dists)
        for r, j in enumerate(idxs[order], start=1):
            rows.append((sarr[i], sarr[j], r, float(D[i, j])))

    return pd.DataFrame(rows, columns=["station", "neighbor", "rank", "distance_km"])


def build_neighbor_map(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k: int,
    include_self: bool = False,
) -> Dict[StationId, List[StationId]]:
    """
    Convenience wrapper used by legacy tests.

    Builds a per-station neighbor list using CENTROIDS
    (median of lat/lon per station) and Haversine KNN.

    Returns
    -------
    dict[station_id -> list_of_neighbors]
    """
    centroids = (
        df.groupby(id_col)[[lat_col, lon_col]]
        .median()
        .reset_index()
        .rename(columns={id_col: "station", lat_col: "latitude", lon_col: "longitude"})
    )
    tbl = neighbor_distances(
        stations=centroids,
        k_neighbors=int(k),
        include_self=bool(include_self),
    )
    nmap: Dict[StationId, List[StationId]] = {}
    for st, sub in tbl.groupby("station"):
        nmap[st] = sub.sort_values("rank")["neighbor"].tolist()
    return nmap
