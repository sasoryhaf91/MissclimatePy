# SPDX-License-Identifier: MIT
"""
Neighbor utilities for missclimatepy.

- neighbor_distances(): exact K nearest neighbors per station using
  Haversine pairwise distances (returns a tidy DataFrame).
- build_neighbor_map(): fast neighbor dictionary {station: [neighbor,...]}
  using sklearn NearestNeighbors on median station coordinates.

Notes
-----
Use build_neighbor_map for modeling (much lower memory and very fast).
Use neighbor_distances only when you really need explicit distances.
"""

from __future__ import annotations
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ---------- helpers ----------

def _to_radians(x: np.ndarray) -> np.ndarray:
    """Degrees -> radians as float64."""
    return np.deg2rad(x.astype("float64", copy=False))


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray, radius_km: float = 6371.0088) -> np.ndarray:
    """
    Full pairwise Haversine distance matrix (km).

    Parameters
    ----------
    lat, lon : (n,) arrays in degrees
    radius_km : float
        Earth radius in km.

    Returns
    -------
    (n, n) ndarray of distances (diagonal=0).
    """
    lat_r = _to_radians(lat)
    lon_r = _to_radians(lon)

    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return radius_km * c


# ---------- public API ----------

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
    stations : DataFrame with at least ['station','latitude','longitude'].
               'elevation'/'altitude' are ignored for distance.
    k_neighbors : int
        Number of neighbors to return PER station.
    include_self : bool
        If True include self-pairs (distance=0); else exclude.

    Returns
    -------
    DataFrame with columns:
      station (source), neighbor, rank (1..K), distance_km (float)
    """
    required = {"station", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"neighbor_distances: missing columns {sorted(missing)}")

    df = (
        stations[[c for c in ["station", "latitude", "longitude"] if c in stations.columns]]
        .drop_duplicates(subset=["station", "latitude", "longitude"])
        .reset_index(drop=True)
        .copy()
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

    rows = []
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

    out = pd.DataFrame(rows, columns=["station", "neighbor", "rank", "distance_km"])
    return out


def build_neighbor_map(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    k_neighbors: int = 20,
) -> Dict[int, List[int]]:
    """
    Fast neighbor dictionary using median coordinates per station and
    sklearn NearestNeighbors on radians.

    Returns
    -------
    dict: {station_id (int): [neighbor_id1, neighbor_id2, ...]}
    """
    cent = (
        data.groupby(id_col)[[lat_col, lon_col]]
        .median()
        .rename(columns={lat_col: "lat", lon_col: "lon"})
        .reset_index()
    )
    if cent.empty:
        return {}
    X = _to_radians(cent[["lat", "lon"]].values)
    n = min(int(k_neighbors) + 1, len(cent))
    nn = NearestNeighbors(n_neighbors=n, algorithm="auto", metric="euclidean")
    nn.fit(X)
    _, idxs = nn.kneighbors(X, return_distance=True)
    ids = cent[id_col].astype(int).to_numpy()
    out: Dict[int, List[int]] = {}
    for i, sid in enumerate(ids):
        neigh = [int(ids[j]) for j in idxs[i] if int(ids[j]) != int(sid)]
        out[int(sid)] = neigh[:k_neighbors]
    return out
