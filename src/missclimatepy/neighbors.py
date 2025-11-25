# src/missclimatepy/neighbors.py
# SPDX-License-Identifier: MIT
"""
Spatial neighbor utilities for MissClimatePy.

This module provides small, focused helpers to work with station
coordinates on the sphere (latitude/longitude in degrees) and to build
neighbor structures based on great-circle distances:

- :func:`haversine_distance`:
    Vectorized great-circle distance (km) between two sets of points.
- :func:`compute_station_centroids`:
    Reduce a long table to one row per station, with mean latitude/longitude.
- :func:`neighbor_distances`:
    Compute K-nearest neighbor pairs (from_id, to_id, distance_km)
    using a haversine BallTree.
- :func:`build_neighbor_map`:
    Convenience wrapper returning a dict {station_id -> [neighbor_ids]}.

All functions assume coordinates in **decimal degrees** and distances
are returned in **kilometers**.
"""

from __future__ import annotations

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Basic haversine distance
# ---------------------------------------------------------------------------


def haversine_distance(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Compute great-circle distance between two sets of points (km).

    Parameters
    ----------
    lat1, lon1 : array-like
        Latitudes/longitudes of the first set of points in degrees.
    lat2, lon2 : array-like
        Latitudes/longitudes of the second set of points in degrees.

    Returns
    -------
    np.ndarray
        Distances in kilometers, with standard NumPy broadcasting.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))

    return EARTH_RADIUS_KM * c


# ---------------------------------------------------------------------------
# Station centroids
# ---------------------------------------------------------------------------


def compute_station_centroids(
    data: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    """
    Compute a single lat/lon centroid per station.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format table with at least the station id and coordinate columns.
    id_col : str
        Name of the station id column.
    lat_col : str
        Name of the latitude column (degrees).
    lon_col : str
        Name of the longitude column (degrees).

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns [id_col, lat_col, lon_col] and one row per
        unique station id. Lat/lon are simple arithmetic means.
    """
    if data.empty:
        return pd.DataFrame(columns=[id_col, lat_col, lon_col])

    centroids = (
        data[[id_col, lat_col, lon_col]]
        .groupby(id_col, as_index=False)
        .mean(numeric_only=True)
    )
    return centroids


# ---------------------------------------------------------------------------
# Neighbor distances (BallTree over haversine)
# ---------------------------------------------------------------------------


def neighbor_distances(
    data: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k: int,
    include_self: bool = True,
) -> pd.DataFrame:
    """
    Compute K-nearest neighbor distances between station centroids.

    This function uses a :class:`sklearn.neighbors.BallTree` in haversine
    space (radians) to obtain, for each station, its K nearest neighbors.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format table with at least [id_col, lat_col, lon_col].
    id_col : str
        Station id column.
    lat_col : str
        Latitude column (degrees).
    lon_col : str
        Longitude column (degrees).
    k : int
        Desired number of neighbors **per station**. When ``include_self``
        is False, neighbors are strictly different stations. When ``k``
        is larger than the number of available stations, the function
        caps the effective k to the maximum possible value.
    include_self : bool, default True
        If True, each station can appear as its own nearest neighbor in
        the output. If False, self-pairs are removed and up to ``k``
        distinct neighbors are kept.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``["from_id", "to_id", "distance_km"]`` where:

        - ``from_id`` is the reference station id.
        - ``to_id``   is the neighbor station id.
        - ``distance_km`` is the great-circle distance in kilometers.
    """
    centroids = compute_station_centroids(
        data,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    if centroids.empty:
        return pd.DataFrame(columns=["from_id", "to_id", "distance_km"])

    ids = centroids[id_col].to_numpy()
    coords_deg = centroids[[lat_col, lon_col]].to_numpy(dtype=float)
    coords_rad = np.radians(coords_deg)

    n_stations = coords_rad.shape[0]
    if n_stations == 0:
        return pd.DataFrame(columns=["from_id", "to_id", "distance_km"])

    # BallTree requires k <= n_stations.
    # If exclude self (include_self=False), we query k+1 and later drop self.
    k = int(max(1, k))
    if include_self:
        query_k = min(k, n_stations)
    else:
        query_k = min(k + 1, n_stations)

    query_k = max(1, query_k)

    tree = BallTree(coords_rad, metric="haversine")
    dist_rad, ind = tree.query(coords_rad, k=query_k)

    # distances in km
    dist_km = dist_rad * EARTH_RADIUS_KM

    rows: List[dict] = []
    for i in range(n_stations):
        from_id = ids[i]
        for j_pos in range(query_k):
            j = ind[i, j_pos]
            to_id = ids[j]
            d = float(dist_km[i, j_pos])
            rows.append(
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "distance_km": d,
                }
            )

    df_pairs = pd.DataFrame(rows)

    if not include_self:
        # Remove self-pairs
        df_pairs = df_pairs[df_pairs["from_id"] != df_pairs["to_id"]]

        # For each from_id, keep at most k closest neighbors
        df_pairs = (
            df_pairs.sort_values(["from_id", "distance_km"], kind="mergesort")
            .groupby("from_id", as_index=False)
            .head(k)
        )

    return df_pairs.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Neighbor map
# ---------------------------------------------------------------------------


def build_neighbor_map(
    data: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k: int,
    include_self: bool = False,
) -> Dict[Union[int, str], List[Union[int, str]]]:
    """
    Build a neighbor map: {station_id -> list_of_neighbor_ids}.

    This is a convenience wrapper around :func:`neighbor_distances`.
    It is mainly used by the imputation and evaluation engines to
    restrict the training pool for each station.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format dataset with at least [id_col, lat_col, lon_col].
    id_col : str
        Station id column.
    lat_col : str
        Latitude column (degrees).
    lon_col : str
        Longitude column (degrees).
    k : int
        Desired number of neighbors. If ``include_self`` is False, neighbors
        are strictly different stations. When ``k`` is larger than the number
        of available stations, the function caps the effective k.
    include_self : bool, default False
        If True, each station may appear in its own neighbor list; this is
        rarely desired for training, so the default is False.

    Returns
    -------
    dict
        Dictionary mapping each station id to an ordered list of neighbor
        ids (closest first). Stations with no neighbors (e.g. single-row
        edge cases) appear with an empty list.
    """
    if data.empty:
        return {}

    pairs = neighbor_distances(
        data=data,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        k=k,
        include_self=include_self,
    )

    neighbor_map: Dict[Union[int, str], List[Union[int, str]]] = {}

    if pairs.empty:
        # Ensure all stations appear with an empty list
        for sid in data[id_col].dropna().unique().tolist():
            neighbor_map[sid] = []
        return neighbor_map

    # Initialize with empty lists for all stations present in the data
    for sid in data[id_col].dropna().unique().tolist():
        neighbor_map[sid] = []

    # Populate neighbor lists, keeping order (already sorted in neighbor_distances)
    for from_id, group in pairs.groupby("from_id", sort=False):
        neighbor_map[from_id] = group["to_id"].tolist()

    return neighbor_map


__all__ = [
    "EARTH_RADIUS_KM",
    "haversine_distance",
    "compute_station_centroids",
    "neighbor_distances",
    "build_neighbor_map",
]

