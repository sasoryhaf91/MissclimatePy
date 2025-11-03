# src/missclimatepy/neighbors.py
from __future__ import annotations

import numpy as np
import pandas as pd
#from typing import Optional


def _to_radians(x: np.ndarray) -> np.ndarray:
    """Helper: degrees -> radians."""
    return np.deg2rad(x.astype(float))


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray, radius_km: float = 6371.0088) -> np.ndarray:
    """
    Compute the full pairwise Haversine distance matrix (km) for coordinates.

    Parameters
    ----------
    lat, lon : np.ndarray
        1D arrays (n,) with latitudes and longitudes in degrees.
    radius_km : float, default 6371.0088
        Earth radius (km).

    Returns
    -------
    np.ndarray
        (n, n) matrix of distances in km (diagonal is 0).
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
    Compute K nearest neighbors per station using Haversine distance.

    Parameters
    ----------
    stations : DataFrame
        DataFrame with at least columns: 'station', 'latitude', 'longitude'.
        'elevation' is accepted but not required for the distance itself.
        Duplicates by ('station','latitude','longitude') will be dropped.
    k_neighbors : int, default 20
        Number of neighbors to return **per station**.
    radius_km : float, default 6371.0088
        Earth radius for Haversine distance.
    include_self : bool, default False
        If True, a station can appear as its own neighbor with distance 0.
        If False, self-pairs are excluded.

    Returns
    -------
    DataFrame
        Columns:
        - station : str (source station)
        - neighbor: str (neighbor station)
        - rank    : int (1..K)
        - distance_km : float

    Notes
    -----
    - Complexity is O(n^2). For >~10k stations consider chunking or
      approximate nearest neighbors.
    """
    required = {"station", "latitude", "longitude"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"neighbor_distances: missing columns {sorted(missing)}")

    # De-duplicate coordinates per station
    df = (
        stations[[c for c in ["station", "latitude", "longitude", "elevation"] if c in stations.columns]]
        .drop_duplicates(subset=["station", "latitude", "longitude"])
        .reset_index(drop=True)
        .copy()
    )
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    # Pairwise distances
    D = _haversine_matrix(df["latitude"].to_numpy(), df["longitude"].to_numpy(), radius_km=radius_km)

    # Exclude self if requested
    if not include_self:
        np.fill_diagonal(D, np.inf)

    # How many neighbors we can actually provide (<= n or n-1)
    k_eff = int(min(max(k_neighbors, 0), n if include_self else max(0, n - 1)))
    if k_eff == 0:
        # Nothing to return (e.g., n=1 and include_self=False)
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    # Get indices of K smallest distances for each row
    # argsort along axis=1 and take first k
    nbr_idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]

    # Sort those K by true distance to have rank-consistent ordering
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

def _station_xy(df: pd.DataFrame, station: str) -> tuple[float, float, float]:
    row = (
        df.loc[df["station"] == station, ["latitude", "longitude", "elevation"]]
        .drop_duplicates()
        .iloc[0]
    )
    return float(row["latitude"]), float(row["longitude"]), float(row["elevation"])


def get_station_neighbors(
    df: pd.DataFrame, station: str, k_neighbors: int = 20
) -> pd.DataFrame:
    """
    Return the `k_neighbors` closest stations to `station` using Euclidean
    distance in (lat, lon, elev) space, together with simple correlation
    to the target if present (computed on overlapping valid rows).
    """
    lat, lon, elev = _station_xy(df, station)
    stations = (
        df[["station", "latitude", "longitude", "elevation"]]
        .drop_duplicates()
        .copy()
    )
    stations = stations[stations["station"] != station]
    dlat = stations["latitude"].to_numpy() - lat
    dlon = stations["longitude"].to_numpy() - lon
    delv = stations["elevation"].to_numpy() - elev
    dist = np.sqrt(dlat**2 + dlon**2 + (delv / 1000.0) ** 2)  # mild scaling elev

    out = stations[["station"]].copy()
    out["dist"] = dist

    # naive corr using daily means per date when target exists
    # note: we don't import target here; corr computed later on demand
    out = out.sort_values("dist", ascending=True).head(k_neighbors)
    out = out.rename(columns={"station": "neighbor"}).reset_index(drop=True)
    out["corr"] = np.nan  # filled by caller optionally
    return out


def neighbor_overlap_ratio(
    df: pd.DataFrame, target: str, station_a: str, station_b: str
) -> float:
    """
    Compute the ratio of overlapping valid days between two stations
    for `target`. Returns 0..1.
    """
    sub = df[["station", "date", target]].copy()
    a = sub[(sub["station"] == station_a) & sub[target].notna()][["date"]]
    b = sub[(sub["station"] == station_b) & sub[target].notna()][["date"]]
    if a.empty or b.empty:
        return 0.0
    merged = a.merge(b, on="date", how="inner")
    num = len(merged)
    den = min(len(a), len(b))
    return float(num / den) if den > 0 else 0.0