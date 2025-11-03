# src/missclimatepy/neighbors.py
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ----------------------------- small helpers ----------------------------- #
def _to_radians(x: np.ndarray) -> np.ndarray:
    """Degrees -> radians with safe float casting."""
    return np.deg2rad(np.asarray(x, dtype=float))


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray, radius_km: float = 6371.0088) -> np.ndarray:
    """
    Full pairwise Haversine distance matrix (km) for coordinates.

    Parameters
    ----------
    lat, lon : 1D arrays (n,) with coordinates in degrees.
    radius_km : Earth radius in km.

    Returns
    -------
    (n, n) ndarray of distances in km (diagonal is 0).
    """
    lat_r = _to_radians(lat)
    lon_r = _to_radians(lon)

    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return radius_km * c


def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    Return the first existing column (case-insensitive) among `candidates`.
    Raise ValueError if none is found.
    """
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        name = c.lower()
        if name in lower:
            return lower[name]
    raise ValueError(f"Expected one of {candidates} in DataFrame, got: {list(df.columns)}")


def _resolve_alt_col(df: pd.DataFrame) -> Optional[str]:
    """Return altitude/elevation column name if present, else None."""
    for c in ["altitude", "elevation", "alt", "elev"]:
        for col in df.columns:
            if col.lower() == c:
                return col
    return None


# ----------------------------- public API -------------------------------- #
def neighbor_distances(
    df_coords: pd.DataFrame,
    n_neighbors: Optional[int] = None,
    *,
    # Backwards-compatible aliases:
    k_neighbors: Optional[int] = None,
    k: Optional[int] = None,
    radius_km: float = 6371.0088,
) -> Dict[int, List[int]]:
    """
    Build a neighbor map (by Haversine distance) for each station.

    This function returns, for every station, an ordered list of neighbors
    **including the station itself in position 0**. This matches the usage in
    `quickstart.py`, where we typically request `n_neighbors = K + 1` and then
    drop index 0 to keep the K nearest other stations.

    Parameters
    ----------
    df_coords : DataFrame
        Must contain (case-insensitive) id and coordinate columns:
        - station id: one of ["station", "estacion", "id"]
        - latitude : one of ["latitude", "lat", "y"]
        - longitude: one of ["longitude", "lon", "x"]
        Extra columns are ignored.
    n_neighbors : int, optional
        Number of neighbors to return **including the station itself**.
        If None, resolved from `k_neighbors` or `k`; defaults to 20.
    k_neighbors, k : int, optional
        Backwards-compatible aliases for `n_neighbors`.
    radius_km : float
        Earth radius for Haversine (km).

    Returns
    -------
    Dict[int, List[int]]
        Mapping station_id -> [self, n1, n2, ...] ordered by distance (ascending).
    """
    # --- resolve K with aliases
    if n_neighbors is None:
        if k_neighbors is not None:
            n_neighbors = int(k_neighbors)
        elif k is not None:
            n_neighbors = int(k)
        else:
            n_neighbors = 20
    n_neighbors = int(max(1, n_neighbors))

    # --- resolve essential columns
    id_col = _resolve_col(df_coords, ["station", "estacion", "id"])
    lat_col = _resolve_col(df_coords, ["latitude", "lat", "y"])
    lon_col = _resolve_col(df_coords, ["longitude", "lon", "x"])

    # one row per station id
    co = df_coords[[id_col, lat_col, lon_col]].drop_duplicates(subset=[id_col]).reset_index(drop=True)
    if co.empty:
        return {}

    # compute pairwise distances
    D = _haversine_matrix(co[lat_col].to_numpy(), co[lon_col].to_numpy(), radius_km=radius_km)

    # cap to dataset size
    k_eff = int(min(n_neighbors, len(co)))

    # indices of k smallest distances per row
    # (include self; diagonal is 0, so self will always appear)
    idx_k = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]

    # build ordered mapping
    ids = co[id_col].astype(int).to_numpy()
    mapping: Dict[int, List[int]] = {}

    for i in range(len(co)):
        idxs = idx_k[i]
        dists = D[i, idxs]
        order = np.argsort(dists)               # ensure ascending by true distance
        idxs = idxs[order]
        neigh_ids = ids[idxs].tolist()

        # guarantee self is at position 0 (should already be, but enforce just in case)
        self_id = int(ids[i])
        if neigh_ids[0] != self_id:
            if self_id in neigh_ids:
                neigh_ids.remove(self_id)
            neigh_ids = [self_id] + neigh_ids

        # trim to requested length
        mapping[self_id] = neigh_ids[:k_eff]

    return mapping


def neighbor_distances_long(
    stations: pd.DataFrame,
    *,
    k_neighbors: int = 20,
    radius_km: float = 6371.0088,
    include_self: bool = False,
) -> pd.DataFrame:
    """
    DataFrame (long format) version similar to your original implementation.

    Returns
    -------
    DataFrame with columns:
        station, neighbor, rank, distance_km
    """
    id_col = _resolve_col(stations, ["station", "estacion", "id"])
    lat_col = _resolve_col(stations, ["latitude", "lat", "y"])
    lon_col = _resolve_col(stations, ["longitude", "lon", "x"])

    df = stations[[id_col, lat_col, lon_col]].drop_duplicates(subset=[id_col]).reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    D = _haversine_matrix(df[lat_col].to_numpy(), df[lon_col].to_numpy(), radius_km=radius_km)
    if not include_self:
        np.fill_diagonal(D, np.inf)

    k_eff = int(min(max(k_neighbors, 0), n if include_self else max(0, n - 1)))
    if k_eff == 0:
        return pd.DataFrame(columns=["station", "neighbor", "rank", "distance_km"])

    nbr_idx = np.argpartition(D, kth=k_eff - 1, axis=1)[:, :k_eff]
    rows: List[Tuple[int, int, int, float]] = []
    stations_arr = df[id_col].astype(int).to_numpy()
    for i in range(n):
        idxs = nbr_idx[i]
        dists = D[i, idxs]
        order = np.argsort(dists)
        idxs = idxs[order]
        dists = dists[order]
        src = int(stations_arr[i])
        for r, (j, d) in enumerate(zip(idxs, dists), start=1):
            rows.append((src, int(stations_arr[j]), r, float(d)))
    out = pd.DataFrame(rows, columns=["station", "neighbor", "rank", "distance_km"])
    return out


# --------------------- utilities with altitude/elevation ------------------ #
def _station_xyz(df: pd.DataFrame, station: int | str) -> Tuple[float, float, float]:
    """Return (lat, lon, alt) using 'altitude' or 'elevation' if available."""
    id_col = _resolve_col(df, ["station", "estacion", "id"])
    lat_col = _resolve_col(df, ["latitude", "lat", "y"])
    lon_col = _resolve_col(df, ["longitude", "lon", "x"])
    alt_col = _resolve_alt_col(df)

    cols = [lat_col, lon_col] + ([alt_col] if alt_col else [])
    row = df.loc[df[id_col] == station, cols].drop_duplicates().iloc[0]
    lat = float(row[lat_col])
    lon = float(row[lon_col])
    alt = float(row[alt_col]) if alt_col else float("nan")
    return lat, lon, alt


def get_station_neighbors(df: pd.DataFrame, station: int | str, k_neighbors: int = 20) -> pd.DataFrame:
    """
    Simple Euclidean neighbor list in (lat, lon, alt) space (alt optional).
    Returns a DataFrame with columns [neighbor, dist, corr], where `corr` is left NaN.
    """
    id_col = _resolve_col(df, ["station", "estacion", "id"])
    lat_col = _resolve_col(df, ["latitude", "lat", "y"])
    lon_col = _resolve_col(df, ["longitude", "lon", "x"])
    alt_col = _resolve_alt_col(df)

    lat, lon, alt = _station_xyz(df, station)

    stations = df[[id_col, lat_col, lon_col] + ([alt_col] if alt_col else [])].drop_duplicates().copy()
    stations = stations[stations[id_col] != station]

    dlat = stations[lat_col].to_numpy(dtype=float) - lat
    dlon = stations[lon_col].to_numpy(dtype=float) - lon
    if alt_col:
        delv = stations[alt_col].to_numpy(dtype=float) - (alt if np.isfinite(alt) else 0.0)
        dist = np.sqrt(dlat**2 + dlon**2 + (delv / 1000.0) ** 2)
    else:
        dist = np.sqrt(dlat**2 + dlon**2)

    out = stations[[id_col]].copy()
    out = out.rename(columns={id_col: "neighbor"})
    out["dist"] = dist
    out = out.sort_values("dist", ascending=True).head(k_neighbors).reset_index(drop=True)
    out["corr"] = np.nan
    return out


def neighbor_overlap_ratio(df: pd.DataFrame, target: str, station_a: int | str, station_b: int | str) -> float:
    """
    Ratio of overlapping valid days between two stations for `target` (0..1).
    """
    id_col = _resolve_col(df, ["station", "estacion", "id"])
    date_col = _resolve_col(df, ["date", "fecha"])
    sub = df[[id_col, date_col, target]].copy()
    a = sub[(sub[id_col] == station_a) & sub[target].notna()][[date_col]]
    b = sub[(sub[id_col] == station_b) & sub[target].notna()][[date_col]]
    if a.empty or b.empty:
        return 0.0
    merged = a.merge(b, on=date_col, how="inner")
    num = len(merged)
    den = min(len(a), len(b))
    return float(num / den) if den > 0 else 0.0
