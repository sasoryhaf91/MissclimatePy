from __future__ import annotations
import numpy as np
import pandas as pd

EARTH_R_KM = 6371.0

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(EARTH_R_KM) * c

def select_neighbors(meta: pd.DataFrame, st_row: pd.Series,
                     k_neighbors: int | None = 15,
                     radius_km: float | None = None) -> pd.DataFrame:
    """Return neighbor stations (excluding the target) ordered by distance."""
    d = np.array([haversine_km(st_row.latitude, st_row.longitude, r.latitude, r.longitude)
                  for r in meta.itertuples(index=False)])
    cand = meta.assign(dist_km=d)
    cand = cand[cand.station != st_row.station]
    cand = cand.sort_values("dist_km")
    if radius_km is not None:
        cand = cand[cand.dist_km <= radius_km]
    if k_neighbors is not None and radius_km is None:
        cand = cand.head(k_neighbors)
    return cand.reset_index(drop=True)
