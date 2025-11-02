from __future__ import annotations
import numpy as np
import pandas as pd

def haversine(lat1, lon1, lat2, lon2, R=6371.0):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def neighbor_distances(meta_df: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    meta = meta_df.drop_duplicates("station").copy()
    lat = meta["latitude"].values
    lon = meta["longitude"].values
    rows = []
    for i, st in enumerate(meta["station"].tolist()):
        d = haversine(lat[i], lon[i], lat, lon)
        order = np.argsort(d)
        nn = order[1:k+1]
        dk = d[nn]
        rows.append({
            "station": st, "k": k,
            "dist_mean_km": float(np.mean(dk)),
            "dist_max_km": float(np.max(dk)),
            "dist_min_km": float(np.min(dk)),
        })
    return pd.DataFrame(rows)
