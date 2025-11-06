# tests/test_neighbors.py
import pandas as pd
from missclimatepy.neighbors import neighbor_distances, build_neighbor_map

def test_neighbor_distances_basic():
    df = pd.DataFrame({
        "station": ["A", "B", "C"],
        "latitude": [19.0, 19.1, 19.2],
        "longitude": [-99.0, -99.1, -99.2],
        "altitude": [2300, 2310, 2290]
    })
    out = neighbor_distances(df,
                             id_col="station",
                             lat_col="latitude",
                             lon_col="longitude",
                             altitude_col="altitude",
                             k_neighbors=2)
    assert not out.empty
    assert {"station", "neighbor", "distance_km"}.issubset(out.columns)
    assert (out["distance_km"] >= 0).all()

def test_build_neighbor_map_consistency():
    df = pd.DataFrame({
        "station": ["A", "B", "C"],
        "latitude": [19.0, 19.1, 19.2],
        "longitude": [-99.0, -99.1, -99.2]
    })
    mapping = build_neighbor_map(df,
                                 id_col="station",
                                 lat_col="latitude",
                                 lon_col="longitude",
                                 k_neighbors=2)
    assert isinstance(mapping, dict)
    assert all(isinstance(v, list) for v in mapping.values())
