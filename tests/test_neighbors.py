# tests/test_neighbors.py

import numpy as np
import pandas as pd

from missclimatepy.neighbors import (
    haversine_distance,
    compute_station_centroids,
    neighbor_distances,
    build_neighbor_map,
)


def test_haversine_distance_zero_and_symmetry():
    lat = np.array([0.0])
    lon = np.array([0.0])

    d0 = haversine_distance(lat, lon, lat, lon)
    assert d0.shape == (1,)
    assert d0[0] == 0.0

    # Symmetry check: distance(a, b) == distance(b, a)
    lat_a = np.array([0.0])
    lon_a = np.array([0.0])
    lat_b = np.array([0.0])
    lon_b = np.array([1.0])

    dab = haversine_distance(lat_a, lon_a, lat_b, lon_b)[0]
    dba = haversine_distance(lat_b, lon_b, lat_a, lon_a)[0]
    assert np.isclose(dab, dba)

    # Rough magnitude: 1 degree lon at equator ~ 111 km
    assert np.isclose(dab, 111.0, atol=2.0)


def test_compute_station_centroids_empty():
    df = pd.DataFrame(columns=["station", "lat", "lon"])
    centroids = compute_station_centroids(df, id_col="station", lat_col="lat", lon_col="lon")
    assert isinstance(centroids, pd.DataFrame)
    assert centroids.empty
    assert list(centroids.columns) == ["station", "lat", "lon"]


def test_compute_station_centroids_mean():
    df = pd.DataFrame(
        {
            "station": ["A", "A", "B", "B", "B"],
            "lat": [10.0, 12.0, 0.0, 1.0, 2.0],
            "lon": [20.0, 22.0, 5.0, 5.0, 5.0],
        }
    )
    centroids = compute_station_centroids(df, id_col="station", lat_col="lat", lon_col="lon")

    assert set(centroids["station"]) == {"A", "B"}

    row_a = centroids.set_index("station").loc["A"]
    assert np.isclose(row_a["lat"], (10.0 + 12.0) / 2.0)
    assert np.isclose(row_a["lon"], (20.0 + 22.0) / 2.0)

    row_b = centroids.set_index("station").loc["B"]
    assert np.isclose(row_b["lat"], (0.0 + 1.0 + 2.0) / 3.0)
    assert np.isclose(row_b["lon"], 5.0)


def _make_small_df():
    """Helper to create a tiny 3-station dataset for neighbor tests."""
    return pd.DataFrame(
        {
            "station": ["S1", "S1", "S2", "S2", "S3"],
            "lat": [0.0, 0.1, 0.0, 0.05, 1.0],
            "lon": [0.0, 0.1, 1.0, 1.0, 0.0],
        }
    )


def test_neighbor_distances_include_self():
    df = _make_small_df()
    pairs = neighbor_distances(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=2,
        include_self=True,
    )

    # We should get at most k neighbors per station
    assert not pairs.empty
    assert set(pairs.columns) == {"from_id", "to_id", "distance_km"}

    grouped = pairs.groupby("from_id")
    for sid, sub in grouped:
        assert len(sub) <= 2
        # At least one self-pair with distance ~0 should exist (depending on k and n)
        has_self = (sub["from_id"] == sub["to_id"]).any()
        if has_self:
            d_self = sub.loc[sub["from_id"] == sub["to_id"], "distance_km"].iloc[0]
            assert np.isclose(d_self, 0.0, atol=1e-6)


def test_neighbor_distances_exclude_self():
    df = _make_small_df()
    pairs = neighbor_distances(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=2,
        include_self=False,
    )

    assert not pairs.empty
    assert set(pairs.columns) == {"from_id", "to_id", "distance_km"}

    # No self-pairs
    assert (pairs["from_id"] != pairs["to_id"]).all()

    grouped = pairs.groupby("from_id")
    for sid, sub in grouped:
        # At most k neighbors per station
        assert len(sub) <= 2
        # Distances must be positive
        assert (sub["distance_km"] > 0.0).all()


def test_build_neighbor_map_basic():
    df = _make_small_df()
    k = 2
    nmap = build_neighbor_map(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=k,
        include_self=False,
    )

    # Keys must match station ids
    stations = set(df["station"].unique())
    assert set(nmap.keys()) == stations

    # Each neighbor list has length <= k and contains no self
    for sid, neighs in nmap.items():
        assert isinstance(neighs, list)
        assert len(neighs) <= k
        assert sid not in neighs


def test_build_neighbor_map_empty():
    df = pd.DataFrame(columns=["station", "lat", "lon"])
    nmap = build_neighbor_map(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=3,
        include_self=False,
    )
    assert nmap == {}
