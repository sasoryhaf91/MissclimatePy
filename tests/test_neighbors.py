# tests/test_neighbors.py
import numpy as np
import pandas as pd

from missclimatepy.neighbors import build_neighbor_map


def _make_simple_df() -> pd.DataFrame:
    """Small helper dataset with a few stations and coordinates."""
    return pd.DataFrame(
        {
            "station": ["A", "B", "C", "D"],
            "lat": [19.0, 19.1, 19.2, 20.0],
            "lon": [-99.0, -99.1, -99.2, -100.0],
            # extra columns should be ignored by build_neighbor_map
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_build_neighbor_map_returns_dict():
    df = _make_simple_df()

    nmap = build_neighbor_map(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=2,
        include_self=False,
    )

    # must be a dict with all station ids as keys
    assert isinstance(nmap, dict)
    assert set(nmap.keys()) == set(df["station"].unique())

    # each value must be a non-empty list
    for sid, neighs in nmap.items():
        assert isinstance(neighs, list)
        assert len(neighs) > 0
        # neighbors must be valid station ids
        for n in neighs:
            assert n in df["station"].values
        # when include_self=False, target id must not appear
        assert sid not in neighs


def test_build_neighbor_map_include_self_true():
    df = _make_simple_df()

    nmap = build_neighbor_map(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=2,
        include_self=True,
    )

    # include_self=True → self id must appear in the neighbor list
    for sid, neighs in nmap.items():
        assert sid in neighs
        # still must not exceed the number of available stations
        assert len(neighs) <= len(df["station"].unique())


def test_build_neighbor_map_handles_large_k_gracefully():
    df = _make_simple_df()
    n_stations = df["station"].nunique()

    # k much larger than number of stations should not error
    k = 100
    nmap = build_neighbor_map(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=k,
        include_self=False,
    )

    # with include_self=False, each station can have at most n_stations-1 neighbors
    for sid, neighs in nmap.items():
        assert len(neighs) <= n_stations - 1
        # neighbors must be unique
        assert len(neighs) == len(set(neighs))
        assert sid not in neighs


def test_build_neighbor_map_works_with_integer_ids():
    # same coordinates, but integer ids
    df = pd.DataFrame(
        {
            "station": [101, 102, 103],
            "lat": [19.0, 19.1, 19.2],
            "lon": [-99.0, -99.1, -99.2],
        }
    )

    nmap = build_neighbor_map(
        df,
        id_col="station",
        lat_col="lat",
        lon_col="lon",
        k=1,
        include_self=False,
    )

    assert set(nmap.keys()) == {101, 102, 103}
    # neighbors must be integers as well
    for neighs in nmap.values():
        for n in neighs:
            assert isinstance(n, (int, np.integer))
