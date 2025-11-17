# tests/test_impute.py
# SPDX-License-Identifier: MIT
"""
Unit tests for missclimatepy.impute.impute_dataset

These tests focus on:
- basic schema and behaviour,
- min_station_rows filtering,
- include_target_pct semantics,
- neighbour-based imputation with include_target_pct=0.
"""

import numpy as np
import pandas as pd

from missclimatepy.impute import impute_dataset


def _make_simple_dataframe():
    """
    Create a small synthetic long-format dataset with two stations:
    - S1: partial missingness
    - S2: almost full observations
    """
    dates = pd.date_range("2000-01-01", periods=10, freq="D")

    df = pd.DataFrame({
        "station": ["S1"] * 10 + ["S2"] * 10,
        "date": list(dates) * 2,
        "latitude": [19.0] * 10 + [20.0] * 10,
        "longitude": [-99.0] * 10 + [-98.0] * 10,
        "altitude": [2300.0] * 10 + [2400.0] * 10,
        "tmin": np.concatenate([
            np.array([10.0, 11.0, np.nan, 13.0, np.nan, 15.0, 16.0, np.nan, 18.0, 19.0]),
            np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),
        ])
    })
    return df


def test_impute_basic_schema_and_content():
    """Basic smoke test: schema is correct and imputation runs."""
    df = _make_simple_dataframe()

    out = impute_dataset(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmin",
        start="2000-01-01",
        end="2000-01-10",
        k_neighbors=1,
        min_station_rows=None,      # no MDR
        include_target_pct=None,    # full visibility
        show_progress=False,
    )

    # Columns and order
    assert list(out.columns) == [
        "station", "date", "latitude", "longitude", "altitude", "tmin", "source"
    ]

    # Both stations must be present
    assert set(out["station"].unique()) == {"S1", "S2"}

    # S2 has no missing values originally → all "observed"
    s2 = out[out["station"] == "S2"].sort_values("date")
    assert s2["tmin"].isna().sum() == 0
    assert (s2["source"] == "observed").all()

    # S1: where original df had values, they must remain unchanged
    s1_in = df[df["station"] == "S1"].sort_values("date").reset_index(drop=True)
    s1_out = out[out["station"] == "S1"].sort_values("date").reset_index(drop=True)
    # For non-NaN original rows, imputed series must match
    mask_obs = ~s1_in["tmin"].isna()
    assert np.allclose(
        s1_out.loc[mask_obs, "tmin"].to_numpy(),
        s1_in.loc[mask_obs, "tmin"].to_numpy(),
        equal_nan=False,
    )


def test_min_station_rows_filters_stations():
    """
    min_station_rows > N: only stations with at least that many observed
    rows are imputed and returned.
    """
    df = _make_simple_dataframe()

    # Impose a high MDR so that S1 is excluded and only S2 remains
    out = impute_dataset(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmin",
        start="2000-01-01",
        end="2000-01-10",
        k_neighbors=1,
        min_station_rows=9,      # S1 has fewer non-NaN rows than S2
        include_target_pct=None,
        show_progress=False,
    )

    stations = set(out["station"].unique())
    assert stations == {"S2"}
    # S2 should remain fully observed
    assert out["tmin"].isna().sum() == 0
    assert (out["source"] == "observed").all()


def test_include_target_pct_partial_visibility_preserves_observed():
    """
    include_target_pct < 100: the model sees only part of the target
    station history, but all observed values must be preserved in the output.
    """
    df = _make_simple_dataframe()

    out = impute_dataset(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmin",
        start="2000-01-01",
        end="2000-01-10",
        k_neighbors=1,
        min_station_rows=None,
        include_target_pct=50.0,   # only half of local history in training
        show_progress=False,
    )

    s1_in = df[df["station"] == "S1"].sort_values("date").reset_index(drop=True)
    s1_out = out[out["station"] == "S1"].sort_values("date").reset_index(drop=True)

    # Observed entries must remain exactly the same, even if not used in training
    mask_obs = ~s1_in["tmin"].isna()
    assert np.allclose(
        s1_out.loc[mask_obs, "tmin"].to_numpy(),
        s1_in.loc[mask_obs, "tmin"].to_numpy(),
        equal_nan=False,
    )


def test_include_target_pct_zero_with_neighbors_allows_loso_imputation():
    """
    include_target_pct=0 with neighbours: the station is imputed using
    only neighbour information (extreme LOSO scenario).
    """
    # Simple scenario: S1 has missing data; S2 has constant value 5.
    dates = pd.date_range("2000-01-01", periods=5, freq="D")
    df = pd.DataFrame({
        "station": ["S1"] * 5 + ["S2"] * 5,
        "date": list(dates) * 2,
        "latitude": [19.0] * 5 + [20.0] * 5,
        "longitude": [-99.0] * 5 + [-98.0] * 5,
        "altitude": [2300.0] * 5 + [2400.0] * 5,
        # S1: all NaN; S2: all 5.0
        "tmin": np.concatenate([
            np.array([np.nan] * 5),
            np.array([5.0] * 5),
        ])
    })

    out = impute_dataset(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmin",
        start="2000-01-01",
        end="2000-01-05",
        k_neighbors=1,           # S1 will see S2 as neighbour
        min_station_rows=None,
        include_target_pct=0.0,  # S1 contributes 0 local rows to training
        show_progress=False,
    )

    s1_out = out[out["station"] == "S1"].sort_values("date").reset_index(drop=True)

    # All rows should be imputed (no observed values in S1)
    assert (s1_out["source"] == "imputed").all()
    # Predictions should be close to 5 (since S2 is constant 5.0)
    assert np.allclose(s1_out["tmin"].to_numpy(), 5.0, atol=1e-6)

