# SPDX-License-Identifier: MIT
"""
Tests for missclimatepy.impute.impute_dataset
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from missclimatepy.impute import impute_dataset


def _make_basic_df() -> pd.DataFrame:
    """
    Small helper dataset with two stations, some missing values
    and complete XYZT features.
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    df = pd.DataFrame(
        {
            "station": ["S1"] * 4 + ["S2"] * 4,
            "date": list(dates) * 2,
            "lat": [10.0] * 4 + [11.0] * 4,
            "lon": [-100.0] * 4 + [-101.0] * 4,
            "alt": [2000.0] * 8,
            # S1 has one missing, S2 is fully observed
            "value": [1.0, 1.0, np.nan, 1.0, 2.0, 2.0, 2.0, 2.0],
        }
    )
    return df


def test_impute_dataset_fills_missing_and_marks_source():
    """
    Basic smoke test: missing values are filled where possible and
    `source` correctly distinguishes observed vs imputed.
    """
    df = _make_basic_df()
    n_missing = df["value"].isna().sum()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        # keep defaults: model_kind="rf"
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
    )

    # Shape is preserved (aside del posible recorte por fechas que aquí no usamos)
    assert out.shape[0] == df.shape[0]
    assert "source" in out.columns

    # Original observed values must remain unchanged
    mask_obs = df["value"].notna()
    assert np.allclose(
        out.loc[mask_obs, "value"].to_numpy(),
        df.loc[mask_obs, "value"].to_numpy(),
    )

    # All previously missing values with valid features should now be filled
    mask_imputed = df["value"].isna()
    # Puede que alguna fila no se pueda imputar si no hay datos de entrenamiento,
    # pero en este dataset simple sí debería rellenarse.
    assert out.loc[mask_imputed, "value"].notna().sum() == n_missing

    # Source flags: observed rows -> "observed", imputed rows -> "imputed"
    observed_sources = set(out.loc[mask_obs, "source"])
    imputed_sources = set(out.loc[mask_imputed, "source"])

    assert observed_sources == {"observed"}
    assert imputed_sources == {"imputed"}


def test_impute_dataset_respects_min_station_rows():
    """
    Stations below `min_station_rows` should be passed through without
    attempting to impute them: their missing values remain NaN and
    `source` stays as 'missing'.
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    df = pd.DataFrame(
        {
            "station": ["S1"] * 4 + ["S2"] * 4,
            "date": list(dates) * 2,
            "lat": [10.0] * 4 + [11.0] * 4,
            "lon": [-100.0] * 4 + [-101.0] * 4,
            "alt": [2000.0] * 8,
            # S1: only 1 observed value, 3 missing
            "value": [1.0, np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0],
        }
    )

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        min_station_rows=2,  # S1 does not meet this threshold
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
    )

    # For S1, missing values should remain NaN and source='missing'
    s1 = out[out["station"] == "S1"].copy()
    assert s1["value"].isna().sum() == 3
    assert set(s1.loc[s1["value"].isna(), "source"]) == {"missing"}

    # For S2, it should behave normally (no missing values)
    s2 = out[out["station"] == "S2"].copy()
    assert s2["value"].isna().sum() == 0
    assert set(s2["source"]) == {"observed"}


def test_impute_dataset_neighbor_map_controls_training():
    """
    A custom neighbor_map can prevent or enable training for a station.

    Case 1: A has no neighbors and include_target_pct=0.0 -> no training pool,
            so its values remain NaN / 'missing'.
    Case 2: A uses B as neighbor with include_target_pct=0.0 -> training pool
            contains B only, so A can be imputed.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")

    df = pd.DataFrame(
        {
            "station": ["A"] * 3 + ["B"] * 3,
            "date": list(dates) * 2,
            "lat": [10.0] * 3 + [11.0] * 3,
            "lon": [-100.0] * 3 + [-101.0] * 3,
            "alt": [2000.0] * 6,
            # A: all missing, B: fully observed
            "value": [np.nan, np.nan, np.nan, 5.0, 5.0, 5.0],
        }
    )

    # Case 1: no neighbors, no leakage -> cannot train
    out_empty = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        neighbor_map={"A": [], "B": []},
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
    )

    a_empty = out_empty[out_empty["station"] == "A"]
    # All values remain NaN and marked as 'missing'
    assert a_empty["value"].isna().all()
    assert set(a_empty["source"]) == {"missing"}

    # Case 2: A uses B as neighbor -> training pool is non-empty
    out_with_neighbors = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        neighbor_map={"A": ["B"], "B": ["A"]},
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
    )

    a_imp = out_with_neighbors[out_with_neighbors["station"] == "A"]
    # Now values should be imputed (non-NaN) and marked as 'imputed'
    assert a_imp["value"].notna().all()
    assert set(a_imp["source"]) == {"imputed"}


def test_impute_dataset_respects_start_end_window():
    """
    When start/end are provided, the output should only contain rows
    within that window.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    df = pd.DataFrame(
        {
            "station": ["S1"] * 5,
            "date": dates,
            "lat": [10.0] * 5,
            "lon": [-100.0] * 5,
            "alt": [2000.0] * 5,
            "value": [1.0, np.nan, 1.5, np.nan, 2.0],
        }
    )

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        start="2020-01-02",
        end="2020-01-04",
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
    )

    # Only dates from 2 to 4 should be present
    assert out["date"].min() == pd.Timestamp("2020-01-02")
    assert out["date"].max() == pd.Timestamp("2020-01-04")
    assert out.shape[0] == 3  # single station, 3 days

    # Still must have a source column for all rows
    assert "source" in out.columns
    assert out["source"].notna().all()
