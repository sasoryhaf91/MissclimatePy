# tests/test_masking.py
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import pytest

from missclimatepy.masking import (
    percent_missing_between,
    gap_profile_by_station,
    missing_matrix,
    describe_missing,
    apply_random_mask_by_station,
)


def _make_simple_df():
    """Small synthetic dataset with two stations and daily data."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    data = []

    # Station S1: full coverage (no missing)
    for d in dates:
        data.append({"station": "S1", "date": d, "value": 1.0})

    # Station S2: only two observed days (others missing)
    # Observed on 1st and 3rd, missing otherwise
    for d, val in zip(dates, [1.0, np.nan, 2.0, np.nan, np.nan]):
        data.append({"station": "S2", "date": d, "value": val})

    return pd.DataFrame(data)


def test_percent_missing_between_basic():
    df = _make_simple_df()

    out = percent_missing_between(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
        start="2020-01-01",
        end="2020-01-05",
    )

    # Two stations
    assert set(out["station"]) == {"S1", "S2"}
    assert (out["total_days"] == 5).all()

    # S1: full coverage
    s1 = out.loc[out["station"] == "S1"].iloc[0]
    assert s1["observed_days"] == 5
    assert s1["missing_days"] == 0
    assert np.isclose(s1["coverage"], 1.0)
    assert np.isclose(s1["percent_missing"], 0.0)

    # S2: 2 observed days out of 5
    s2 = out.loc[out["station"] == "S2"].iloc[0]
    assert s2["observed_days"] == 2
    assert s2["missing_days"] == 3
    assert np.isclose(s2["coverage"], 2 / 5)
    assert np.isclose(s2["percent_missing"], 100.0 * 3 / 5)


def test_gap_profile_by_station_basic():
    # Build a single-station dataset with known gap structure:
    # dates: 1..5, values: [1, NaN, NaN, 2, NaN]
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    values = [1.0, np.nan, np.nan, 2.0, np.nan]
    df = pd.DataFrame(
        {"station": "S1", "date": dates, "value": values}
    )

    out = gap_profile_by_station(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["station"] == "S1"

    # Boolean valid by day: [True, False, False, True, False]
    # Missing runs: lengths [2, 1] → n_gaps = 2, mean = 1.5, max = 2
    assert row["n_gaps"] == 2
    assert np.isclose(row["mean_gap"], 1.5)
    assert row["max_gap"] == 2


def test_missing_matrix_with_global_window():
    df = _make_simple_df()

    mat = missing_matrix(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
        start="2020-01-01",
        end="2020-01-05",
        sort_by_coverage=True,
        as_uint8=True,
    )

    # Shape: 2 stations × 5 days
    assert mat.shape == (2, 5)
    assert mat.dtypes.unique()[0] == np.dtype("uint8")

    # Row order: S1 first (full coverage), then S2 (lower coverage)
    assert list(mat.index) == ["S1", "S2"]

    # S1: all ones
    assert (mat.loc["S1"] == 1).all()

    # S2: ones only on days 1 and 3
    s2_vals = mat.loc["S2"].to_numpy()
    assert s2_vals.tolist() == [1, 0, 1, 0, 0]


def test_describe_missing_default_span():
    df = _make_simple_df()

    out = describe_missing(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
    )

    # Should have one row per station
    assert set(out["station"]) == {"S1", "S2"}

    # Columns present
    expected_cols = {
        "station",
        "total_days",
        "observed_days",
        "missing_days",
        "coverage",
        "percent_missing",
        "n_gaps",
        "mean_gap",
        "max_gap",
    }
    assert expected_cols.issubset(out.columns)

    # Sorted by percent_missing descending
    pm = out["percent_missing"].to_numpy()
    assert np.all(pm[:-1] >= pm[1:])


def test_apply_random_mask_by_station_respects_percent_and_seed():
    df = _make_simple_df()

    # Only mask station S1 to make counting easy:
    # S1 has 5 non-null values; with 40% we expect floor(5*0.4) = 2 masked.
    df_masked = apply_random_mask_by_station(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
        percent_to_mask=40.0,
        random_state=123,
        only_with_observation=True,
    )

    # Station S1 originally had 5 observed values
    s1_original = df.loc[df["station"] == "S1", "value"]
    assert s1_original.notna().sum() == 5

    s1_masked = df_masked.loc[df_masked["station"] == "S1", "value"]
    # Expect exactly 3 non-nulls left → 2 masked
    assert s1_masked.notna().sum() == 3

    # Station S2 should not gain extra NaNs beyond the existing ones
    s2_original_nans = df.loc[df["station"] == "S2", "value"].isna().sum()
    s2_masked_nans = df_masked.loc[df_masked["station"] == "S2", "value"].isna().sum()
    # It may change if random picks existing NaNs (no-ops), but should never
    # reduce the number of NaNs:
    assert s2_masked_nans >= s2_original_nans


def test_apply_random_mask_by_station_invalid_percent_raises():
    df = _make_simple_df()
    with pytest.raises(ValueError):
        apply_random_mask_by_station(
            df,
            id_col="station",
            date_col="date",
            target_col="value",
            percent_to_mask=-1.0,
        )

    with pytest.raises(ValueError):
        apply_random_mask_by_station(
            df,
            id_col="station",
            date_col="date",
            target_col="value",
            percent_to_mask=120.0,
        )
