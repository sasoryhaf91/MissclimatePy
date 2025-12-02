# tests/test_masking.py
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from missclimatepy.masking import (
    percent_missing_between,
    gap_profile_by_station,
    missing_matrix,
    describe_missing,
    apply_random_mask_by_station,
)


def _make_simple_df() -> pd.DataFrame:
    """
    Small helper dataset used across tests.

    Stations:
    - S1: full coverage, 5 consecutive days, all observed.
    - S2: only days 2..5, with a mix of observed and missing.

    Dates: 2020-01-01 .. 2020-01-05
    Values:
        S1: [1, 1, 1, 1, 1]
        S2: [NaN, 2, NaN, NaN] on days 2..5
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    df_s1 = pd.DataFrame(
        {
            "station": "S1",
            "date": dates,
            "value": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    df_s2 = pd.DataFrame(
        {
            "station": ["S2"] * 4,
            "date": dates[1:],  # 2..5
            "value": [np.nan, 2.0, np.nan, np.nan],
        }
    )

    return pd.concat([df_s1, df_s2], ignore_index=True)


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

    # Expected columns
    expected_cols = {
        "station",
        "total_days",
        "observed_days",
        "missing_days",
        "coverage",
        "percent_missing",
    }
    assert expected_cols.issubset(out.columns)

    # Sort by station for deterministic checks
    out = out.sort_values("station").reset_index(drop=True)

    # S1: full coverage 5/5
    row_s1 = out[out["station"] == "S1"].iloc[0]
    assert row_s1["total_days"] == 5
    assert row_s1["observed_days"] == 5
    assert row_s1["missing_days"] == 0
    assert np.isclose(row_s1["coverage"], 1.0)
    assert np.isclose(row_s1["percent_missing"], 0.0)

    # S2: only day 3 observed within [1..5]
    row_s2 = out[out["station"] == "S2"].iloc[0]
    assert row_s2["total_days"] == 5
    assert row_s2["observed_days"] == 1
    assert row_s2["missing_days"] == 4
    assert np.isclose(row_s2["coverage"], 1.0 / 5.0)
    assert np.isclose(row_s2["percent_missing"], 80.0)


def test_gap_profile_by_station_basic():
    # Single-station dataset with known gap structure:
    # dates: 1..5, values: [1, NaN, NaN, 2, NaN]
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    values = [1.0, np.nan, np.nan, 2.0, np.nan]
    df = pd.DataFrame({"station": "S1", "date": dates, "value": values})

    out = gap_profile_by_station(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
    )

    assert list(out.columns) == ["station", "n_gaps", "mean_gap", "max_gap"]
    assert len(out) == 1
    row = out.iloc[0]

    # Missing runs:
    #   day 2-3 → length 2
    #   day 5   → length 1
    # => n_gaps = 2, mean_gap = 1.5, max_gap = 2
    assert row["station"] == "S1"
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

    # Two stations, five days
    assert mat.shape == (2, 5)
    assert mat.index.isin(["S1", "S2"]).all()

    # Values must be 0/1 and uint-like when as_uint8=True
    assert issubclass(mat.values.dtype.type, np.unsignedinteger)
    assert set(np.unique(mat.values)).issubset({0, 1})

    # S1 has full coverage in the window: row sum = 5
    row_s1 = mat.loc["S1"]
    assert row_s1.sum() == 5

    # S2: only date 2020-01-03 observed
    cols = list(mat.columns)
    idx_day3 = cols.index(pd.Timestamp("2020-01-03"))
    row_s2 = mat.loc["S2"]
    assert row_s2.iloc[idx_day3] == 1
    assert row_s2.sum() == 1


def test_describe_missing_default_span():
    df = _make_simple_df()

    out = describe_missing(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
    )

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

    out = out.sort_values("station").reset_index(drop=True)

    # S1: span 1..5, all observed
    row_s1 = out[out["station"] == "S1"].iloc[0]
    assert row_s1["total_days"] == 5
    assert row_s1["observed_days"] == 5
    assert row_s1["missing_days"] == 0
    assert np.isclose(row_s1["coverage"], 1.0)
    assert np.isclose(row_s1["percent_missing"], 0.0)
    # No gaps (continuous observed)
    assert row_s1["n_gaps"] == 0
    assert np.isclose(row_s1["mean_gap"], 0.0)
    assert row_s1["max_gap"] == 0

    # S2: span 2..5 → 4 days, only date 3 observed
    row_s2 = out[out["station"] == "S2"].iloc[0]
    assert row_s2["total_days"] == 4
    assert row_s2["observed_days"] == 1
    assert row_s2["missing_days"] == 3
    assert np.isclose(row_s2["coverage"], 0.25)
    assert np.isclose(row_s2["percent_missing"], 75.0)

    # Gap structure for S2:
    #   span: 2..5
    #   mask: [NaN, 2, NaN, NaN] → observed at day 3 only
    #   missing runs: day2 (len1), day4-5 (len2) → n_gaps=2, mean=1.5, max=2
    assert row_s2["n_gaps"] == 2
    assert np.isclose(row_s2["mean_gap"], 1.5)
    assert row_s2["max_gap"] == 2


def test_apply_random_mask_by_station_respects_percent_and_seed():
    # Simple single-station dataset: 5 observed values
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "station": "S1",
            "date": dates,
            "value": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    # With 40% masking, we expect floor(5 * 0.4) = 2 masked values.
    df_masked_1 = apply_random_mask_by_station(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
        percent_to_mask=40.0,
        random_state=123,
        only_with_observation=True,
    )

    # Check total number of NaNs
    n_nans_1 = df_masked_1["value"].isna().sum()
    assert n_nans_1 == 2

    # Same parameters & seed → identical masking pattern
    df_masked_2 = apply_random_mask_by_station(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
        percent_to_mask=40.0,
        random_state=123,
        only_with_observation=True,
    )
    pd.testing.assert_series_equal(df_masked_1["value"], df_masked_2["value"])


def test_apply_random_mask_invalid_percent_raises():
    df = _make_simple_df()

    for bad in (-1.0, 120.0):
        try:
            apply_random_mask_by_station(
                df,
                id_col="station",
                date_col="date",
                target_col="value",
                percent_to_mask=bad,
            )
        except ValueError:
            # Expected
            continue
        else:
            raise AssertionError("Expected ValueError for invalid percent_to_mask")
