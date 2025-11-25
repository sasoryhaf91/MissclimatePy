# SPDX-License-Identifier: MIT
"""
Tests for missclimatepy.features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from missclimatepy.features import (
    ensure_datetime_naive,
    validate_required_columns,
    add_time_features,
    select_station_ids,
    filter_by_min_station_rows,
)


# --------------------------------------------------------------------------- #
# ensure_datetime_naive
# --------------------------------------------------------------------------- #


def test_ensure_datetime_naive_parses_strings_and_is_naive():
    s = pd.Series(["2020-01-01", "2020-01-02", "not-a-date"])
    out = ensure_datetime_naive(s)

    # dtype datetime64[ns] and unparsable -> NaT
    assert str(out.dtype) == "datetime64[ns]"
    assert out.isna().sum() == 1


def test_ensure_datetime_naive_drops_timezone():
    base = pd.Series(["2020-01-01", "2020-01-02"])
    tz_aware = pd.to_datetime(base).dt.tz_localize("UTC")

    out = ensure_datetime_naive(tz_aware)

    assert str(out.dtype) == "datetime64[ns]"
    # naive series must not have tz information
    assert getattr(out.dt, "tz", None) is None


# --------------------------------------------------------------------------- #
# validate_required_columns
# --------------------------------------------------------------------------- #


def test_validate_required_columns_ok():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    # Should not raise
    validate_required_columns(df, ["a", "b"])


def test_validate_required_columns_raises_with_context():
    df = pd.DataFrame({"a": [1], "b": [2]})

    with pytest.raises(ValueError) as exc:
        validate_required_columns(df, ["a", "c"], context="my_func")

    msg = str(exc.value)
    assert "[my_func]" in msg
    assert "['c']" in msg  # missing column name is mentioned


# --------------------------------------------------------------------------- #
# add_time_features
# --------------------------------------------------------------------------- #


def test_add_time_features_basic_fields():
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"date": dates})

    out = add_time_features(df, date_col="date", add_cyclic=False)

    for col in ["year", "month", "doy"]:
        assert col in out.columns

    assert out["year"].iloc[0] == 2020
    assert out["month"].iloc[0] == 1
    assert out["doy"].tolist() == [1, 2, 3]


def test_add_time_features_cyclic_added_when_requested():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({"date": dates})

    out = add_time_features(df, date_col="date", add_cyclic=True)

    for col in ["year", "month", "doy", "doy_sin", "doy_cos"]:
        assert col in out.columns

    # Values must be finite and within [-1, 1]
    assert np.isfinite(out["doy_sin"]).all()
    assert np.isfinite(out["doy_cos"]).all()
    assert (out["doy_sin"].abs() <= 1 + 1e-8).all()
    assert (out["doy_cos"].abs() <= 1 + 1e-8).all()


# --------------------------------------------------------------------------- #
# select_station_ids
# --------------------------------------------------------------------------- #


def _make_station_df():
    return pd.DataFrame(
        {
            "station": ["15001", "15002", "32010", "99999"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_select_station_ids_default_returns_all():
    df = _make_station_df()
    out = select_station_ids(df, id_col="station")
    assert out == ["15001", "15002", "32010", "99999"]


def test_select_station_ids_prefix_single_and_multi():
    df = _make_station_df()

    out = select_station_ids(df, id_col="station", prefix="15")
    assert out == ["15001", "15002"]

    out2 = select_station_ids(df, id_col="station", prefix=["15", "32"])
    assert out2 == ["15001", "15002", "32010"]


def test_select_station_ids_station_ids_and_or_semantics():
    df = _make_station_df()

    out = select_station_ids(
        df,
        id_col="station",
        prefix="15",
        station_ids=["99999"],
    )
    # union of prefix + explicit ids, ordered by first appearance in df
    assert out == ["15001", "15002", "99999"]


def test_select_station_ids_regex_and_custom_filter():
    df = _make_station_df()

    # regex for the "32..." station
    out = select_station_ids(df, id_col="station", regex=r"^32")
    assert out == ["32010"]

    # custom_filter: keep stations whose numeric id is even
    out2 = select_station_ids(
        df,
        id_col="station",
        custom_filter=lambda s: int(s) % 2 == 0,
    )
    # 15002 and 32010 are even
    assert out2 == ["15002", "32010"]


def test_select_station_ids_fallback_to_all_if_no_match():
    df = _make_station_df()

    out = select_station_ids(df, id_col="station", prefix="00")  # no station starts with 00
    # Should gracefully fall back to all stations
    assert out == ["15001", "15002", "32010", "99999"]


# --------------------------------------------------------------------------- #
# filter_by_min_station_rows
# --------------------------------------------------------------------------- #


def test_filter_by_min_station_rows_counts_non_missing_only():
    df = pd.DataFrame(
        {
            "station": ["A", "A", "A", "B", "B", "C"],
            "value": [1.0, np.nan, 2.0, np.nan, np.nan, 5.0],
        }
    )
    # Non-null counts: A -> 2, B -> 0, C -> 1

    out2 = filter_by_min_station_rows(
        df,
        id_col="station",
        target_col="value",
        min_station_rows=2,
    )
    assert out2 == ["A"]

    out1 = filter_by_min_station_rows(
        df,
        id_col="station",
        target_col="value",
        min_station_rows=1,
    )
    # A and C meet the threshold; order is the natural station order
    assert out1 == ["A", "C"]

    out0 = filter_by_min_station_rows(
        df,
        id_col="station",
        target_col="value",
        min_station_rows=0,
    )
    # min_station_rows <= 0 => all stations
    assert out0 == ["A", "B", "C"]
