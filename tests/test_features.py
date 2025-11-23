# tests/test_features.py
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
import pytest

from missclimatepy.features import (
    ensure_datetime_naive,
    add_calendar_features,
    default_feature_names,
    validate_required_columns,
)


def test_ensure_datetime_naive_converts_strings_and_drops_tz():
    s = pd.Series(["2020-01-01", "2020-01-02"])
    out = ensure_datetime_naive(s)

    assert pd.api.types.is_datetime64_ns_dtype(out)
    # sin tz
    assert out.dt.tz is None
    assert out.iloc[0].year == 2020
    assert out.iloc[0].month == 1
    assert out.iloc[0].day == 1


def test_add_calendar_features_creates_expected_columns():
    df = pd.DataFrame({"date": pd.date_range("2000-01-01", periods=3, freq="D")})

    out = add_calendar_features(df.copy(), "date", add_cyclic=True)

    for col in ["year", "month", "day", "doy"]:
        assert col in out.columns

    # columnas cíclicas
    for col in ["doy_sin", "doy_cos"]:
        assert col in out.columns
        assert np.issubdtype(out[col].dtype, np.floating)


def test_default_feature_names_without_cyclic():
    feats = default_feature_names(
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        add_cyclic=False,
    )
    assert feats == ["lat", "lon", "alt", "year", "month", "doy"]


def test_default_feature_names_with_cyclic():
    feats = default_feature_names(
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        add_cyclic=True,
    )
    # orden exacto
    assert feats == [
        "lat",
        "lon",
        "alt",
        "year",
        "month",
        "doy",
        "doy_sin",
        "doy_cos",
    ]


def test_validate_required_columns_ok():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "date": pd.date_range("2000-01-01", periods=2),
            "lat": [10.0, 11.0],
        }
    )
    # no debería lanzar error
    validate_required_columns(df, ["id", "date", "lat"], context="unit_test")


def test_validate_required_columns_missing_raises():
    df = pd.DataFrame({"id": [1, 2]})

    with pytest.raises(KeyError):
        validate_required_columns(df, ["id", "date"], context="unit_test")
