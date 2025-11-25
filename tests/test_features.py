# tests/test_features.py
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np
import pytest

from missclimatepy.features import (
    validate_required_columns,
    ensure_datetime_naive,
    add_calendar_features,
    default_feature_names,
    preprocess_for_model,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_df():
    return pd.DataFrame(
        {
            "station": ["A", "A", "B", "B"],
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-01",
                "2020-01-02",
            ],
            "lat": [10.0, 10.0, 11.0, 11.0],
            "lon": [20.0, 20.0, 21.0, 21.0],
            "alt": [100.0, 100.0, 200.0, 200.0],
            "tmin": [5.0, 6.0, 7.0, 8.0],
        }
    )


# ---------------------------------------------------------------------------
# validate_required_columns
# ---------------------------------------------------------------------------

def test_validate_required_columns_ok():
    df = _make_simple_df()
    # Should not raise
    validate_required_columns(df, ["station", "date", "lat"])


def test_validate_required_columns_raises():
    df = _make_simple_df()
    with pytest.raises(ValueError) as exc:
        validate_required_columns(df, ["station", "date", "missing_col"])
    msg = str(exc.value)
    assert "Missing required columns" in msg
    assert "missing_col" in msg


# ---------------------------------------------------------------------------
# ensure_datetime_naive
# ---------------------------------------------------------------------------

def test_ensure_datetime_naive_parses_and_drops_tz():
    s = pd.Series(
        [
            "2020-01-01",
            pd.Timestamp("2020-01-02", tz="UTC"),
            "not-a-date",
        ]
    )

    out = ensure_datetime_naive(s)

    # Debe ser datetime64[ns] sin tz
    assert pd.api.types.is_datetime64_ns_dtype(out)
    # La fila 0 debe ser 2020-01-01
    assert out.iloc[0] == pd.Timestamp("2020-01-01")
    # La fila 1 debe ser 2020-01-02 (sin tz)
    assert out.iloc[1] == pd.Timestamp("2020-01-02")
    # La fila 2 debe ser NaT
    assert pd.isna(out.iloc[2])


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------

def test_add_calendar_features_basic():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-02-15"]})
    out = add_calendar_features(df, date_col="date", add_cyclic=False)

    for c in ["year", "month", "doy"]:
        assert c in out.columns

    assert list(out["year"]) == [2020, 2020]
    assert list(out["month"]) == [1, 2]
    # doy numéricamente correcto
    assert int(out.loc[0, "doy"]) == 1
    assert int(out.loc[1, "doy"]) == 46  # 15 feb 2020


def test_add_calendar_features_with_cyclic():
    df = pd.DataFrame({"date": ["2020-01-01", "2020-06-30"]})
    out = add_calendar_features(df, date_col="date", add_cyclic=True)

    for c in ["year", "month", "doy", "doy_sin", "doy_cos"]:
        assert c in out.columns

    # Harmónicos deben ser numéricos sin NaN
    assert np.isfinite(out["doy_sin"]).all()
    assert np.isfinite(out["doy_cos"]).all()


# ---------------------------------------------------------------------------
# default_feature_names
# ---------------------------------------------------------------------------

def test_default_feature_names_no_cyclic():
    feats = default_feature_names(lat_col="lat", lon_col="lon", alt_col="alt", add_cyclic=False)
    assert feats == ["lat", "lon", "alt", "year", "month", "doy"]


def test_default_feature_names_with_cyclic_and_extra():
    feats = default_feature_names(
        lat_col="LAT",
        lon_col="LON",
        alt_col="Z",
        add_cyclic=True,
        extra=["extra1", "doy", "extra2", "extra1"],
    )
    # Debe deduplicar manteniendo orden, y añadir sin/cos
    assert feats == [
        "LAT",
        "LON",
        "Z",
        "year",
        "month",
        "doy",
        "doy_sin",
        "doy_cos",
        "extra1",
        "extra2",
    ]


# ---------------------------------------------------------------------------
# preprocess_for_model
# ---------------------------------------------------------------------------

def test_preprocess_for_model_default_features_no_cyclic():
    df = _make_simple_df()

    prep, feats = preprocess_for_model(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start=None,
        end=None,
        add_cyclic=False,
        feature_cols=None,
    )

    # Feature list should be the default (no cyclic)
    assert feats == ["lat", "lon", "alt", "year", "month", "doy"]

    # Prepared df must contain base + calendar fields
    for c in ["station", "date", "lat", "lon", "alt", "tmin", "year", "month", "doy"]:
        assert c in prep.columns

    # No cyclic columns
    assert "doy_sin" not in prep.columns
    assert "doy_cos" not in prep.columns


def test_preprocess_for_model_default_features_with_cyclic():
    df = _make_simple_df()

    prep, feats = preprocess_for_model(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start=None,
        end=None,
        add_cyclic=True,
        feature_cols=None,
    )

    assert "doy_sin" in prep.columns
    assert "doy_cos" in prep.columns

    assert feats == ["lat", "lon", "alt", "year", "month", "doy", "doy_sin", "doy_cos"]


def test_preprocess_for_model_with_custom_features():
    df = _make_simple_df()
    df["extra"] = [1, 2, 3, 4]

    prep, feats = preprocess_for_model(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start=None,
        end=None,
        add_cyclic=True,  # no debería afectar porque pasamos feature_cols
        feature_cols=["lat", "lon", "alt", "extra"],
    )

    # Feature list is exactly what we passed (deduped, same order)
    assert feats == ["lat", "lon", "alt", "extra"]

    # Calendar fields should still be present in the prepared DataFrame
    for col in ["year", "month", "doy"]:
        assert col in prep.columns

    # Extra debe estar presente
    assert "extra" in prep.columns


def test_preprocess_for_model_with_window_filter():
    df = _make_simple_df()

    prep, feats = preprocess_for_model(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start="2020-01-02",
        end="2020-01-02",
        add_cyclic=False,
        feature_cols=None,
    )

    # Sólo debe haber filas del 2 de enero de 2020
    assert prep["date"].nunique() == 1
    assert pd.to_datetime("2020-01-02") in prep["date"].unique()

    # Features siguen siendo los mismos
    assert feats == ["lat", "lon", "alt", "year", "month", "doy"]


def test_preprocess_for_model_empty_after_window_returns_empty_df():
    df = _make_simple_df()

    prep, feats = preprocess_for_model(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start="2030-01-01",
        end="2030-12-31",
        add_cyclic=False,
        feature_cols=None,
    )

    # No rows
    assert prep.empty

    # Column scaffold should at least contain the base + default feature columns
    expected_feats = ["lat", "lon", "alt", "year", "month", "doy"]
    for c in ["station", "date", "lat", "lon", "alt", "tmin"] + expected_feats:
        assert c in prep.columns

    # And feats must match the default
    assert feats == expected_feats
