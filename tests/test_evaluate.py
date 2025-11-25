# SPDX-License-Identifier: MIT
"""
Tests for missclimatepy.evaluate.evaluate_stations.

These tests focus on:
- basic shapes and required columns in the outputs,
- station filtering by min_station_rows,
- behaviour with and without target leakage,
- support for multiple model kinds.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from missclimatepy.evaluate import evaluate_stations
from missclimatepy.models import SUPPORTED_MODELS


# ---------------------------------------------------------------------------
# Small helper datasets
# ---------------------------------------------------------------------------


def _make_simple_df() -> pd.DataFrame:
    """
    Two stations (A, B) with 4 daily records each and a simple tmin pattern.
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rows = []

    # Station A: lat=10, lon=20, alt=100
    for i, d in enumerate(dates, start=0):
        rows.append(
            {
                "station": "A",
                "date": d,
                "lat": 10.0,
                "lon": 20.0,
                "alt": 100.0,
                "tmin": 5.0 + i,
            }
        )

    # Station B: lat=11, lon=21, alt=200
    for i, d in enumerate(dates, start=0):
        rows.append(
            {
                "station": "B",
                "date": d,
                "lat": 11.0,
                "lon": 21.0,
                "alt": 200.0,
                "tmin": 7.0 + i,
            }
        )

    return pd.DataFrame(rows)


def _make_precip_df(n_per_station: int = 10) -> pd.DataFrame:
    """
    Two stations (A, B) with n_per_station daily precipitation records.

    Precipitation alternates between 0 (dry) and 3 (wet) to exercise
    the month × dry/wet stratified leakage sampler.
    """
    rows = []
    base_dates = pd.date_range("2020-01-01", periods=n_per_station, freq="D")

    for st, (lat, lon, alt) in [("A", (10.0, 20.0, 100.0)), ("B", (11.0, 21.0, 200.0))]:
        for i, d in enumerate(base_dates):
            rows.append(
                {
                    "station": st,
                    "date": d,
                    "lat": lat,
                    "lon": lon,
                    "alt": alt,
                    "prec": 0.0 if i % 2 == 0 else 3.0,
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evaluate_basic_shapes_and_columns():
    df = _make_simple_df()

    report, preds = evaluate_stations(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        k_neighbors=None,  # use all-other-stations pool
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        show_progress=False,
    )

    # Report: one row per station
    assert report.shape[0] == 2
    assert {"station", "rows_train", "rows_test", "MAE_d", "RMSE_d", "R2_d"}.issubset(
        set(report.columns)
    )
    assert {"latitude", "longitude", "altitude"}.issubset(set(report.columns))

    # Predictions: both stations should appear
    assert "station" in preds.columns
    assert preds["station"].nunique() == 2
    assert {"latitude", "longitude", "altitude", "y_obs", "y_mod"}.issubset(
        set(preds.columns)
    )

    # There must be some training and test rows per station
    for rows_tr in report["rows_train"]:
        assert rows_tr >= 0
    for rows_te in report["rows_test"]:
        assert rows_te >= 0


def test_evaluate_respects_min_station_rows():
    df = _make_simple_df()

    # Drop most rows from station B so it has only 1 observation
    df_reduced = df[~((df["station"] == "B") & (df["date"] > pd.Timestamp("2020-01-01")))]

    report, _ = evaluate_stations(
        df_reduced,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        k_neighbors=None,
        include_target_pct=0.0,
        min_station_rows=2,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        show_progress=False,
    )

    # Only station A should remain
    stations_in_report = set(report["station"].tolist())
    assert stations_in_report == {"A"}


def test_evaluate_LOSO_single_station_produces_empty_train():
    """
    With a single station and include_target_pct=0, LOSO-like evaluation
    cannot use any training rows, so rows_train must be zero.
    """
    df = _make_simple_df()
    df_single = df[df["station"] == "A"].copy()

    report, _ = evaluate_stations(
        df_single,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        k_neighbors=None,
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 5, "random_state": 0, "n_jobs": 1},
        show_progress=False,
    )

    # Exactly one row in the report and zero training rows
    assert report.shape[0] == 1
    assert int(report["rows_train"].iloc[0]) == 0
    # There should still be test rows (all valid rows for that station)
    assert int(report["rows_test"].iloc[0]) > 0 or np.isnan(
        report["rows_test"].iloc[0]
    ) is False


def test_evaluate_with_leakage_increases_rows_train():
    df = _make_precip_df(n_per_station=10)

    # LOSO-like (no leakage)
    report_0, _ = evaluate_stations(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="prec",
        k_neighbors=None,
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        show_progress=False,
    )

    # With 50% leakage
    report_50, _ = evaluate_stations(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="prec",
        k_neighbors=None,
        include_target_pct=50.0,
        include_target_seed=123,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        show_progress=False,
    )

    # Build dicts {station -> rows_train}
    rt0: Dict[str, int] = {
        str(s): int(r) for s, r in zip(report_0["station"], report_0["rows_train"])
    }
    rt50: Dict[str, int] = {
        str(s): int(r) for s, r in zip(report_50["station"], report_50["rows_train"])
    }

    # For every station, training rows with leakage must be >= without leakage,
    # and strictly greater when there are valid target rows.
    for sid in rt0:
        assert sid in rt50
        assert rt50[sid] >= rt0[sid]
        assert rt50[sid] > rt0[sid]


def test_evaluate_supports_multiple_model_kinds():
    df = _make_simple_df()

    # We expect at least these core models to exist in the registry.
    candidate_kinds = [k for k in ("rf", "etr", "linreg") if k in SUPPORTED_MODELS]

    assert "rf" in candidate_kinds  # sanity check: RF must be available

    results = {}
    for kind in candidate_kinds:
        report, preds = evaluate_stations(
            df,
            id_col="station",
            date_col="date",
            lat_col="lat",
            lon_col="lon",
            alt_col="alt",
            target_col="tmin",
            k_neighbors=None,
            include_target_pct=0.0,
            model_kind=kind,
            model_params={"random_state": 0} if kind != "linreg" else {},
            show_progress=False,
        )
        results[kind] = (report, preds)

    # All models should return a non-empty report with the same number of stations
    n_stations_expected = 2
    for kind, (rep, preds) in results.items():
        assert rep.shape[0] == n_stations_expected, f"{kind} report rows mismatch"
        assert preds["station"].nunique() == n_stations_expected, f"{kind} preds stations mismatch"


def test_evaluate_predictions_include_coordinates_and_values():
    df = _make_simple_df()

    _, preds = evaluate_stations(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        k_neighbors=None,
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        show_progress=False,
    )

    # Columns must be present
    for col in ["station", "date", "latitude", "longitude", "altitude", "y_obs", "y_mod"]:
        assert col in preds.columns

    # There should be at least one prediction
    assert len(preds) > 0
