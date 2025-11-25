# SPDX-License-Identifier: MIT
"""
Unit tests for missclimatepy.evaluate.evaluate_stations
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from missclimatepy.evaluate import evaluate_stations


def _make_simple_df() -> pd.DataFrame:
    """
    Build a tiny but non-trivial dataset with two stations, 20 days, and
    a simple target pattern that includes dry/wet days for stratified
    sampling and MCM baseline.

    - station S1: increasing values + some zeros
    - station S2: decreasing values + some zeros
    """
    dates = pd.date_range("2020-01-01", periods=20, freq="D")

    # Station 1: mostly increasing, zeros on some days
    s1_vals = np.arange(1, 21, dtype=float)
    s1_vals[::5] = 0.0  # dry days at regular intervals

    # Station 2: mostly decreasing, zeros on some days
    s2_vals = np.arange(20, 0, -1, dtype=float)
    s2_vals[::4] = 0.0  # different dry-day pattern

    df = pd.DataFrame(
        {
            "station": (["S1"] * 20) + (["S2"] * 20),
            "date": list(dates) * 2,
            "latitude": [10.0] * 20 + [20.0] * 20,
            "longitude": [-100.0] * 20 + [-99.0] * 20,
            "altitude": [1000.0] * 20 + [2000.0] * 20,
            "value": np.concatenate([s1_vals, s2_vals]),
        }
    )
    return df


def test_evaluate_basic_shapes_and_columns():
    """
    Basic smoke test:

    - evaluate_stations runs without error.
    - report has one row per station.
    - preds has only test rows with the expected columns.
    - key metric and metadata columns exist (including baseline _mcm and KGE).
    """
    df = _make_simple_df()

    report, preds = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="value",
        start="2020-01-01",
        end="2020-01-20",
        add_cyclic=True,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        include_target_pct=0.0,  # LOSO-like
        show_progress=False,
    )

    # Two stations -> two rows in report
    assert report.shape[0] == 2
    assert set(report["station"]) == {"S1", "S2"}

    # Preds should contain only these two stations
    assert not preds.empty
    assert set(preds["station"].unique()) == {"S1", "S2"}

    # Check expected columns in report
    expected_cols = {
        "station",
        "n_rows",
        "seconds",
        "rows_train",
        "rows_test",
        "used_k_neighbors",
        "include_target_pct",
        "latitude",
        "longitude",
        "altitude",
        "MAE_d",
        "RMSE_d",
        "R2_d",
        "KGE_d",
        "MAE_m",
        "RMSE_m",
        "R2_m",
        "KGE_m",
        "MAE_y",
        "RMSE_y",
        "R2_y",
        "KGE_y",
        "MAE_d_mcm",
        "RMSE_d_mcm",
        "R2_d_mcm",
        "KGE_d_mcm",
        "MAE_m_mcm",
        "RMSE_m_mcm",
        "R2_m_mcm",
        "KGE_m_mcm",
        "MAE_y_mcm",
        "RMSE_y_mcm",
        "R2_y_mcm",
        "KGE_y_mcm",
    }
    missing = expected_cols.difference(set(report.columns))
    assert not missing, f"Report is missing columns: {missing}"

    # Check expected columns in preds
    expected_pred_cols = {
        "station",
        "date",
        "latitude",
        "longitude",
        "altitude",
        "y_obs",
        "y_mod",
    }
    assert expected_pred_cols.issubset(preds.columns)

    # For each station, rows_test should match number of preds for that station
    for sid in ["S1", "S2"]:
        rows_test_report = int(report.loc[report["station"] == sid, "rows_test"].iloc[0])
        rows_test_pred = preds[preds["station"] == sid].shape[0]
        assert rows_test_report == rows_test_pred


def test_evaluate_respects_min_station_rows():
    """
    Stations with fewer than min_station_rows observed target values
    must be skipped.
    """
    df = _make_simple_df()

    # Force S2 to have very few observed values by setting most to NaN
    mask_s2 = df["station"] == "S2"
    df.loc[mask_s2, "value"] = np.nan
    # Keep just two non-NaN values for S2
    valid_idx = df[mask_s2].index[:2]
    df.loc[valid_idx, "value"] = [5.0, 6.0]

    report, preds = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="value",
        start="2020-01-01",
        end="2020-01-20",
        min_station_rows=5,  # S2 has only 2 observed rows
        model_kind="rf",
        model_params={"n_estimators": 5, "random_state": 0},
        include_target_pct=0.0,
        show_progress=False,
    )

    # Only S1 should remain
    assert set(report["station"]) == {"S1"}
    assert set(preds["station"].unique()) == {"S1"}


def test_evaluate_with_neighbor_map_and_model_params():
    """
    Using an explicit neighbor_map should not crash and should set
    used_k_neighbors to NaN (since lengths may vary). Also check that
    custom model_params are accepted.
    """
    df = _make_simple_df()

    neighbor_map = {
        "S1": ["S2"],  # S1 trained only on S2 + leakage
        "S2": ["S1"],  # S2 trained only on S1 + leakage
    }

    report, preds = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="value",
        start="2020-01-01",
        end="2020-01-20",
        neighbor_map=neighbor_map,
        k_neighbors=99,  # should be ignored because neighbor_map is provided
        model_kind="rf",
        model_params={"n_estimators": 5, "random_state": 123},
        include_target_pct=10.0,
        show_progress=False,
    )

    # Both stations evaluated
    assert set(report["station"]) == {"S1", "S2"}
    assert set(preds["station"].unique()) == {"S1", "S2"}

    # When neighbor_map is provided, used_k_neighbors should be NaN
    assert report["used_k_neighbors"].isna().all()


def test_evaluate_includes_kge_and_baseline_metrics():
    """
    Ensure that KGE and baseline (_mcm) metrics are present and numeric
    (or NaN) for all stations.
    """
    df = _make_simple_df()

    report, _ = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="value",
        start="2020-01-01",
        end="2020-01-20",
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 1},
        include_target_pct=20.0,
        show_progress=False,
    )

    kge_cols = ["KGE_d", "KGE_m", "KGE_y", "KGE_d_mcm", "KGE_m_mcm", "KGE_y_mcm"]
    for col in kge_cols:
        assert col in report.columns
        # Should be float dtype; values may be NaN or finite floats
        assert report[col].dtype.kind in {"f"}
