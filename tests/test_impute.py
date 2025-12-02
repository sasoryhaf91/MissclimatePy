import numpy as np
import pandas as pd

from missclimatepy.impute import impute_dataset


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_two_station_df() -> pd.DataFrame:
    """
    Simple toy dataset with two stations (A, B) and 3 days.

    A has a missing value in the middle; B is fully observed.
    """
    dates = pd.date_range("2020-01-01", periods=3, freq="D")

    df = pd.DataFrame(
        {
            "station": ["A"] * 3 + ["B"] * 3,
            "date": list(dates) * 2,
            "lat": [10.0] * 3 + [11.0] * 3,
            "lon": [-100.0] * 3 + [-101.0] * 3,
            "alt": [2000.0] * 6,
            # A: [1, NaN, 3], B: [5, 5, 5]
            "value": [1.0, np.nan, 3.0, 5.0, 5.0, 5.0],
        }
    )
    return df


def _make_min_rows_df() -> pd.DataFrame:
    """
    Two stations (S1, S2); S1 has only one observed value,
    S2 is fully observed.
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    df = pd.DataFrame(
        {
            "station": ["S1"] * 4 + ["S2"] * 4,
            "date": list(dates) * 2,
            "lat": [10.0] * 4 + [11.0] * 4,
            "lon": [-100.0] * 4 + [-101.0] * 4,
            "alt": [2000.0] * 8,
            # S1: one observed + 3 NaN; S2: fully observed
            "value": [1.0, np.nan, np.nan, np.nan, 2.0, 2.0, 2.0, 2.0],
        }
    )
    return df


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_impute_dataset_fills_missing_and_marks_source():
    """
    Basic smoke test:

    - A has a gap that must be imputed using B as neighbor.
    - B is fully observed.
    - Output contains full 3-day series for both A and B.
    - `source` correctly marks 'observed' vs 'imputed'.
    """
    df = _make_two_station_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        k_neighbors=1,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    # We should get 3 days * 2 stations
    assert out.shape[0] == 6

    # Station A: 3 rows, with exactly one imputed day (the original NaN)
    a = out[out["station"] == "A"].copy().sort_values("date")
    assert a.shape[0] == 3
    assert set(a["source"]) <= {"observed", "imputed"}
    assert (a["value"].isna().sum()) == 0  # all filled
    # Original missing was the middle date
    assert (a.loc[a["source"] == "imputed", "date"].tolist() ==
            [pd.Timestamp("2020-01-02")])

    # Station B: fully observed, all days should stay 'observed'
    b = out[out["station"] == "B"].copy().sort_values("date")
    assert b.shape[0] == 3
    assert set(b["source"]) == {"observed"}
    # Values must match original
    orig_b = df[df["station"] == "B"].sort_values("date")["value"].to_numpy()
    np.testing.assert_allclose(b["value"].to_numpy(), orig_b)


def test_impute_dataset_respects_min_station_rows_excluding_stations():
    """
    Stations below `min_station_rows` are *excluded* from the output.

    - S1 does not meet the threshold -> should not appear in the result.
    - S2 is fully observed -> appears in the result with full series,
      all marked as 'observed'.
    """
    df = _make_min_rows_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        min_station_rows=2,  # S1 does not meet this threshold
        k_neighbors=1,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    # S1 should be completely absent
    assert "S1" not in set(out["station"])

    # S2 must be present with 4 days, all observed
    s2 = out[out["station"] == "S2"].copy().sort_values("date")
    assert s2.shape[0] == 4
    assert set(s2["source"]) == {"observed"}
    orig_s2 = df[df["station"] == "S2"].sort_values("date")["value"].to_numpy()
    np.testing.assert_allclose(s2["value"].to_numpy(), orig_s2)


def test_impute_dataset_respects_start_end_window_for_target_station():
    """
    When start/end are provided and prefix selects a single station:

    - The grid for that station is restricted to the requested window.
    - Missing days inside that window are imputed.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    df = pd.DataFrame(
        {
            "station": ["S1"] * 5 + ["S2"] * 5,
            "date": list(dates) * 2,
            "lat": [10.0] * 5 + [11.0] * 5,
            "lon": [-100.0] * 5 + [-101.0] * 5,
            "alt": [2000.0] * 10,
            # S1: values with gaps on days 2 and 4
            "value": [1.0, np.nan, 1.5, np.nan, 2.0] + [2.0] * 5,
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
        prefix=["S1"],  # only impute S1; S2 used as neighbor
        k_neighbors=1,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    # Only S1 should be in the output
    assert set(out["station"]) == {"S1"}

    # And only dates 2..4
    assert out["date"].min() == pd.Timestamp("2020-01-02")
    assert out["date"].max() == pd.Timestamp("2020-01-04")
    assert out.shape[0] == 3

    s1 = out.sort_values("date")
    # Day 3 had observation; 2 and 4 were missing
    assert set(s1["source"]) <= {"observed", "imputed"}
    assert pd.Timestamp("2020-01-03") in s1.loc[s1["source"] == "observed", "date"].tolist()
    # At least one imputed day in that window
    assert (s1["source"] == "imputed").any()


def test_impute_dataset_neighbor_map_controls_training():
    """
    A custom neighbor_map can prevent or enable training for a station.

    Case 1: A has no neighbors and include_target_pct=0.0
            -> no training pool, so station A is skipped (no rows).

    Case 2: A uses B as neighbor with include_target_pct=0.0
            -> training pool contains B only, so A can be imputed.
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

    # Case 1: A has no neighbors -> skipped
    out_empty = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        prefix=["A"],  # only try to impute A
        neighbor_map={"A": [], "B": []},
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )
    assert out_empty.empty

    # Case 2: A uses B as neighbor -> can be imputed
    out_nb = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="value",
        prefix=["A"],
        neighbor_map={"A": ["B"], "B": ["A"]},
        include_target_pct=0.0,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    # Now we must get a full 3-day series for A, all imputed
    assert not out_nb.empty
    assert set(out_nb["station"]) == {"A"}
    assert out_nb.shape[0] == 3
    assert set(out_nb["source"]) == {"imputed"}
    assert out_nb["value"].notna().all()
