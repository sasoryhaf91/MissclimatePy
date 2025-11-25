# tests/test_impute.py
import numpy as np
import pandas as pd
import pytest

from missclimatepy.impute import impute_dataset


def _make_simple_df() -> pd.DataFrame:
    """
    Small toy dataset with two stations, a few days, and one missing value.

    Columns:
        station, date, lat, lon, alt, tmin
    """
    dates = pd.date_range("2020-01-01", periods=4, freq="D")

    df = pd.DataFrame(
        {
            "station": ["A"] * 4 + ["B"] * 4,
            "date": list(dates) * 2,
            "lat": [10.0] * 4 + [11.0] * 4,
            "lon": [20.0] * 4 + [21.0] * 4,
            "alt": [100.0] * 4 + [200.0] * 4,
            # One missing value for station A on the 3rd day
            "tmin": [5.0, 6.0, np.nan, 8.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    return df


def test_impute_returns_expected_columns_and_fills_gaps():
    df = _make_simple_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        # extend window by one extra day to test continuous grid
        start="2020-01-01",
        end="2020-01-05",
        k_neighbors=None,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        include_target_pct=None,
        show_progress=False,
    )

    # Expected columns (canonical output schema)
    expected_cols = ["station", "date", "latitude", "longitude", "altitude", "tmin", "source"]
    assert list(out.columns) == expected_cols

    # Two stations, 5 days each → 10 rows total
    assert len(out) == 2 * 5

    # No NaNs in the output target
    assert out["tmin"].isna().sum() == 0

    # source must be either "observed" or "imputed"
    assert set(out["source"].unique()).issubset({"observed", "imputed"})

    # Merge back with original to check that observed values are preserved
    merged = out.merge(
        df[["station", "date", "tmin"]],
        on=["station", "date"],
        how="left",
        suffixes=("", "_orig"),
    )

    # Rows that had an original observation should remain unchanged
    mask_obs = ~merged["tmin_orig"].isna()
    assert (merged.loc[mask_obs, "tmin"] == merged.loc[mask_obs, "tmin_orig"]).all()
    assert (merged.loc[mask_obs, "source"] == "observed").all()

    # Rows that were originally missing (or outside input range) must be imputed
    mask_miss = merged["tmin_orig"].isna()
    assert (merged.loc[mask_miss, "source"] == "imputed").all()
    assert merged.loc[mask_miss, "tmin"].notna().all()


def test_impute_respects_min_station_rows_filters_stations():
    """
    Stations with fewer than min_station_rows observed values in the window
    should be skipped entirely in the output.
    """
    df = _make_simple_df()

    # Make station B have only 1 observed row (others NaN)
    mask_b = df["station"] == "B"
    df.loc[mask_b, "tmin"] = np.nan
    df.loc[(mask_b) & (df["date"] == pd.Timestamp("2020-01-01")), "tmin"] = 7.0

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start="2020-01-01",
        end="2020-01-04",
        k_neighbors=None,
        min_station_rows=2,  # B should be filtered out
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        include_target_pct=None,
        show_progress=False,
    )

    # Only station A should remain
    assert set(out["station"].unique()) == {"A"}

    # Station A should still have a complete grid over the 4 days
    assert len(out) == 4
    assert out["date"].nunique() == 4


def test_impute_LOSO_like_mode_include_target_zero_runs_and_marks_imputed():
    """
    With include_target_pct=0.0 the model should not see the local history,
    but all gaps must still be filled and source must correctly mark imputed rows.
    """
    df = _make_simple_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start="2020-01-01",
        end="2020-01-04",
        k_neighbors=None,
        model_kind="rf",
        model_params={"n_estimators": 10, "random_state": 0, "n_jobs": 1},
        include_target_pct=0.0,
        include_target_seed=123,
        show_progress=False,
    )

    # Still 2 stations × 4 days
    assert len(out) == 8

    # No NaNs in the output target
    assert out["tmin"].isna().sum() == 0

    # The specific missing value in station A on 2020-01-03 must be imputed
    mask_gap = (out["station"] == "A") & (out["date"] == pd.Timestamp("2020-01-03"))
    assert mask_gap.sum() == 1
    row = out.loc[mask_gap].iloc[0]
    assert row["source"] == "imputed"
    assert not np.isnan(row["tmin"])


def test_impute_uses_neighbor_map_when_k_neighbors_specified():
    """
    Basic smoke test to ensure the neighbor-based code path runs without error.
    """
    df = _make_simple_df()

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start="2020-01-01",
        end="2020-01-04",
        k_neighbors=1,  # force neighbor-map path
        model_kind="rf",
        model_params={"n_estimators": 5, "random_state": 0, "n_jobs": 1},
        include_target_pct=None,
        show_progress=False,
    )

    # Must have same schema and no NaNs
    expected_cols = ["station", "date", "latitude", "longitude", "altitude", "tmin", "source"]
    assert list(out.columns) == expected_cols
    assert out["tmin"].isna().sum() == 0


def test_impute_can_save_results_to_disk(tmp_path: pytest.TempPathFactory):
    df = _make_simple_df()

    # Parquet, single file
    out_path_parquet = tmp_path / "imputed.parquet"
    _ = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start="2020-01-01",
        end="2020-01-04",
        k_neighbors=None,
        model_kind="rf",
        model_params={"n_estimators": 5, "random_state": 0, "n_jobs": 1},
        include_target_pct=None,
        save_path=str(out_path_parquet),
        save_format="auto",
        save_index=False,
        save_partitions=False,
        show_progress=False,
    )
    assert out_path_parquet.exists()

    # CSV, multiple partitions (one per station)
    out_path_csv = tmp_path / "imputed.csv"
    _ = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        start="2020-01-01",
        end="2020-01-04",
        k_neighbors=None,
        model_kind="rf",
        model_params={"n_estimators": 5, "random_state": 0, "n_jobs": 1},
        include_target_pct=None,
        save_path=str(out_path_csv),
        save_format="csv",
        save_index=False,
        save_partitions=True,
        show_progress=False,
    )

    # Expect at least one file with the _station=<ID> suffix
    station_files = list(tmp_path.glob("imputed_station=*.csv"))
    assert len(station_files) >= 1

