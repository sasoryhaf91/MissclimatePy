# tests/test_evaluate.py

import numpy as np
import pandas as pd

from missclimatepy.evaluate import evaluate_stations, RFParams
from missclimatepy.impute import impute_dataset


# ----------------------------------------------------------------------
# Helper: generate a minimal toy dataset
# ----------------------------------------------------------------------
def _make_df(n_st=4, n_days=40, missing_frac=0.20, seed=0):
    rng = np.random.default_rng(seed)
    stations = [f"S{i:02d}" for i in range(n_st)]
    dates = pd.date_range("2000-01-01", periods=n_days, freq="D")

    rows = []
    for st in stations:
        vals = rng.normal(loc=15, scale=3, size=n_days)
        miss_mask = rng.random(n_days) < missing_frac
        vals[miss_mask] = np.nan
        for d, v in zip(dates, vals):
            rows.append(
                dict(
                    station=st,
                    date=d,
                    lat=19 + rng.random() * 0.2,
                    lon=-99 + rng.random() * 0.2,
                    alt=2400 + rng.random() * 50,
                    tmin=v,
                )
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 1. Basic smoke test
# ----------------------------------------------------------------------
def test_evaluate_smoke():
    df = _make_df()

    out = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        model_kind="rf",
        rf_params=RFParams(n_estimators=10, random_state=1),
        show_progress=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert "station" in out.columns
    assert "RMSE_d" in out.columns
    assert len(out) > 0


# ----------------------------------------------------------------------
# 2. Test evaluation with model_kind = mcm
# ----------------------------------------------------------------------
def test_evaluate_with_mcm_model():
    df = _make_df(missing_frac=0.5)

    out = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        model_kind="mcm",
        show_progress=False,
    )

    assert "RMSE_d" in out.columns
    assert len(out) == df["station"].nunique()


# ----------------------------------------------------------------------
# 3. Test evaluation with k_neighbors small
# ----------------------------------------------------------------------
def test_evaluate_with_knn_neighbors():
    df = _make_df()

    out = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        k_neighbors=2,
        model_kind="rf",
        rf_params={"n_estimators": 5, "random_state": 0},
        show_progress=False,
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == df["station"].nunique()


# ----------------------------------------------------------------------
# 4. include_target_pct should not break evaluation
# ----------------------------------------------------------------------
def test_evaluate_with_target_leakage():
    df = _make_df()

    out = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        include_target_pct=50.0,
        model_kind="rf",
        rf_params={"n_estimators": 10, "random_state": 0},
        show_progress=False,
    )

    assert "RMSE_d" in out.columns
    assert len(out) > 0


# ----------------------------------------------------------------------
# 5. Evaluation should work even if all values at a station are NaN
# ----------------------------------------------------------------------
def test_evaluate_station_with_no_observed_values():
    df = _make_df()
    df.loc[df["station"] == "S00", "tmin"] = np.nan  # remove all values of one station

    out = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        model_kind="mcm",      # safest backend
        show_progress=False,
    )

    assert len(out) == df["station"].nunique()
    assert out["RMSE_d"].notna().all()


# ----------------------------------------------------------------------
# 6. Test consistency: evaluate -> impute should not crash
# ----------------------------------------------------------------------
def test_evaluate_and_impute_consistency():
    df = _make_df()

    # Evaluate first
    ev = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        model_kind="rf",
        rf_params={"n_estimators": 15, "random_state": 0},
        show_progress=False,
    )

    assert "RMSE_d" in ev.columns

    # Then impute with same model
    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        model_kind="rf",
        rf_params={"n_estimators": 15, "random_state": 0},
        show_progress=False,
    )

    # Output should have correct columns
    assert set(["station", "date", "lat", "lon", "alt", "tmin", "source"]).issubset(out.columns)
    assert "imputed" in out["source"].unique()


# ----------------------------------------------------------------------
# 7. Test evaluation with model_kind = linear
# ----------------------------------------------------------------------
def test_evaluate_linear_regression():
    df = _make_df()

    out = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        model_kind="linear",
        show_progress=False,
    )

    assert len(out) == df["station"].nunique()
    assert "RMSE_d" in out.columns


# ----------------------------------------------------------------------
# 8. Sorting and saving options shouldn't break evaluation
# ----------------------------------------------------------------------
def test_evaluate_sorting():
    df = _make_df()

    out = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="lat",
        lon_col="lon",
        alt_col="alt",
        target_col="tmin",
        model_kind="rf",
        order_by=("RMSE_d", True),
        show_progress=False,
    )

    assert out.iloc[0]["RMSE_d"] <= out.iloc[-1]["RMSE_d"]
