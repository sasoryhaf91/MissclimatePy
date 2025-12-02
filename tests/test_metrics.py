# SPDX-License-Identifier: MIT
"""
Tests for missclimatepy.metrics
"""

import numpy as np
import pandas as pd

from missclimatepy.metrics import (
    mae,
    rmse,
    r2,
    kge,
    compute_metrics,
    aggregate_and_compute,
)


# --------------------------------------------------------------------------- #
# Basic metrics: mae, rmse, r2
# --------------------------------------------------------------------------- #


def test_mae_rmse_r2_perfect_fit():
    y = [1.0, 2.0, 3.0, 4.0]
    yhat = [1.0, 2.0, 3.0, 4.0]

    assert mae(y, yhat) == 0.0
    assert rmse(y, yhat) == 0.0
    # R2 should be exactly 1 for non-constant series with perfect fit.
    assert r2(y, yhat) == 1.0


def test_mae_rmse_r2_with_nans_and_mismatch_handling():
    # Paired NaNs should be dropped before computing metrics
    y = [1.0, 2.0, np.nan, 4.0]
    yhat = [1.0, 3.0, np.nan, 5.0]

    # Clean pairs: (1,1), (2,3), (4,5)
    # MAE = (0 + 1 + 1)/3 = 2/3
    expected_mae = 2.0 / 3.0
    # RMSE = sqrt( (0^2 + 1^2 + 1^2)/3 ) = sqrt(2/3)
    expected_rmse = np.sqrt(2.0 / 3.0)

    assert np.isclose(mae(y, yhat), expected_mae)
    assert np.isclose(rmse(y, yhat), expected_rmse)

    # R2 we just check in a reasonable range and finite
    r2_val = r2(y, yhat)
    assert np.isfinite(r2_val)
    assert -1.0 <= r2_val <= 1.0

    # Shape mismatch should raise ValueError
    y_short = [1.0, 2.0]
    yhat_long = [1.0, 2.0, 3.0]
    try:
        mae(y_short, yhat_long)
    except ValueError:
        pass
    else:
        raise AssertionError("mae should fail on shape mismatch but did not.")


def test_r2_degenerate_cases_return_nan():
    # Constant y_true → variance zero → R2 = nan
    y = [3.0, 3.0, 3.0, 3.0]
    yhat = [1.0, 2.0, 3.0, 4.0]
    assert np.isnan(r2(y, yhat))

    # Single-point series → length < 2 → R2 = nan
    y = [1.0]
    yhat = [1.0]
    assert np.isnan(r2(y, yhat))


# --------------------------------------------------------------------------- #
# KGE
# --------------------------------------------------------------------------- #


def test_kge_perfect_fit_is_one():
    # Non-constant, perfectly predicted series → KGE ≈ 1
    y = [1.0, 2.0, 4.0, 3.0, 5.0]
    yhat = [1.0, 2.0, 4.0, 3.0, 5.0]

    val = kge(y, yhat)
    assert np.isclose(val, 1.0, atol=1e-12)


def test_kge_degenerate_returns_nan():
    # Constant y_true: std = 0, mean ≠ 0 → KGE should be NaN
    y = [2.0, 2.0, 2.0, 2.0]
    yhat = [1.0, 2.0, 3.0, 4.0]
    assert np.isnan(kge(y, yhat))

    # Zero-length after dropping NaNs → NaN
    y = [np.nan, np.nan]
    yhat = [1.0, 2.0]
    assert np.isnan(kge(y, yhat))


def test_kge_handles_nans_pairwise():
    # Only finite pairs should be used
    y = [1.0, np.nan, 3.0, 4.0]
    yhat = [1.0, 2.0, 3.0, np.nan]

    # Clean pairs: (1,1), (3,3) → perfect where defined → KGE ≈ 1
    val = kge(y, yhat)
    assert np.isclose(val, 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# compute_metrics
# --------------------------------------------------------------------------- #


def test_compute_metrics_includes_kge_by_default():
    y = [1.0, 2.0, 3.0]
    yhat = [1.1, 1.9, 3.1]

    res = compute_metrics(y, yhat)
    assert set(res.keys()) == {"MAE", "RMSE", "R2", "KGE"}
    for v in res.values():
        assert isinstance(v, float)


def test_compute_metrics_without_kge():
    y = [1.0, 2.0, 3.0]
    yhat = [1.1, 1.9, 3.1]

    res = compute_metrics(y, yhat, include_kge=False)
    assert set(res.keys()) == {"MAE", "RMSE", "R2"}


# --------------------------------------------------------------------------- #
# aggregate_and_compute
# --------------------------------------------------------------------------- #


def test_aggregate_and_compute_monthly_sum():
    # Build exactly 2 months of daily data (Jan + Feb 2020)
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    # Observed: 1,2,3,...,60
    y = np.arange(1, 61, dtype=float)
    # Predicted = observed + 1
    yhat = y + 1.0

    df = pd.DataFrame(
        {
            "date": dates,
            "y_obs": y,
            "y_mod": yhat,
        }
    )

    metrics, agg_df = aggregate_and_compute(
        df,
        date_col="date",
        y_col="y_obs",
        yhat_col="y_mod",
        freq="M",
        agg="sum",
        include_kge=True,
    )

    # We should get 2 aggregated rows (Jan, Feb)
    assert agg_df.shape[0] == 2

    # Since yhat = y + 1, the aggregated sums differ by a constant offset:
    # sum(yhat) - sum(y) = number_of_points_in_window * 1
    # Jan has 31 days, Feb 2020 has 29 days
    diff = (agg_df["y_mod"] - agg_df["y_obs"]).to_numpy()
    assert np.allclose(diff, np.array([31.0, 29.0]))

    # Metrics dict has all four keys
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2", "KGE"}
    # R2 may not be 1 but should be finite
    assert np.isfinite(metrics["R2"])


def test_aggregate_and_compute_freq_alias_and_mean():
    # Simple yearly series with small noise
    dates = pd.date_range("2019-01-01", periods=730, freq="D")  # 2019 & 2020
    y = np.sin(np.linspace(0, 10, 730))
    yhat = y + 0.1  # small bias

    df = pd.DataFrame(
        {
            "date": dates,
            "y_obs": y,
            "y_mod": yhat,
        }
    )

    # Use "Y" alias, which should be normalized to "YS" (if implemented)
    metrics, agg_df = aggregate_and_compute(
        df,
        date_col="date",
        y_col="y_obs",
        yhat_col="y_mod",
        freq="Y",
        agg="mean",
        include_kge=False,
    )

    # Two aggregated years
    assert agg_df.shape[0] == 2
    # Metrics only MAE/RMSE/R2
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2"}


def test_aggregate_and_compute_empty_input_returns_nan_metrics():
    df = pd.DataFrame(columns=["date", "y_obs", "y_mod"])

    metrics, agg_df = aggregate_and_compute(
        df,
        date_col="date",
        y_col="y_obs",
        yhat_col="y_mod",
        freq="M",
        agg="sum",
        include_kge=True,
    )

    # agg_df remains empty
    assert agg_df.empty
    # All metrics should be NaN
    assert set(metrics.keys()) == {"MAE", "RMSE", "R2", "KGE"}
    for v in metrics.values():
        assert np.isnan(v)


def test_aggregate_and_compute_invalid_agg_raises():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "y_obs": [1, 2, 3, 4, 5],
            "y_mod": [1, 2, 3, 4, 5],
        }
    )

    try:
        aggregate_and_compute(
            df,
            date_col="date",
            y_col="y_obs",
            yhat_col="y_mod",
            freq="M",
            agg="invalid_agg",
            include_kge=True,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("aggregate_and_compute should raise on invalid agg but did not.")