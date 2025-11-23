import numpy as np
import pandas as pd
import pytest

from missclimatepy.metrics import (
    compute_kge,
    compute_mcm_baseline,
    multiscale_metrics,
)


# ---------------------------------------------------------
# KGE tests
# ---------------------------------------------------------

def test_compute_kge_perfect_match():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])

    kge = compute_kge(y_true, y_pred)

    assert kge == pytest.approx(1.0, abs=1e-12)


def test_compute_kge_with_bias():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([2, 3, 4])  # +1 bias

    kge = compute_kge(y_true, y_pred)
    assert kge < 1.0


def test_compute_kge_handles_nan():
    y_true = np.array([1, 2, np.nan, 4])
    y_pred = np.array([1, 2, 3, 4])

    kge = compute_kge(y_true, y_pred)
    assert np.isfinite(kge)


# ---------------------------------------------------------
# MCM baseline tests
# ---------------------------------------------------------

def test_compute_mcm_baseline_global():
    dates = pd.date_range("2000-01-01", periods=4, freq="D")
    values = pd.Series([1.0, 2.0, 3.0, 4.0])

    baseline = compute_mcm_baseline(
        dates=dates,
        values=values,
        mode="global",
        min_samples=1,
    )

    expected = values.mean()

    assert isinstance(baseline, pd.Series)
    assert len(baseline) == len(values)

    # CORRECCIÓN CRÍTICA
    assert baseline.values == pytest.approx(
        expected * np.ones(len(values)),
        rel=1e-6,
        abs=1e-6,
    )


def test_compute_mcm_baseline_doy_mode():
    dates = pd.to_datetime(
        ["2000-01-01", "1999-01-01", "2000-01-02", "1999-01-02"]
    )
    values = pd.Series([1.0, 3.0, 2.0, 4.0])

    baseline = compute_mcm_baseline(
        dates=dates,
        values=values,
        mode="doy",
        min_samples=1,
    )

    assert len(baseline) == 4
    assert baseline.iloc[0] == pytest.approx((1.0 + 3.0) / 2)
    assert baseline.iloc[2] == pytest.approx((2.0 + 4.0) / 2)


# ---------------------------------------------------------
# Multiscale metrics tests
# ---------------------------------------------------------

def test_multiscale_metrics_perfect_match_gives_zero_rmse_and_high_r2():
    dates = pd.date_range("2000-01-01", periods=12, freq="D")
    y_true = np.linspace(0, 11, 12)
    df = pd.DataFrame(
        {
            "date": dates,
            "y_true": y_true,
            "y_pred": y_true,  # perfect match
        }
    )

    ms = multiscale_metrics(
        df,
        date_col="date",
        y_col="y_true",
        yhat_col="y_pred",
        monthly_agg="sum",
        annual_agg="sum",
    )

    assert set(ms.keys()) == {"daily", "monthly", "annual"}

    for scale in ["daily", "monthly", "annual"]:
        assert ms[scale]["MAE"] == pytest.approx(0.0, abs=1e-12)
        assert ms[scale]["RMSE"] == pytest.approx(0.0, abs=1e-12)
        assert ms[scale]["R2"] == pytest.approx(1.0, rel=1e-6, abs=1e-6)
        assert ms[scale]["KGE"] == pytest.approx(1.0, rel=1e-6, abs=1e-6)


def test_multiscale_metrics_with_noise_increases_rmse():
    dates = pd.date_range("2000-01-01", periods=20, freq="D")

    y_true = np.sin(np.linspace(0, 4 * np.pi, 20))
    y_pred = y_true + np.random.normal(0, 0.1, size=20)

    df = pd.DataFrame(
        {"date": dates, "y_true": y_true, "y_pred": y_pred}
    )

    ms = multiscale_metrics(
        df,
        date_col="date",
        y_col="y_true",
        yhat_col="y_pred",
    )

    assert ms["daily"]["RMSE"] > 0.0


def test_multiscale_metrics_handles_nans_gracefully():
    dates = pd.date_range("2000-01-01", periods=6, freq="D")

    y_true = np.array([1, 2, np.nan, 4, 5, 6])
    y_pred = np.array([1, 2, 3, np.nan, 5, 6])

    df = pd.DataFrame(
        {"date": dates, "y_true": y_true, "y_pred": y_pred}
    )

    ms = multiscale_metrics(
        df,
        date_col="date",
        y_col="y_true",
        yhat_col="y_pred",
    )

    assert "daily" in ms
    assert np.isfinite(ms["daily"]["RMSE"])
    assert np.isfinite(ms["daily"]["MAE"])

