# tests/test_viz.py
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from missclimatepy.viz import (
    plot_missing_matrix,
    plot_metrics_distribution,
    plot_parity_scatter,
    plot_time_series_overlay,
    plot_spatial_scatter,
    plot_gap_histogram,
    plot_imputed_series,
    plot_imputation_coverage,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _make_missing_df():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    data = []

    # S1: full coverage
    for d in dates:
        data.append({"station": "S1", "date": d, "value": 1.0})

    # S2: partial coverage
    vals = [1.0, np.nan, 2.0, np.nan, np.nan]
    for d, v in zip(dates, vals):
        data.append({"station": "S2", "date": d, "value": v})

    return pd.DataFrame(data)


def _make_eval_report_df():
    return pd.DataFrame(
        {
            "station": ["S1", "S2", "S3"],
            "MAE_d": [0.1, 0.2, 0.15],
            "RMSE_d": [0.2, 0.3, 0.25],
            "R2_d": [0.9, 0.8, 0.85],
            "latitude": [19.0, 20.0, 21.0],
            "longitude": [-99.0, -98.5, -98.0],
        }
    )


def _make_parity_df():
    rng = np.random.RandomState(0)
    y_true = rng.normal(loc=10.0, scale=2.0, size=200)
    y_pred = y_true + rng.normal(loc=0.0, scale=0.5, size=200)
    return pd.DataFrame({"y_obs": y_true, "y_mod": y_pred})


def _make_time_series_df():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "station": ["S1"] * len(dates) + ["S2"] * len(dates),
            "date": list(dates) + list(dates),
            "obs": np.arange(len(dates)).tolist() + np.arange(len(dates)).tolist(),
            "pred": (np.arange(len(dates)) + 0.5).tolist()
            + (np.arange(len(dates)) + 0.3).tolist(),
        }
    )


def _make_gap_df():
    return pd.DataFrame(
        {
            "station": ["S1", "S2", "S3"],
            "n_gaps": [0, 2, 1],
            "mean_gap": [0.0, 3.5, 2.0],
            "max_gap": [0, 7, 3],
        }
    )


def _make_imputed_df():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []

    # S1: mixture of observed and imputed
    src_s1 = ["observed", "imputed", "observed", "imputed", "observed", "imputed"]
    vals_s1 = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for d, s, v in zip(dates, src_s1, vals_s1):
        rows.append({"station": "S1", "date": d, "tmin": v, "source": s})

    # S2: mostly observed
    src_s2 = ["observed"] * len(dates)
    vals_s2 = [2.0 + 0.1 * i for i in range(len(dates))]
    for d, s, v in zip(dates, src_s2, vals_s2):
        rows.append({"station": "S2", "date": d, "tmin": v, "source": s})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Tests: plot_missing_matrix
# ---------------------------------------------------------------------


def test_plot_missing_matrix_basic():
    df = _make_missing_df()
    fig, ax = plt.subplots()

    out_ax = plot_missing_matrix(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
        ax=ax,
    )

    assert isinstance(out_ax, Axes)
    # Should have y-ticks for both stations
    assert len(out_ax.get_yticks()) >= 2


def test_plot_missing_matrix_handles_empty():
    df = pd.DataFrame(columns=["station", "date", "value"])
    fig, ax = plt.subplots()

    out_ax = plot_missing_matrix(
        df,
        id_col="station",
        date_col="date",
        target_col="value",
        ax=ax,
    )

    assert isinstance(out_ax, Axes)  # just no crash


# ---------------------------------------------------------------------
# Tests: plot_metrics_distribution
# ---------------------------------------------------------------------


def test_plot_metrics_distribution_hist_and_box():
    report = _make_eval_report_df()

    # Histogram
    fig1, ax1 = plt.subplots()
    ax_hist = plot_metrics_distribution(
        report,
        metric_cols=("MAE_d", "RMSE_d", "R2_d"),
        kind="hist",
        ax=ax1,
    )
    assert isinstance(ax_hist, Axes)

    # Boxplot
    fig2, ax2 = plt.subplots()
    ax_box = plot_metrics_distribution(
        report,
        metric_cols=("MAE_d", "RMSE_d"),
        kind="box",
        ax=ax2,
    )
    assert isinstance(ax_box, Axes)


def test_plot_metrics_distribution_handles_empty():
    report = pd.DataFrame()
    fig, ax = plt.subplots()
    out_ax = plot_metrics_distribution(report, ax=ax)
    assert isinstance(out_ax, Axes)


# ---------------------------------------------------------------------
# Tests: plot_parity_scatter
# ---------------------------------------------------------------------


def test_plot_parity_scatter_basic():
    df = _make_parity_df()
    fig, ax = plt.subplots()

    out_ax = plot_parity_scatter(df, y_true_col="y_obs", y_pred_col="y_mod", ax=ax)
    assert isinstance(out_ax, Axes)


def test_plot_parity_scatter_handles_empty():
    df = pd.DataFrame(columns=["y_obs", "y_mod"])
    fig, ax = plt.subplots()
    out_ax = plot_parity_scatter(df, ax=ax)
    assert isinstance(out_ax, Axes)


# ---------------------------------------------------------------------
# Tests: plot_time_series_overlay
# ---------------------------------------------------------------------


def test_plot_time_series_overlay_with_and_without_pred():
    df = _make_time_series_df()

    # With predictions
    fig1, ax1 = plt.subplots()
    ax_with = plot_time_series_overlay(
        df,
        station_id="S1",
        id_col="station",
        date_col="date",
        y_true_col="obs",
        y_pred_col="pred",
        ax=ax1,
    )
    assert isinstance(ax_with, Axes)

    # Without specifying y_pred_col
    fig2, ax2 = plt.subplots()
    ax_without = plot_time_series_overlay(
        df,
        station_id="S2",
        id_col="station",
        date_col="date",
        y_true_col="obs",
        ax=ax2,
    )
    assert isinstance(ax_without, Axes)


def test_plot_time_series_overlay_handles_missing_station():
    df = _make_time_series_df()
    fig, ax = plt.subplots()

    out_ax = plot_time_series_overlay(
        df,
        station_id="NON_EXISTENT",
        id_col="station",
        date_col="date",
        y_true_col="obs",
        ax=ax,
    )
    assert isinstance(out_ax, Axes)


# ---------------------------------------------------------------------
# Tests: plot_spatial_scatter
# ---------------------------------------------------------------------


def test_plot_spatial_scatter_basic():
    df = _make_eval_report_df()
    fig, ax = plt.subplots()

    out_ax = plot_spatial_scatter(
        df,
        lat_col="latitude",
        lon_col="longitude",
        value_col="RMSE_d",
        ax=ax,
    )
    assert isinstance(out_ax, Axes)


def test_plot_spatial_scatter_handles_empty():
    df = pd.DataFrame(columns=["latitude", "longitude", "RMSE_d"])
    fig, ax = plt.subplots()
    out_ax = plot_spatial_scatter(df, ax=ax)
    assert isinstance(out_ax, Axes)


# ---------------------------------------------------------------------
# Tests: plot_gap_histogram
# ---------------------------------------------------------------------


def test_plot_gap_histogram_basic():
    gap_df = _make_gap_df()
    fig, ax = plt.subplots()

    out_ax = plot_gap_histogram(gap_df, gap_len_col="max_gap", ax=ax)
    assert isinstance(out_ax, Axes)


def test_plot_gap_histogram_handles_empty():
    gap_df = pd.DataFrame(columns=["station", "max_gap"])
    fig, ax = plt.subplots()
    out_ax = plot_gap_histogram(gap_df, gap_len_col="max_gap", ax=ax)
    assert isinstance(out_ax, Axes)


# ---------------------------------------------------------------------
# Tests: plot_imputed_series & plot_imputation_coverage
# ---------------------------------------------------------------------


def test_plot_imputed_series_basic():
    df = _make_imputed_df()
    ax = plot_imputed_series(
        df,
        station="S1",
        id_col="station",
        date_col="date",
        target_col="tmin",
        source_col="source",
    )
    assert isinstance(ax, Axes)


def test_plot_imputation_coverage_basic():
    df = _make_imputed_df()
    ax = plot_imputation_coverage(
        df,
        id_col="station",
        source_col="source",
        sort_by="imputed_ratio",
    )
    assert isinstance(ax, Axes)


def test_plot_imputation_coverage_handles_empty():
    df = pd.DataFrame(columns=["station", "source"])
    ax = plot_imputation_coverage(df)
    assert isinstance(ax, Axes)

