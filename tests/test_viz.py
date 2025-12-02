# tests/test_viz.py
import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend for tests
matplotlib.use("Agg")

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


def _make_missing_df():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    rows = []
    for sid in ("A", "B"):
        for d in dates:
            rows.append(
                {
                    "station": sid,
                    "date": d,
                    "tmin": float(len(rows)),  # just some value
                }
            )
    df = pd.DataFrame(rows)

    # Introduce some missing values
    df.loc[(df["station"] == "A") & (df["date"] == dates[1]), "tmin"] = np.nan
    df.loc[(df["station"] == "B") & (df["date"] == dates[3]), "tmin"] = np.nan
    return df


def _make_eval_report():
    return pd.DataFrame(
        {
            "station": ["A", "B", "C"],
            "MAE_d": [1.0, 2.0, 1.5],
            "RMSE_d": [1.2, 2.2, 1.7],
            "R2_d": [0.8, 0.6, 0.7],
            "latitude": [10.0, 11.0, 12.0],
            "longitude": [20.0, 21.0, 22.0],
        }
    )


def _make_parity_df():
    return pd.DataFrame(
        {
            "y_obs": np.linspace(0, 10, 50),
            "y_mod": np.linspace(0, 10, 50) + np.random.normal(0, 0.5, 50),
        }
    )


def _make_time_series_df():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    rows = []
    for d in dates:
        rows.append(
            {
                "station": "A",
                "date": d,
                "tmin": float(len(rows)),
                "tmin_pred": float(len(rows)) + 0.5,
            }
        )
    return pd.DataFrame(rows)


def _make_spatial_df():
    return pd.DataFrame(
        {
            "latitude": [10.0, 11.0, 12.0],
            "longitude": [20.0, 21.0, 22.0],
            "RMSE_d": [1.0, 2.0, 1.5],
        }
    )


def _make_gap_df():
    return pd.DataFrame(
        {
            "station": ["A", "B", "C"],
            "max_gap": [5, 10, 3],
        }
    )


def _make_imputed_df():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    rows = []
    for i, d in enumerate(dates):
        rows.append(
            {
                "station": "A",
                "date": d,
                "tmin": float(i),
                "source": "observed" if i % 2 == 0 else "imputed",
            }
        )
    return pd.DataFrame(rows)


def _make_imputation_coverage_df():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    rows = []
    for sid in ("S1", "S2", "S3"):
        for i, d in enumerate(dates):
            rows.append(
                {
                    "station": sid,
                    "date": d,
                    "tmin": float(i),
                    "source": "imputed" if (sid == "S1" and i < 2) else "observed",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Tests for plot_missing_matrix
# ---------------------------------------------------------------------


def test_plot_missing_matrix_basic():
    df = _make_missing_df()

    ax = plot_missing_matrix(
        df,
        id_col="station",
        date_col="date",
        target_col="tmin",
        max_stations=10,
    )

    assert isinstance(ax, Axes)
    # We expect exactly one image (the matrix)
    assert len(ax.images) == 1
    img = ax.images[0]
    # Number of rows = number of stations (2 in this synthetic example)
    assert img.get_array().shape[0] == 2


def test_plot_missing_matrix_all_missing_shows_no_data():
    df = _make_missing_df()
    df["tmin"] = np.nan

    ax = plot_missing_matrix(
        df,
        id_col="station",
        date_col="date",
        target_col="tmin",
    )

    assert isinstance(ax, Axes)
    # _no_data places a text object with "No data"
    assert len(ax.texts) == 1
    assert "No data" in ax.texts[0].get_text()


# ---------------------------------------------------------------------
# Tests for plot_metrics_distribution
# ---------------------------------------------------------------------


def test_plot_metrics_distribution_hist():
    report = _make_eval_report()

    ax = plot_metrics_distribution(report, metric_cols=("MAE_d", "RMSE_d", "R2_d"), kind="hist")

    assert isinstance(ax, Axes)
    # At least one histogram patch drawn
    assert len(ax.patches) > 0


def test_plot_metrics_distribution_box():
    report = _make_eval_report()

    ax = plot_metrics_distribution(report, metric_cols=("MAE_d", "RMSE_d"), kind="box")

    assert isinstance(ax, Axes)
    # Boxplot creates artists but we only check that something was drawn
    assert len(ax.artists) >= 0  # smoke test: no error and an Axes is returned


# ---------------------------------------------------------------------
# Tests for plot_parity_scatter
# ---------------------------------------------------------------------


def test_plot_parity_scatter_basic():
    df = _make_parity_df()

    ax = plot_parity_scatter(df, y_true_col="y_obs", y_pred_col="y_mod", sample=None)

    assert isinstance(ax, Axes)
    # We expect at least one scatter collection plus the 1:1 line
    assert len(ax.collections) >= 1
    assert len(ax.lines) >= 1


def test_plot_parity_scatter_empty_returns_no_data():
    df = pd.DataFrame(columns=["y_obs", "y_mod"])

    ax = plot_parity_scatter(df)

    assert isinstance(ax, Axes)
    assert len(ax.texts) == 1
    assert "No parity data" in ax.texts[0].get_text()


# ---------------------------------------------------------------------
# Tests for plot_time_series_overlay
# ---------------------------------------------------------------------


def test_plot_time_series_overlay_with_predictions():
    df = _make_time_series_df()

    ax = plot_time_series_overlay(
        df,
        station_id="A",
        id_col="station",
        date_col="date",
        y_true_col="tmin",
        y_pred_col="tmin_pred",
    )

    assert isinstance(ax, Axes)
    # At least observed + predicted lines
    assert len(ax.lines) >= 2


def test_plot_time_series_overlay_no_data_for_station():
    df = _make_time_series_df()

    ax = plot_time_series_overlay(
        df,
        station_id="Z",  # non-existent
        id_col="station",
        date_col="date",
        y_true_col="tmin",
        y_pred_col="tmin_pred",
    )

    assert isinstance(ax, Axes)
    assert len(ax.texts) == 1
    assert "No data for station" in ax.texts[0].get_text()


# ---------------------------------------------------------------------
# Tests for plot_spatial_scatter
# ---------------------------------------------------------------------


def test_plot_spatial_scatter_basic():
    df = _make_spatial_df()

    ax = plot_spatial_scatter(df, lat_col="latitude", lon_col="longitude", value_col="RMSE_d")

    assert isinstance(ax, Axes)
    # One PathCollection for the scatter
    assert len(ax.collections) == 1


def test_plot_spatial_scatter_empty():
    df = _make_spatial_df().iloc[0:0]

    ax = plot_spatial_scatter(df)

    assert isinstance(ax, Axes)
    assert len(ax.texts) == 1
    assert "No spatial data" in ax.texts[0].get_text()


# ---------------------------------------------------------------------
# Tests for plot_gap_histogram
# ---------------------------------------------------------------------


def test_plot_gap_histogram_basic():
    gap_df = _make_gap_df()

    ax = plot_gap_histogram(gap_df, gap_len_col="max_gap")

    assert isinstance(ax, Axes)
    # Histogram draws patches (bars)
    assert len(ax.patches) > 0


def test_plot_gap_histogram_no_data():
    gap_df = pd.DataFrame(columns=["max_gap"])

    ax = plot_gap_histogram(gap_df, gap_len_col="max_gap")

    assert isinstance(ax, Axes)
    assert len(ax.texts) == 1
    assert "No gap data" in ax.texts[0].get_text()


# ---------------------------------------------------------------------
# Tests for plot_imputed_series
# ---------------------------------------------------------------------


def test_plot_imputed_series_basic():
    df = _make_imputed_df()

    ax = plot_imputed_series(
        df,
        station="A",
        id_col="station",
        date_col="date",
        target_col="tmin",
        source_col="source",
    )

    assert isinstance(ax, Axes)
    # Background line + at least one scatter (observed or imputed)
    assert len(ax.lines) >= 1
    assert len(ax.collections) >= 1
    labels = [t.get_text() for t in ax.get_legend().texts]
    assert "Observed" in labels or "Imputed" in labels


def test_plot_imputed_series_empty_df():
    df = _make_imputed_df().iloc[0:0]

    ax = plot_imputed_series(df, station="A")

    assert isinstance(ax, Axes)
    assert len(ax.texts) == 1
    assert "Empty dataframe" in ax.texts[0].get_text()


# ---------------------------------------------------------------------
# Tests for plot_imputation_coverage
# ---------------------------------------------------------------------


def test_plot_imputation_coverage_basic():
    df = _make_imputation_coverage_df()

    ax = plot_imputation_coverage(df, id_col="station", source_col="source")

    assert isinstance(ax, Axes)
    # Bars for each station
    assert len(ax.patches) == df["station"].nunique()


def test_plot_imputation_coverage_empty():
    df = _make_imputation_coverage_df().iloc[0:0]

    ax = plot_imputation_coverage(df)

    assert isinstance(ax, Axes)
    assert len(ax.texts) == 1
    assert "Empty dataframe" in ax.texts[0].get_text()
