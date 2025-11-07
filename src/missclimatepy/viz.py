# src/missclimatepy/viz.py
# SPDX-License-Identifier: MIT
"""
Visualization utilities for MissclimatePy.

This module provides small, composable plotting helpers to explore:
- Missingness patterns (heatmap-like matrix).
- Distribution of evaluation metrics across stations.
- Observed vs. modeled parity.
- Station-level time series overlays (observed vs. predicted).
- Simple spatial scatter of performance by station coordinates.
- Gap length distributions (requires gap profiles from masking.gap_profile_by_station).
- Observed vs imputed series visualization (from impute_dataset output).
- Imputation coverage per station (share of imputed points).

Design principles
-----------------
- Minimal dependencies: matplotlib, numpy, pandas only.
- Safe defaults; plots render sensibly without additional styling.
- Stable return types: either an Axes, or (Figure, Axes) when we create the Figure.
- Column names are parameters to keep the code schema-agnostic.

Notes
-----
- All functions are robust to empty inputs and will annotate “No data” rather than failing.
- Colors/rcParams are *not* customized; callers can style externally as desired.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _ensure_ax(ax: Optional[Axes] = None, figsize: Tuple[float, float] = (8.0, 4.0)) -> Tuple[Figure, Axes, bool]:
    """
    Create a new Figure/Axes if `ax` is None. Return (fig, ax, created_flag).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.figure, ax, False


def _no_data(ax: Axes, message: str = "No data") -> Axes:
    """
    Render a centered 'No data' message on the provided axes.
    """
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def _coerce_datetime(s: pd.Series) -> pd.Series:
    """
    Best-effort datetime coercion without timezones (naive).
    """
    out = pd.to_datetime(s, errors="coerce")
    # Avoid deprecated is_datetime64tz_dtype; check dtype class instead
    if isinstance(out.dtype, pd.DatetimeTZDtype):
        out = out.dt.tz_localize(None)
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def plot_missing_matrix(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
    stations: Optional[Sequence[object]] = None,
    max_stations: int = 50,
    sort_by_missing: bool = True,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10.0, 6.0),
) -> Axes:
    """
    Visualize missingness as a binary matrix (stations × time).

    Each cell is 1 where the target is present and 0 where it is missing.

    Parameters
    ----------
    df : DataFrame
        Long-format table including id, date, and target columns.
    id_col, date_col, target_col : str
        Column names for station id, timestamp, and the target variable.
    stations : sequence or None
        Optional subset of station ids to display.
    max_stations : int
        Maximum number of stations to plot. If more are available, the top stations
        (by missingness order) are selected.
    sort_by_missing : bool
        If True, order stations from most missing to least missing.
    ax : Axes or None
        Axes to draw on. If None, a new Figure/Axes is created.
    figsize : (float, float)
        Figure size used when `ax` is None.

    Returns
    -------
    Axes
        The axes with the matrix image.
    """
    fig, ax, created = _ensure_ax(ax, figsize)

    if df.empty or df[target_col].isna().all():
        return _no_data(ax, "No data (all missing)")

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = _coerce_datetime(work[date_col])
    work = work.dropna(subset=[date_col])

    # Select station subset if requested
    if stations is not None:
        stations_set = set(stations)
        work = work[work[id_col].isin(stations_set)]

    if work.empty:
        return _no_data(ax, "No data after station/date filtering")

    # Build pivot: rows = station, cols = date, values = present(1)/missing(0)
    work["present"] = (~work[target_col].isna()).astype(int)
    pivot = (
        work.pivot_table(index=id_col, columns=date_col, values="present", aggfunc="max", fill_value=0)
        .sort_index(axis=1)
    )

    if pivot.empty:
        return _no_data(ax, "No data to plot")

    # Sort stations by missingness (descending = more missing first)
    if sort_by_missing:
        order = pivot.sum(axis=1).sort_values(ascending=True).index  # fewer present → more missing
        pivot = pivot.loc[order]

    # Cap number of stations
    if pivot.shape[0] > max_stations:
        pivot = pivot.iloc[:max_stations]

    # Render
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Station")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str))

    # Reduce x tick clutter: show first, middle, last dates
    dates = pivot.columns.to_pydatetime()
    if dates.size > 0:
        xticks = [0, max(0, dates.size // 2), max(0, dates.size - 1)]
        xtick_labels = [dates[i].strftime("%Y-%m-%d") for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels, rotation=30, ha="right")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Present (1) / Missing (0)")
    return ax


def plot_metrics_distribution(
    report: pd.DataFrame,
    *,
    metric_cols: Sequence[str] = ("MAE_d", "RMSE_d", "R2_d"),
    kind: str = "hist",
    bins: int = 30,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (10.0, 4.0),
) -> Axes:
    """
    Plot distributions of evaluation metrics across stations.

    Parameters
    ----------
    report : DataFrame
        Output from evaluation (e.g., evaluate_stations). Must contain metric columns.
    metric_cols : sequence of str
        Column names of metrics to plot.
    kind : {"hist", "box"}
        Plot style: histogram or boxplot.
    bins : int
        Number of histogram bins (when kind == "hist").
    ax : Axes or None
        Axes to draw on. If None, create a new Figure/Axes.
    figsize : (float, float)
        Figure size used when `ax` is None.

    Returns
    -------
    Axes
    """
    fig, ax, created = _ensure_ax(ax, figsize)

    if report.empty:
        return _no_data(ax, "No metrics to plot")

    data = report[list(metric_cols)].select_dtypes(include=[np.number]).dropna(how="all")
    if data.empty:
        return _no_data(ax, "No numeric metrics found")

    if kind == "box":
        ax.boxplot([data[c].dropna().values for c in data.columns], labels=list(data.columns))
        ax.set_ylabel("Metric value")
    else:
        # Histogram overlay; draw sequentially
        for col in data.columns:
            vals = data[col].dropna().values
            if vals.size == 0:
                continue
            ax.hist(vals, bins=bins, alpha=0.6, label=col)
        ax.legend()
        ax.set_ylabel("Frequency")

    ax.set_title("Metric distribution across stations")
    ax.set_xlabel("Value")
    return ax


def plot_parity_scatter(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y_obs",
    y_pred_col: str = "y_mod",
    sample: Optional[int] = 10000,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (5.0, 5.0),
) -> Axes:
    """
    Parity scatter: observed vs. predicted with a 1:1 reference line.

    Parameters
    ----------
    df : DataFrame
        Table containing observed and modeled columns.
    y_true_col, y_pred_col : str
        Column names for observed and modeled values.
    sample : int or None
        Optional random subsample size to avoid overplotting. If None, plot all.
    ax : Axes or None
        Axes to draw on. If None, create a new Figure/Axes.
    figsize : (float, float)
        Figure size used when `ax` is None.

    Returns
    -------
    Axes
    """
    fig, ax, created = _ensure_ax(ax, figsize)

    if df.empty or y_true_col not in df.columns or y_pred_col not in df.columns:
        return _no_data(ax, "No parity data")

    work = df[[y_true_col, y_pred_col]].dropna()
    if work.empty:
        return _no_data(ax, "No parity data (after dropna)")

    if sample is not None and work.shape[0] > sample:
        work = work.sample(n=sample, random_state=42)

    x = work[y_true_col].to_numpy()
    y = work[y_pred_col].to_numpy()

    ax.scatter(x, y, s=8, alpha=0.6)
    # 1:1 line
    lo = np.nanmin([x.min(), y.min()])
    hi = np.nanmax([x.max(), y.max()])
    ax.plot([lo, hi], [lo, hi], linestyle="--")

    ax.set_xlabel("Observed")
    ax.set_ylabel("Modeled")
    ax.set_title("Observed vs. Modeled (parity)")
    return ax


def plot_time_series_overlay(
    df: pd.DataFrame,
    *,
    station_id: object,
    id_col: str,
    date_col: str,
    y_true_col: str,
    y_pred_col: Optional[str] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (12.0, 3.5),
) -> Axes:
    """
    Plot a single-station time series, optionally overlaying model predictions.

    Parameters
    ----------
    df : DataFrame
        Long-format table with station id, date, observed (and optionally modeled) values.
    station_id : object
        Station identifier to filter.
    id_col, date_col : str
        Column names for station id and timestamp.
    y_true_col : str
        Observed/ground-truth column.
    y_pred_col : str or None
        Optional model prediction column to overlay.
    ax : Axes or None
        Axes to draw on. If None, create a new Figure/Axes.
    figsize : (float, float)
        Figure size used when `ax` is None.

    Returns
    -------
    Axes
    """
    fig, ax, created = _ensure_ax(ax, figsize)

    if df.empty:
        return _no_data(ax, "No time series data")

    sub = df[df[id_col] == station_id].copy()
    if sub.empty:
        return _no_data(ax, f"No data for station {station_id}")

    sub[date_col] = _coerce_datetime(sub[date_col])
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)

    if sub.empty:
        return _no_data(ax, f"No dated data for station {station_id}")

    ax.plot(sub[date_col], sub[y_true_col], linewidth=1.2, label="Observed")
    if y_pred_col is not None and y_pred_col in sub.columns:
        ax.plot(sub[date_col], sub[y_pred_col], linewidth=1.0, linestyle="--", label="Modeled")

    ax.set_title(f"Station {station_id} – {y_true_col}")
    ax.set_xlabel("Date")
    ax.set_ylabel(y_true_col)
    ax.legend()
    return ax


def plot_spatial_scatter(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    value_col: str = "RMSE_d",
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (6.0, 5.0),
) -> Axes:
    """
    Simple lon/lat scatter colored by a performance value (e.g., RMSE_d).

    Parameters
    ----------
    df : DataFrame
        Table containing coordinates and a scalar metric.
    lat_col, lon_col : str
        Column names for latitude and longitude (degrees).
    value_col : str
        Column name of the value to color points by.
    ax : Axes or None
        Axes to draw on. If None, create a new Figure/Axes.
    figsize : (float, float)
        Figure size used when `ax` is None.

    Returns
    -------
    Axes
    """
    fig, ax, created = _ensure_ax(ax, figsize)

    if df.empty or lat_col not in df.columns or lon_col not in df.columns or value_col not in df.columns:
        return _no_data(ax, "No spatial data")

    dat = df[[lat_col, lon_col, value_col]].dropna()
    if dat.empty:
        return _no_data(ax, "No spatial data (after dropna)")

    sc = ax.scatter(dat[lon_col], dat[lat_col], c=dat[value_col], s=30)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial scatter colored by {value_col}")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(value_col)
    return ax


def plot_gap_histogram(
    gap_df: pd.DataFrame,
    *,
    gap_len_col: str = "max_gap",
    ax: Optional[Axes] = None,
    bins: int = 30,
    figsize: Tuple[float, float] = (8.0, 4.0),
) -> Axes:
    """
    Plot a histogram of gap lengths (e.g., longest missing run per station).

    Parameters
    ----------
    gap_df : DataFrame
        Output from `masking.gap_profile_by_station`, which typically includes
        columns like 'max_gap', 'mean_gap', etc.
    gap_len_col : str
        Column name representing gap lengths in days (or periods).
    ax : Axes or None
        Axes to draw on. If None, create a new Figure/Axes.
    bins : int
        Number of histogram bins.
    figsize : (float, float)
        Figure size used when `ax` is None.

    Returns
    -------
    Axes
    """
    fig, ax, created = _ensure_ax(ax, figsize)

    if gap_df.empty or gap_len_col not in gap_df.columns:
        return _no_data(ax, "No gap data")

    values = pd.to_numeric(gap_df[gap_len_col], errors="coerce").dropna().values
    if values.size == 0:
        return _no_data(ax, "No numeric gap data")

    ax.hist(values, bins=bins, alpha=0.8)
    ax.set_xlabel(f"{gap_len_col} (days)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Distribution of {gap_len_col}")
    return ax


# ---------------------------------------------------------------------
# New: Observed vs Imputed series (from imputation output)
# ---------------------------------------------------------------------
def plot_imputed_series(
    df: pd.DataFrame,
    *,
    station: str,
    id_col: str = "station",
    date_col: str = "date",
    target_col: str = "tmin",
    source_col: str = "source",
    start: str | None = None,
    end: str | None = None,
    title: str | None = None,
    figsize: Tuple[float, float] = (11.0, 4.0),
    alpha_line: float = 0.9,
    ms: int = 12,
) -> Axes:
    """
    Plot a single station's time series highlighting observed vs imputed points.
    Intended for the dataframe returned by `impute_dataset(...)`, which must
    include `source_col` ∈ {"observed","imputed"}.

    Parameters
    ----------
    df : DataFrame
        Output of `impute_dataset(...)` (must include `source_col`).
    station : str
        Station identifier to plot.
    id_col, date_col, target_col, source_col : str
        Column names. `source_col` should contain "observed" / "imputed".
    start, end : str or None
        Optional inclusive window to display.
    title : str or None
        Custom plot title; if None a default is constructed.
    figsize : (float, float)
        Figure size.
    alpha_line : float
        Alpha for the continuous background line.
    ms : int
        Marker size for observed/imputed points.

    Returns
    -------
    Axes
    """
    fig, ax, created = _ensure_ax(None, figsize)

    if df.empty:
        return _no_data(ax, "Empty dataframe: nothing to plot.")

    if date_col not in df.columns:
        return _no_data(ax, f"Column '{date_col}' not found.")

    # Filter single station and optional window
    sub = df[df[id_col] == station].copy()
    if sub.empty:
        return _no_data(ax, f"No rows for station '{station}'.")
    sub[date_col] = _coerce_datetime(sub[date_col])
    if start or end:
        lo = pd.to_datetime(start) if start else sub[date_col].min()
        hi = pd.to_datetime(end) if end else sub[date_col].max()
        sub = sub[(sub[date_col] >= lo) & (sub[date_col] <= hi)].copy()
        if sub.empty:
            return _no_data(ax, "No data in the requested window.")

    # Sort, split by source
    sub = sub.sort_values(date_col)
    obs = sub[sub[source_col] == "observed"]
    imp = sub[sub[source_col] == "imputed"]

    # Background line for continuity
    ax.plot(sub[date_col], sub[target_col], lw=1.2, alpha=alpha_line)

    # Scatters to highlight observed vs imputed
    if not obs.empty:
        ax.scatter(obs[date_col], obs[target_col], marker="o", s=ms, label="Observed", zorder=3)
    if not imp.empty:
        ax.scatter(imp[date_col], imp[target_col], marker="x", s=ms, label="Imputed", zorder=3)

    # Cosmetics
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=False)
    ax.set_title(title or f"{station} — {target_col} (observed vs imputed)")
    fig.tight_layout()
    return ax


def plot_imputation_coverage(
    df: pd.DataFrame,
    *,
    id_col: str = "station",
    source_col: str = "source",
    sort_by: str = "imputed_ratio",
    figsize: Tuple[float, float] = (10.0, 5.0),
) -> Axes:
    """
    Bar chart of imputation coverage per station (share of imputed points).

    Parameters
    ----------
    df : DataFrame
        Output of `impute_dataset(...)`.
    id_col, source_col : str
        Column names; `source_col` ∈ {"observed","imputed"}.
    sort_by : {"imputed_ratio","observed_ratio","station"}
        Sort criterion for bars.
    figsize : (float, float)
        Figure size.

    Returns
    -------
    Axes
    """
    fig, ax, created = _ensure_ax(None, figsize)

    if df.empty:
        return _no_data(ax, "Empty dataframe.")

    if source_col not in df.columns or id_col not in df.columns:
        return _no_data(ax, "Missing required columns for coverage plot.")

    # Ratios per station
    tab = (df
           .assign(is_imp=(df[source_col] == "imputed").astype(int))
           .groupby(id_col)["is_imp"]
           .agg(imputed="sum", total="count"))
    if tab.empty:
        return _no_data(ax, "No grouped data for coverage.")

    tab["imputed_ratio"] = tab["imputed"] / tab["total"]
    tab["observed_ratio"] = 1.0 - tab["imputed_ratio"]

    if sort_by not in {"imputed_ratio", "observed_ratio", "station"}:
        return _no_data(ax, "Invalid sort_by parameter.")

    if sort_by == "station":
        tab = tab.sort_index()
    else:
        tab = tab.sort_values(sort_by, ascending=False)

    ax.bar(tab.index.astype(str), tab["imputed_ratio"].values)
    ax.set_ylabel("Imputed share")
    ax.set_xlabel(id_col)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Imputation coverage per station")
    # Reduce x label clutter for many stations
    if len(tab) > 20:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
            label.set_ha("center")
    return ax


__all__ = [
    "plot_missing_matrix",
    "plot_metrics_distribution",
    "plot_parity_scatter",
    "plot_time_series_overlay",
    "plot_spatial_scatter",
    "plot_gap_histogram",
    "plot_imputed_series",
    "plot_imputation_coverage",
]
