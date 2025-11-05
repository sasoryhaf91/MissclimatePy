# SPDX-License-Identifier: MIT
"""
missclimatepy.viz
=================

Lightweight visualization helpers for MissClimatePy.

Design rules
------------
- Pure matplotlib (no seaborn, no styles, no custom colors).
- One chart per function (no subplots).
- Functions return the Axes instance for further customization/saving.
- Inputs are pandas/numpy; no heavy GIS deps. A minimal "map" uses lon/lat axes.

Functions
---------
- plot_missingness_calendar: day-level heatmap (present vs missing) for one station.
- plot_station_completeness: histogram of valid counts per station.
- plot_parity: y_true vs y_pred scatter with identity line and text metrics.
- plot_error_by_doy: absolute error vs day-of-year (box-like via scatter jitter/median).
- plot_error_by_month: absolute error by calendar month (scatter + medians).
- map_rmse: lon/lat scatter sized by RMSE (simple spatial view).

Notes
-----
- For large datasets, consider sampling before plotting.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _to_datetime_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_localize(None)
    return s


def _identity_limits(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    lo = float(np.nanmin([np.nanmin(a), np.nanmin(b)]))
    hi = float(np.nanmax([np.nanmax(a), np.nanmax(b)]))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _safe_median(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else np.nan


# ---------------------------------------------------------------------
# 1) Missingness calendar (per station)
# ---------------------------------------------------------------------
def plot_missingness_calendar(
    df: pd.DataFrame,
    *,
    station_id: str | int,
    id_col: str,
    date_col: str,
    target_col: str,
    figsize: Tuple[float, float] = (10, 2.5),
):
    """
    Draw a day-level present/missing "calendar strip" for one station.

    The strip shows 1 row of pixels (one per day). Present=1, Missing=0.

    Parameters
    ----------
    df : DataFrame (long format)
    station_id : station to display
    id_col, date_col, target_col : column names
    figsize : figure size

    Returns
    -------
    Axes
    """
    sub = df.loc[df[id_col] == station_id, [date_col, target_col]].copy()
    if sub.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data for station", ha="center", va="center")
        ax.set_axis_off()
        return ax

    sub[date_col] = _to_datetime_naive(sub[date_col])
    sub = sub.dropna(subset=[date_col]).sort_values(date_col)

    # Build continuous date index
    full_idx = pd.date_range(sub[date_col].min(), sub[date_col].max(), freq="D")
    present = pd.Series(1.0, index=sub.loc[sub[target_col].notna(), date_col].values)
    present = present.reindex(full_idx, fill_value=0.0).to_numpy()[None, :]  # shape (1, n_days)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(present, aspect="auto", extent=[0, present.shape[1], 0, 1], interpolation="nearest")
    ax.set_yticks([])
    ax.set_xlabel("Days")
    ax.set_title(f"Missingness calendar — {station_id}")
    return ax


# ---------------------------------------------------------------------
# 2) Station completeness histogram
# ---------------------------------------------------------------------
def plot_station_completeness(
    df: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    bins: int = 20,
    figsize: Tuple[float, float] = (6, 4),
):
    """
    Histogram of valid (non-missing) observations per station.

    Parameters
    ----------
    df : DataFrame
    id_col, target_col : column names
    bins : histogram bins
    figsize : figure size

    Returns
    -------
    Axes
    """
    counts = df[df[target_col].notna()].groupby(id_col).size().astype(int)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(counts.values, bins=bins)
    ax.set_xlabel("Valid observations per station")
    ax.set_ylabel("Frequency")
    ax.set_title("Station completeness")
    return ax


# ---------------------------------------------------------------------
# 3) Parity plot (y_true vs y_pred)
# ---------------------------------------------------------------------
def plot_parity(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    sample: Optional[int] = None,
    figsize: Tuple[float, float] = (5, 5),
):
    """
    Scatter of y_true vs y_pred with identity line and text MAE/RMSE/R².

    Parameters
    ----------
    y_true, y_pred : sequences (same length)
    sample : optional int to subsample points for large arrays
    figsize : figure size

    Returns
    -------
    Axes
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if sample is not None and y_true.size > sample:
        idx = np.random.RandomState(0).choice(y_true.size, size=sample, replace=False)
        y_true, y_pred = y_true[idx], y_pred[idx]

    lo, hi = _identity_limits(y_true, y_pred)

    # metrics
    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff))) if diff.size else np.nan
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else np.nan
    var_y = float(np.var(y_true)) if diff.size else np.nan
    r2 = float(1.0 - np.mean(diff ** 2) / var_y) if diff.size and var_y > 0 else np.nan

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, s=10)
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title("Parity plot")

    txt = f"MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.3f}" if np.isfinite(mae) else "No data"
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, va="top")
    return ax


# ---------------------------------------------------------------------
# 4) Error by day-of-year
# ---------------------------------------------------------------------
def plot_error_by_doy(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    figsize: Tuple[float, float] = (7, 4),
):
    """
    Absolute error |y_true - y_pred| vs day-of-year.

    Parameters
    ----------
    df_pred : DataFrame containing columns [date_col, y_true_col, y_pred_col]
    date_col : date column name
    y_true_col, y_pred_col : column names
    figsize : figure size

    Returns
    -------
    Axes
    """
    work = df_pred[[date_col, y_true_col, y_pred_col]].copy()
    work[date_col] = _to_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col, y_true_col, y_pred_col])
    if work.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    work["doy"] = work[date_col].dt.dayofyear.astype(int)
    work["ae"] = np.abs(work[y_true_col].to_numpy() - work[y_pred_col].to_numpy())

    med = work.groupby("doy")["ae"].median().reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(work["doy"].to_numpy(), work["ae"].to_numpy(), s=6, alpha=0.6)
    ax.plot(med["doy"].to_numpy(), med["ae"].to_numpy())
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Absolute error")
    ax.set_title("Error by day-of-year")
    return ax


# ---------------------------------------------------------------------
# 5) Error by month
# ---------------------------------------------------------------------
def plot_error_by_month(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    figsize: Tuple[float, float] = (7, 4),
):
    """
    Absolute error |y_true - y_pred| by calendar month.

    Parameters
    ----------
    df_pred : DataFrame with [date_col, y_true_col, y_pred_col]
    date_col : date column
    y_true_col, y_pred_col : columns
    figsize : figure size

    Returns
    -------
    Axes
    """
    work = df_pred[[date_col, y_true_col, y_pred_col]].copy()
    work[date_col] = _to_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col, y_true_col, y_pred_col])
    if work.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    work["month"] = work[date_col].dt.month.astype(int)
    work["ae"] = np.abs(work[y_true_col].to_numpy() - work[y_pred_col].to_numpy())
    med = work.groupby("month")["ae"].median().reset_index()

    fig, ax = plt.subplots(figsize=figsize)
    # scatter with jitter
    jitter = (np.random.RandomState(0).rand(len(work)) - 0.5) * 0.2
    ax.scatter(work["month"].to_numpy() + jitter, work["ae"].to_numpy(), s=6, alpha=0.6)
    ax.plot(med["month"].to_numpy(), med["ae"].to_numpy())
    ax.set_xticks(np.arange(1, 13))
    ax.set_xlabel("Month")
    ax.set_ylabel("Absolute error")
    ax.set_title("Error by month")
    return ax


# ---------------------------------------------------------------------
# 6) Minimal spatial view: RMSE on lon/lat axes
# ---------------------------------------------------------------------
def map_rmse(
    station_report: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    rmse_col: str = "RMSE_d",
    s_min: float = 10.0,
    s_max: float = 80.0,
    figsize: Tuple[float, float] = (6, 4),
):
    """
    Minimal geographical scatter: RMSE vs lon/lat (no GIS libs required).

    Parameters
    ----------
    station_report : DataFrame with per-station metrics and coordinates
    id_col, lat_col, lon_col : column names
    rmse_col : which RMSE to draw (e.g., daily)
    s_min, s_max : point size range
    figsize : figure size

    Returns
    -------
    Axes
    """
    required = [id_col, lat_col, lon_col, rmse_col]
    for c in required:
        if c not in station_report.columns:
            raise ValueError(f"map_rmse: missing column '{c}'")

    work = station_report[required].dropna()
    if work.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_axis_off()
        return ax

    rmse = work[rmse_col].to_numpy()
    if np.nanmax(rmse) == np.nanmin(rmse):
        sizes = np.full_like(rmse, (s_min + s_max) / 2.0, dtype=float)
    else:
        sizes = (rmse - np.nanmin(rmse)) / (np.nanmax(rmse) - np.nanmin(rmse))
        sizes = s_min + sizes * (s_max - s_min)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(work[lon_col].to_numpy(), work[lat_col].to_numpy(), s=sizes)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Station RMSE ({rmse_col})")
    return ax

