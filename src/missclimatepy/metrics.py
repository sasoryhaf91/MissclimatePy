# SPDX-License-Identifier: MIT
"""
missclimatepy.metrics
=====================

Core regression metrics for MissClimatePy.

This module centralizes metric calculations used across the package:

- Point-scale metrics on paired series:

  * MAE  (mean absolute error)
  * RMSE (root-mean-square error)
  * R2   (coefficient of determination)
  * KGE  (Kling–Gupta efficiency)

- Convenience helpers:

  * ``compute_metrics``: returns a dictionary with MAE/RMSE/R2 and
    optionally KGE, handling NaNs and empty inputs gracefully.
  * ``aggregate_and_compute``: aggregates a time series to a given
    frequency (e.g. monthly, yearly) and computes the same metrics.

The design is intentionally lightweight:

- Input arrays are converted to NumPy float arrays and paired NaNs
  are removed before computing metrics.
- Degenerate cases (empty series, zero variance, zero means) return
  metrics as ``np.nan`` instead of raising errors.
- Time aggregation uses :func:`missclimatepy.features.ensure_datetime_naive`
  to standardize datetime handling.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .features import ensure_datetime_naive, validate_required_columns


# --------------------------------------------------------------------------- #
# Basic metric primitives
# --------------------------------------------------------------------------- #


def _to_clean_pairs(
    y_true,
    y_pred,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert inputs to paired float arrays and drop NaNs pairwise.

    Parameters
    ----------
    y_true, y_pred : array-like
        Observed and predicted values.

    Returns
    -------
    (yt, yp) : tuple of ndarray
        Cleaned arrays of matching length, dtype float64.
    """
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")

    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape. "
            f"Got {yt.shape} vs {yp.shape}."
        )

    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt = yt[mask]
    yp = yp[mask]
    return yt, yp


def mae(y_true, y_pred) -> float:
    """
    Mean absolute error (MAE) with NaN/empty handling.

    Returns ``np.nan`` if the cleaned series is empty.
    """
    yt, yp = _to_clean_pairs(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """
    Root-mean-square error (RMSE) with NaN/empty handling.

    Returns ``np.nan`` if the cleaned series is empty.
    """
    yt, yp = _to_clean_pairs(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    diff = yt - yp
    return float(np.sqrt(np.mean(diff * diff)))


def r2(y_true, y_pred) -> float:
    """
    Coefficient of determination R² with NaN/degenerate handling.

    - If the cleaned series is empty, returns NaN.
    - If variance of y_true is zero or length < 2, returns NaN.
    """
    yt, yp = _to_clean_pairs(y_true, y_pred)
    if yt.size < 2:
        return float("nan")

    var_y = float(np.var(yt))
    if var_y == 0.0:
        return float("nan")

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    if ss_tot == 0.0:
        return float("nan")

    return float(1.0 - ss_res / ss_tot)


def kge(y_true, y_pred) -> float:
    """
    Kling–Gupta Efficiency (KGE) following the revised formulation:

        KGE = 1 - sqrt( (r - 1)^2 + (α - 1)^2 + (β - 1)^2 )

    where:

        r  = Pearson correlation between y_pred and y_true
        α  = σ_pred / σ_true  (ratio of standard deviations)
        β  = μ_pred / μ_true  (ratio of means)

    Notes
    -----
    - Pairwise NaNs are dropped before calculation.
    - Returns ``np.nan`` if the cleaned series is empty or if any
      of the required statistics (mean, std, correlation) is not
      well defined (e.g. zero variance, zero mean).
    """
    yt, yp = _to_clean_pairs(y_true, y_pred)
    n = yt.size

    if n == 0:
        return float("nan")

    mu_o = float(np.mean(yt))
    mu_p = float(np.mean(yp))
    std_o = float(np.std(yt, ddof=1)) if n > 1 else 0.0
    std_p = float(np.std(yp, ddof=1)) if n > 1 else 0.0

    # Degenerate cases: undefined stats → NaN
    if n < 2 or std_o == 0.0 or mu_o == 0.0:
        return float("nan")

    # Pearson correlation
    r_num = float(np.sum((yt - mu_o) * (yp - mu_p)))
    r_den = float((n - 1) * std_o * std_p) if std_p != 0.0 else 0.0
    if r_den == 0.0:
        return float("nan")
    r = r_num / r_den

    alpha = std_p / std_o
    beta = mu_p / mu_o

    kge_val = 1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(kge_val)


# --------------------------------------------------------------------------- #
# High-level helpers
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true,
    y_pred,
    *,
    include_kge: bool = True,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R² and optionally KGE for a paired series.

    Parameters
    ----------
    y_true, y_pred : array-like
        Observed and predicted values.
    include_kge : bool, default True
        If True, includes "KGE" in the returned dictionary.

    Returns
    -------
    dict
        Keys:

        - "MAE"
        - "RMSE"
        - "R2"
        - "KGE" (optional, if ``include_kge=True``)

        Values are floats, possibly NaN in degenerate cases.
    """
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    if include_kge:
        out["KGE"] = kge(y_true, y_pred)
    return out


# --------------------------------------------------------------------------- #
# Time aggregation + metrics
# --------------------------------------------------------------------------- #

_FREQ_ALIAS = {"M": "MS", "A": "YS", "Y": "YS", "Q": "QS"}


def _normalize_freq(freq: str) -> str:
    """
    Normalize short frequency codes to start-of-period aliases:

    - "M"  → "MS" (month start)
    - "Y"/"A" → "YS" (year start)
    - "Q"      → "QS" (quarter start)
    """
    return _FREQ_ALIAS.get(freq, freq)


def _validate_agg(agg: str) -> str:
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg must be one of {'sum', 'mean', 'median'}.")
    return agg


def aggregate_and_compute(
    df: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    yhat_col: str,
    freq: str = "M",
    agg: str = "sum",
    include_kge: bool = True,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Aggregate a time series to a given frequency and compute metrics.

    Parameters
    ----------
    df : DataFrame
        Table with at least [date_col, y_col, yhat_col].
    date_col : str
        Name of the datetime column.
    y_col, yhat_col : str
        Column names for observed and predicted values.
    freq : str, default "M"
        Pandas offset alias for resampling. Short forms "M", "Y"/"A", "Q"
        are normalized to their start-of-period equivalents.
    agg : {"sum","mean","median"}, default "sum"
        Aggregation function applied to both observed and predicted.
    include_kge : bool, default True
        If True, includes KGE in the returned metrics.

    Returns
    -------
    metrics : dict
        Same structure as :func:`compute_metrics`, computed on the
        aggregated series.
    agg_df : DataFrame
        Aggregated table with columns [y_col, yhat_col], indexed by
        the resampled datetime index.

    Notes
    -----
    - If ``df`` is empty or the aggregation results in an empty table,
      all metrics are returned as NaN and ``agg_df`` is empty.
    """
    validate_required_columns(
        df,
        [date_col, y_col, yhat_col],
        context="aggregate_and_compute",
    )

    if df.empty:
        return {k: float("nan") for k in (["MAE", "RMSE", "R2"] + (["KGE"] if include_kge else []))}, df

    work = df[[date_col, y_col, yhat_col]].copy()
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    if work.empty:
        return {k: float("nan") for k in (["MAE", "RMSE", "R2"] + (["KGE"] if include_kge else []))}, work

    freq_norm = _normalize_freq(freq)
    agg_op = _validate_agg(agg)

    work = work.set_index(date_col).sort_index()
    agg_df = work.resample(freq_norm).agg({y_col: agg_op, yhat_col: agg_op}).dropna()

    if agg_df.empty:
        return {k: float("nan") for k in (["MAE", "RMSE", "R2"] + (["KGE"] if include_kge else []))}, agg_df

    metrics = compute_metrics(agg_df[y_col].values, agg_df[yhat_col].values, include_kge=include_kge)
    return metrics, agg_df


__all__ = [
    "mae",
    "rmse",
    "r2",
    "kge",
    "compute_metrics",
    "aggregate_and_compute",
]
