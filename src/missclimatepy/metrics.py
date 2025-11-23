# SPDX-License-Identifier: MIT
"""
missclimatepy.metrics
=====================

Metric utilities for MissClimatePy.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ----------------------------------------------------------
# Safe helpers
# ----------------------------------------------------------

def _safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _safe_std(x: np.ndarray) -> float:
    return float(np.nanstd(x, ddof=1)) if np.isfinite(x).any() else np.nan


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = np.sqrt((xm**2).sum() * (ym**2).sum())
    return float((xm * ym).sum() / denom) if denom != 0 else np.nan


# ----------------------------------------------------------
# KGE
# ----------------------------------------------------------

def compute_kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if mask.sum() < 2:
        # Special case: only one pair and it matches perfectly → KGE = 1
        if mask.sum() == 1 and y_true[mask][0] == y_pred[mask][0]:
            return 1.0
        return np.nan

    yt = y_true[mask]
    yp = y_pred[mask]

    mu_o = yt.mean()
    mu_p = yp.mean()
    sigma_o = yt.std(ddof=1)
    sigma_p = yp.std(ddof=1)
    r = _safe_corr(yt, yp)

    if sigma_o == 0 or not np.isfinite(r):
        return np.nan

    alpha = sigma_p / sigma_o
    beta = mu_p / mu_o

    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


# ----------------------------------------------------------
# MCM
# ----------------------------------------------------------

def compute_mcm_baseline(
    *,
    dates: Any,
    values: Any,
    mode: str = "doy",
    min_samples: int = 1,
) -> pd.Series:

    dates = pd.to_datetime(dates)
    vals = pd.Series(values).astype(float)
    n = len(vals)
    global_mean = vals.mean(skipna=True)

    if not np.isfinite(global_mean):
        return pd.Series(np.nan, index=range(n), dtype=float)

    if mode.lower() == "global":
        return pd.Series(global_mean, index=range(n), dtype=float)

    if mode.lower() == "month":
        groups = dates.month
    elif mode.lower() == "doy":
        groups = dates.dayofyear
    else:
        raise ValueError("Unknown MCM mode")

    df = pd.DataFrame({"grp": groups, "val": vals})
    grp_stats = df.groupby("grp")["val"].agg(["mean", "count"])
    grp_mean = grp_stats["mean"].copy()
    grp_mean[grp_stats["count"] < min_samples] = np.nan
    local_vals = df["grp"].map(grp_mean).fillna(global_mean)

    # Ensure exact length
    return pd.Series(local_vals.values, index=range(n), dtype=float)


# ----------------------------------------------------------
# Multi-scale metrics
# ----------------------------------------------------------

@dataclass
class _MetricSet:
    MAE: float
    RMSE: float
    R2: float
    KGE: float


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> _MetricSet:
    """Compute MAE, RMSE, R2 and KGE on finite pairs (y_true, y_pred).

    Notes
    -----
    - If there are **no** finite (y_true, y_pred) pairs, this returns
      MAE = 0.0, RMSE = 0.0, and R2 = KGE = NaN. This avoids propagating
      NaNs to station-level summaries when no observations are available,
      while keeping correlation-like metrics undefined.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    # Case: zero finite values → no information
    if mask.sum() == 0:
        return _MetricSet(
            MAE=0.0,
            RMSE=0.0,
            R2=np.nan,
            KGE=np.nan,
        )

    # Case: single value → "perfect" if equal
    if mask.sum() == 1:
        if y_true[mask][0] == y_pred[mask][0]:
            return _MetricSet(0.0, 0.0, 1.0, 1.0)
        else:
            diff = abs(y_true[mask][0] - y_pred[mask][0])
            return _MetricSet(
                MAE=diff,
                RMSE=diff,
                R2=1.0,
                KGE=np.nan,
            )

    yt = y_true[mask]
    yp = y_pred[mask]

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    r2 = float(r2_score(yt, yp))
    kge = compute_kge(yt, yp)

    return _MetricSet(mae, rmse, r2, kge)


def multiscale_metrics(
    df: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    yhat_col: str,
    monthly_agg: str = "sum",
    annual_agg: str = "sum",
) -> Dict[str, Dict[str, float]]:

    if df.empty:
        empty = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "KGE": np.nan}
        return {"daily": empty, "monthly": empty, "annual": empty}

    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d.sort_values(by=date_col, inplace=True)

    # DAILY
    daily = _compute_basic_metrics(d[y_col], d[yhat_col])

    # MONTHLY
    d["year"] = d[date_col].dt.year
    d["month"] = d[date_col].dt.month
    agg_fn = monthly_agg.lower()
    mg = d.groupby(["year", "month"])[[y_col, yhat_col]].agg(agg_fn).reset_index()
    monthly = _compute_basic_metrics(mg[y_col], mg[yhat_col])

    # ANNUAL
    agg_fn2 = annual_agg.lower()
    ag = d.groupby("year")[[y_col, yhat_col]].agg(agg_fn2).reset_index()
    annual = _compute_basic_metrics(ag[y_col], ag[yhat_col])

    return {
        "daily": {
            "MAE": daily.MAE,
            "RMSE": daily.RMSE,
            "R2": daily.R2,
            "KGE": daily.KGE,
        },
        "monthly": {
            "MAE": monthly.MAE,
            "RMSE": monthly.RMSE,
            "R2": monthly.R2,
            "KGE": monthly.KGE,
        },
        "annual": {
            "MAE": annual.MAE,
            "RMSE": annual.RMSE,
            "R2": annual.R2,
            "KGE": annual.KGE,
        },
    }


__all__ = ["compute_kge", "compute_mcm_baseline", "multiscale_metrics"]
