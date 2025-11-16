# src/missclimatepy/features.py
# SPDX-License-Identifier: MIT
"""
missclimatepy.features
======================

Lightweight feature engineering utilities used across MissclimatePy.

This module focuses on the minimal, reproducible feature space adopted by the
package: spatial coordinates (latitude, longitude, altitude) and calendar
descriptors (year, month, day-of-year), with optional cyclic transforms
(sin/cos of day-of-year).

Why a separate module?
----------------------
- Single source of truth for feature creation (avoids drift across modules).
- Clean, dependency-minimal helpers that are easy to test.
- No deprecated pandas dtype checks; timezone handling is explicit.

Public
------
- ensure_datetime_naive
- validate_required_columns
- add_calendar_features
- default_feature_names
- assemble_feature_space
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_datetime_naive(series: pd.Series) -> pd.Series:
    """
    Parse to datetime (coercing errors) and drop timezone info if present.

    This function intentionally avoids deprecated checks like
    ``pd.api.types.is_datetime64tz_dtype`` and instead relies on the dtype
    class introduced by pandas for tz-aware datetimes.

    Parameters
    ----------
    series : pd.Series
        A pandas Series expected to contain datetime-like values.

    Returns
    -------
    pd.Series
        A timezone-naive datetime64[ns] Series. Invalid parses become NaT.
    """
    s = pd.to_datetime(series, errors="coerce")
    try:
        # pandas >= 1.0
        from pandas.api.types import DatetimeTZDtype  # type: ignore
        if isinstance(s.dtype, DatetimeTZDtype):
            s = s.dt.tz_localize(None)
    except Exception:
        # Fallback: best-effort if runtime does not expose DatetimeTZDtype
        try:
            _ = s.dt  # raises if not datetime-like
            s = s.dt.tz_localize(None)
        except Exception:
            pass
    return s


def validate_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """
    Raise a clear ValueError if any required column is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    columns : Sequence[str]
        Column names that must exist.

    Raises
    ------
    ValueError
        If one or more columns are not present in ``df``.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available: {list(df.columns)[:10]}..."
        )


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    add_cyclic: bool = False,
) -> pd.DataFrame:
    """
    Append calendar and (optionally) cyclic features based on a datetime column.

    Creates ``year``, ``month``, ``doy``. If ``add_cyclic`` is True, also adds
    ``doy_sin`` and ``doy_cos`` using period 365.25.

    Parameters
    ----------
    df : pd.DataFrame
        Input table. Must contain ``date_col`` with datetime64[ns] values.
    date_col : str
        Column name containing datetimes (timezone-naive preferred).
    add_cyclic : bool, optional
        Whether to include sin/cos of day-of-year. Default False.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with the new columns appended.

    Notes
    -----
    - Uses small integer dtypes for compact memory usage.
    - Cyclic features are float32 to keep arrays lean without losing signal.
    """
    out = df.copy()
    out["year"] = out[date_col].dt.year.astype("int16")
    out["month"] = out[date_col].dt.month.astype("int8")
    out["doy"] = out[date_col].dt.dayofyear.astype("int16")
    if add_cyclic:
        two_pi = np.float32(2.0 * np.pi)
        frac = out["doy"].astype("float32") / np.float32(365.25)
        out["doy_sin"] = np.sin(two_pi * frac)
        out["doy_cos"] = np.cos(two_pi * frac)
    return out


def default_feature_names(add_cyclic: bool = False) -> List[str]:
    """
    Return the canonical feature set used by MissclimatePy.

    Parameters
    ----------
    add_cyclic : bool, optional
        Include cyclic day-of-year terms if True.

    Returns
    -------
    list of str
        ``["latitude", "longitude", "altitude", "year", "month", "doy"]`` plus
        ``["doy_sin", "doy_cos"]`` when ``add_cyclic`` is True.
    """
    base = ["latitude", "longitude", "altitude", "year", "month", "doy"]
    if add_cyclic:
        base += ["doy_sin", "doy_cos"]
    return base


def assemble_feature_space(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    add_cyclic: bool = False,
    custom_feature_cols: Optional[Sequence[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build a *single* preprocessed DataFrame and the list of feature columns.

    This is the standard entry point used by both high-level API and the
    station-wise evaluator. It performs, in order:
      1) datetime parsing and timezone stripping;
      2) optional date-range clipping;
      3) calendar feature creation (and optional cyclic);
      4) feature list assembly and column subset.

    Parameters
    ----------
    df : pd.DataFrame
        Input long-format table.
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Schema column names supplied by the user.
    add_cyclic : bool, optional
        Include sin/cos of day-of-year. Default False.
    custom_feature_cols : Sequence[str] or None, optional
        If provided, these columns are used as the feature set *instead of*
        the default spatial + calendar set.
    start, end : str or None, optional
        Inclusive date window (ISO-like strings are fine).
        If both are None, no clipping is applied.

    Returns
    -------
    (DataFrame, List[str])
        - The preprocessed DataFrame (with calendar features added).
        - The list of feature column names in the returned DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = [id_col, date_col, lat_col, lon_col, alt_col, target_col]
    validate_required_columns(df, required)

    work = df.copy()

    # 1) Date parsing / TZ removal
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    # 2) Optional clipping (correct boolean '&', not Python 'and')
    if start is not None or end is not None:
        lo = pd.to_datetime(start) if start is not None else work[date_col].min()
        hi = pd.to_datetime(end) if end is not None else work[date_col].max()
        work = work[(work[date_col] >= lo) & (work[date_col] <= hi)]

    # 3) Calendar features
    work = add_calendar_features(work, date_col=date_col, add_cyclic=add_cyclic)

    # 4) Feature list assembly
    if custom_feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(custom_feature_cols)

    # Keep only relevant columns (feature + minimal schema for downstream ops)
    keep = sorted(set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats))
    work = work[keep]

    return work, feats


__all__ = [
    "ensure_datetime_naive",
    "validate_required_columns",
    "add_calendar_features",
    "default_feature_names",
    "assemble_feature_space",
]
