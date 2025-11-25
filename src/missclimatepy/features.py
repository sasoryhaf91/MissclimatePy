# SPDX-License-Identifier: MIT
"""
Core feature engineering utilities for MissClimatePy.

This module provides small, composable helpers to prepare long-format
daily station data for XYZT-style models, where inputs are:

    X = (x, y, z, t) = (longitude, latitude, elevation, calendar features)

and the target is typically a single climate variable (e.g. precipitation,
tmin, tmax, evap).

Design goals
------------
- Column-name agnostic: callers pass id/date/coord/target column names.
- Timezone-safe: datetime parsing always yields naive datetime64[ns].
- Minimal dependencies: only pandas and numpy.
- Reusable: used by both evaluate.py and impute.py.

Public functions
----------------
- validate_required_columns : strict schema checking.
- ensure_datetime_naive     : parse and drop timezone in a robust way.
- add_calendar_features     : add year / month / day-of-year (+ cyclic).
- default_feature_names     : canonical feature list for XYZT models.
- preprocess_for_model      : one-stop preprocessing for modelling.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def validate_required_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """
    Raise a ValueError if any of the requested columns is missing.

    This is intentionally strict and should be used early in user-facing
    functions (evaluate, impute, etc.) to provide clear error messages.

    Parameters
    ----------
    df : DataFrame
        Input table.
    cols : sequence of str
        Column names that must be present.

    Raises
    ------
    ValueError
        If one or more columns are missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sample_cols = list(df.columns)[:12]
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns include: {sample_cols}..."
        )


# ---------------------------------------------------------------------------
# Datetime handling
# ---------------------------------------------------------------------------


def ensure_datetime_naive(series: pd.Series) -> pd.Series:
    """
    Parse a Series to naive datetime64[ns] and drop any timezone.

    Implementation notes
    --------------------
    - Uses ``utc=True`` in ``pd.to_datetime`` to avoid future warnings
      about mixed timezones; this yields a tz-aware UTC series.
    - Then converts to naive (timezone-free) using ``tz_convert(None)``.

    Any non-parseable values become NaT.

    Parameters
    ----------
    series : pandas.Series
        Input values (strings, datetimes, mixed).

    Returns
    -------
    pandas.Series
        Series with dtype datetime64[ns] (tz-naive).
    """
    dt_utc = pd.to_datetime(series, errors="coerce", utc=True)
    dt_naive = dt_utc.dt.tz_convert(None)
    return dt_naive


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    add_cyclic: bool = False,
) -> pd.DataFrame:
    """
    Add standard calendar features derived from a date column.

    Features added
    --------------
    - year : int32
    - month: int16
    - doy  : int16 (day-of-year)

    If ``add_cyclic=True`` also adds:
    - doy_sin, doy_cos : sin/cos transforms on a 365.25-day cycle.

    The function returns a *copy* of the input with additional columns.

    Parameters
    ----------
    df : DataFrame
        Input data, must contain ``date_col``.
    date_col : str
        Name of the date column.
    add_cyclic : bool, default False
        Whether to add harmonic (sin/cos) transforms of day-of-year.

    Returns
    -------
    DataFrame
        Copy of ``df`` with calendar features added.
    """
    out = df.copy()
    out[date_col] = ensure_datetime_naive(out[date_col])

    # Basic calendar fields
    out["year"] = out[date_col].dt.year.astype("int32", copy=False)
    out["month"] = out[date_col].dt.month.astype("int16", copy=False)
    out["doy"] = out[date_col].dt.dayofyear.astype("int16", copy=False)

    if add_cyclic:
        two_pi = 2.0 * np.pi
        doy_arr = out["doy"].to_numpy()
        out["doy_sin"] = np.sin(two_pi * doy_arr / 365.25)
        out["doy_cos"] = np.cos(two_pi * doy_arr / 365.25)

    return out


def default_feature_names(
    *,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    add_cyclic: bool = False,
    extra: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Build the canonical feature list for XYZT-style models.

    Base features
    -------------
    - latitude, longitude, altitude (as provided by the caller)
    - year, month, doy

    If ``add_cyclic=True``:
    - doy_sin, doy_cos

    Any ``extra`` features are appended and deduplicated while preserving
    the original order of appearance.

    Parameters
    ----------
    lat_col, lon_col, alt_col : str
        Column names for coordinates in the *input* DataFrame.
    add_cyclic : bool, default False
        Whether cyclic calendar terms will be present/used.
    extra : sequence of str or None
        Additional feature column names to include.

    Returns
    -------
    list of str
        Ordered list of feature column names.
    """
    feats: List[str] = [lat_col, lon_col, alt_col, "year", "month", "doy"]
    if add_cyclic:
        feats += ["doy_sin", "doy_cos"]
    if extra is not None:
        feats += list(extra)

    # Deduplicate preserving order
    seen = set()
    out: List[str] = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# One-stop preprocessing for XYZT models
# ---------------------------------------------------------------------------


def preprocess_for_model(
    data: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess a long-format daily climate table for XYZT-style modeling.

    Steps
    -----
    - Validate required columns.
    - Coerce ``date_col`` to naive datetime and drop rows with invalid dates.
    - Optionally clip to [start, end].
    - Add calendar features via :func:`add_calendar_features`.
    - Build the feature list:

      * If ``feature_cols`` is None: use :func:`default_feature_names`
        (coords + calendar [+ cyclic]).
      * Otherwise: use ``feature_cols`` exactly (deduplicated, order preserved).

    The returned DataFrame always includes:

        [id_col, date_col, lat_col, lon_col, alt_col, target_col,
         year, month, doy, (doy_sin, doy_cos if add_cyclic),
         and any user-specified feature columns].

    The returned ``feature_names`` list is exactly the feature set used for
    modeling; calendar columns may or may not be included there depending on
    ``feature_cols``.
    """
    # 1) Basic validation
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
    )

    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    # 2) Optional time window
    if start is not None or end is not None:
        lo = pd.to_datetime(start) if start is not None else df[date_col].min()
        hi = pd.to_datetime(end) if end is not None else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # If nothing remains, return an empty scaffold and a reasonable feature list
    if df.empty:
        feats = default_feature_names(
            lat_col=lat_col,
            lon_col=lon_col,
            alt_col=alt_col,
            add_cyclic=add_cyclic,
        )
        empty_cols = sorted(
            set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats)
        )
        empty = pd.DataFrame(columns=empty_cols)
        return empty, feats

    # 3) Add calendar features
    df = add_calendar_features(df, date_col=date_col, add_cyclic=add_cyclic)

    # Calendar columns we always want to retain in the prepared DataFrame
    calendar_cols = ["year", "month", "doy"]
    if add_cyclic:
        calendar_cols += ["doy_sin", "doy_cos"]

    # 4) Define the feature list for the model
    if feature_cols is None:
        # Standard path: coords + calendar (+ cyclic)
        feats = default_feature_names(
            lat_col=lat_col,
            lon_col=lon_col,
            alt_col=alt_col,
            add_cyclic=add_cyclic,
        )
    else:
        # Respect exactly what the user requested (deduplicated, order preserved)
        seen = set()
        feats: List[str] = []
        for c in feature_cols:
            if c not in seen:
                feats.append(c)
                seen.add(c)

    # 5) Final column subset to keep
    base_cols = [id_col, date_col, lat_col, lon_col, alt_col, target_col]
    keep = sorted(set(base_cols + feats + calendar_cols))
    df = df[keep]

    return df, feats


__all__ = [
    "validate_required_columns",
    "ensure_datetime_naive",
    "add_calendar_features",
    "default_feature_names",
    "preprocess_for_model",
]

