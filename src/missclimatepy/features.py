# SPDX-License-Identifier: MIT
"""
Core feature utilities for MissClimatePy.

This module centralises small, reusable helpers for:

- Ensuring a naive (timezone–free) datetime column.
- Adding standard calendar features (year, month, day-of-year, optional
  sin/cos transforms) to a dataframe.
- Validating that required columns are present in an input dataframe.
- Selecting station identifiers based on prefixes, explicit lists, regex
  patterns, or custom predicates.
- Filtering stations by a minimum number of observed (non-missing) target
  values.

These functions are intentionally lightweight and are reused across
`evaluate`, `impute`, `masking`, and `viz`, so that column handling
and feature construction remain consistent throughout the package.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Basic column helpers
# --------------------------------------------------------------------------- #


def ensure_datetime_naive(s: pd.Series) -> pd.Series:
    """
    Convert a Series to ``datetime64[ns]`` without timezone information.

    The function is strict but robust:

    - Values are parsed with :func:`pandas.to_datetime` using
      ``errors="coerce"``, so unparsable entries become ``NaT``.
    - If the resulting dtype is timezone-aware, the timezone is dropped via
      ``.dt.tz_localize(None)``.

    Parameters
    ----------
    s : Series
        Input series (any dtype); typically a date column.

    Returns
    -------
    Series
        A ``datetime64[ns]`` series with no timezone information.
    """
    out = pd.to_datetime(s, errors="coerce")

    # Drop timezone if present (tz-aware -> naive)
    # `.dt.tz` is None for naive series.
    tz = getattr(out.dt, "tz", None)
    if tz is not None:
        out = out.dt.tz_localize(None)

    return out


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    *,
    context: str | None = None,
) -> None:
    """
    Raise a :class:`ValueError` if any of the requested columns is missing.

    This helper is used early in most public functions to provide clearer
    error messages to end users.

    Parameters
    ----------
    df : DataFrame
        Input table.
    required : sequence of str
        Column names that must be present.
    context : str, optional
        Optional short label indicating which function is performing
        the validation. This is only used in the error message.

    Raises
    ------
    ValueError
        If at least one required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return

    prefix = f"[{context}] " if context else ""
    available_preview = list(df.columns[:12])
    raise ValueError(
        f"{prefix}Missing required columns: {missing}. "
        f"Available columns include: {available_preview}..."
    )


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #


def add_time_features(
    df: pd.DataFrame,
    *,
    date_col: str,
    add_cyclic: bool = False,
) -> pd.DataFrame:
    """
    Add standard calendar features derived from a date column.

    The following columns are added:

    - ``year``  : calendar year (int32)
    - ``month`` : calendar month (1–12, int16)
    - ``doy``   : day-of-year (1–366, int16)

    If ``add_cyclic=True``, two additional columns are added:

    - ``doy_sin`` : sine transform of day-of-year on a 365.25-day cycle,
    - ``doy_cos`` : cosine transform of day-of-year on a 365.25-day cycle.

    Parameters
    ----------
    df : DataFrame
        Input table containing at least ``date_col``.
    date_col : str
        Name of the column with date information.
    add_cyclic : bool, default False
        Whether to add cyclic (sin/cos) encodings of the day-of-year.

    Returns
    -------
    DataFrame
        A *copy* of the input dataframe with the additional columns.
    """
    validate_required_columns(df, [date_col], context="add_time_features")

    out = df.copy()
    out[date_col] = ensure_datetime_naive(out[date_col])

    # Basic calendar fields
    out["year"] = out[date_col].dt.year.astype("int32", copy=False)
    out["month"] = out[date_col].dt.month.astype("int16", copy=False)
    out["doy"] = out[date_col].dt.dayofyear.astype("int16", copy=False)

    if add_cyclic:
        two_pi = 2.0 * np.pi
        doy_arr = out["doy"].to_numpy(dtype=float)
        out["doy_sin"] = np.sin(two_pi * doy_arr / 365.25)
        out["doy_cos"] = np.cos(two_pi * doy_arr / 365.25)

    return out


# --------------------------------------------------------------------------- #
# Station selection helpers
# --------------------------------------------------------------------------- #


def select_station_ids(
    df: pd.DataFrame,
    *,
    id_col: str,
    prefix: str | Iterable[str] | None = None,
    station_ids: Iterable[object] | None = None,
    regex: str | None = None,
    custom_filter: Callable[[object], bool] | None = None,
) -> List[object]:
    """
    Select station identifiers using flexible filters (OR semantics).

    The underlying order is the natural order of first appearance in the
    dataframe. When multiple filters are provided, the union of all matches
    is returned, with duplicates removed while preserving this order.

    Parameters
    ----------
    df : DataFrame
        Input table containing at least ``id_col``.
    id_col : str
        Column with station identifiers.
    prefix : str or iterable of str, optional
        One or several prefixes. A station is selected if its string
        representation ``str(station_id)`` starts with *any* of these.
    station_ids : iterable, optional
        Explicit list of station identifiers to include.
    regex : str, optional
        Regular expression pattern applied to ``str(station_id)``.
    custom_filter : callable, optional
        Function ``f(station_id) -> bool``. Stations for which this function
        returns True are included.

    Returns
    -------
    list
        Selected station identifiers. If no filters are provided, all unique
        stations in the dataframe are returned.
    """
    validate_required_columns(df, [id_col], context="select_station_ids")

    all_ids = df[id_col].dropna().unique().tolist()
    chosen: List[object] = []

    # If no filters at all, return all unique ids
    if prefix is None and station_ids is None and regex is None and custom_filter is None:
        return all_ids

    # Prefix filter
    if prefix is not None:
        if isinstance(prefix, str):
            prefixes = [prefix]
        else:
            prefixes = list(prefix)
        for sid in all_ids:
            s = str(sid)
            if any(s.startswith(p) for p in prefixes):
                chosen.append(sid)

    # Explicit station ids
    if station_ids is not None:
        for sid in station_ids:
            chosen.append(sid)

    # Regex filter
    if regex is not None:
        import re

        pat = re.compile(regex)
        for sid in all_ids:
            if pat.match(str(sid)):
                chosen.append(sid)

    # Custom predicate
    if custom_filter is not None:
        for sid in all_ids:
            try:
                if custom_filter(sid):
                    chosen.append(sid)
            except Exception:
                # Be lenient: ignore errors in user-supplied predicate
                continue

    # Deduplicate while preserving order and restricting to existing ids
    seen = set()
    result: List[object] = []
    existing = set(all_ids)
    for sid in chosen:
        if sid in existing and sid not in seen:
            seen.add(sid)
            result.append(sid)

    # If filters yielded nothing, fall back to all_ids
    return result or all_ids


def filter_by_min_station_rows(
    df: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    min_station_rows: int,
) -> List[object]:
    """
    Return station IDs that have at least a minimum number of observations.

    Only **non-missing** values of ``target_col`` are counted.

    Parameters
    ----------
    df : DataFrame
        Input table.
    id_col : str
        Station identifier column.
    target_col : str
        Target variable whose non-null counts are used.
    min_station_rows : int
        Minimum required number of non-null observations.

    Returns
    -------
    list
        Station identifiers that satisfy the threshold.
    """
    validate_required_columns(
        df,
        [id_col, target_col],
        context="filter_by_min_station_rows",
    )

    if min_station_rows <= 0:
        # Trivial case: all stations qualify
        return df[id_col].dropna().unique().tolist()

    counts = (
        df.loc[~df[target_col].isna(), [id_col, target_col]]
        .groupby(id_col)[target_col]
        .size()
        .astype(int)
    )

    ok = counts[counts >= int(min_station_rows)]
    return ok.index.tolist()


__all__ = [
    "ensure_datetime_naive",
    "validate_required_columns",
    "add_time_features",
    "select_station_ids",
    "filter_by_min_station_rows",
]
