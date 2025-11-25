# SPDX-License-Identifier: MIT
"""
missclimatepy.features
======================

Core feature-engineering helpers for MissClimatePy.

This module centralizes small, reusable utilities to:

- Normalize datetime columns to timezone-free pandas datetimes.
- Add calendar features (year, month, day-of-year) and optional cyclic encodings.
- Build a consistent XYZT feature table for climate station data:
  X = (lat, lon, alt, calendar_features), T = time, Z = target.
- Validate required columns in user-provided DataFrames.
- Select station identifiers with flexible filters (prefix, regex, etc.).
- Filter station ids by a minimum number of observed rows.

These helpers are used by both the evaluation and imputation routines and are
kept generic so they can work with any long-format climate table where the
caller provides column names.
"""

from __future__ import annotations

from typing import (
    Callable,
    Hashable,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Datetime utilities
# ---------------------------------------------------------------------------


def ensure_datetime_naive(s: pd.Series) -> pd.Series:
    """
    Ensure that a Series is converted to timezone-free ``datetime64[ns]``.

    Steps
    -----
    1. Coerce values to datetime with ``errors='coerce'``.
    2. If the resulting dtype is timezone-aware, drop the timezone.

    Parameters
    ----------
    s : Series
        Input series with dates/timestamps.

    Returns
    -------
    Series
        Datetime64[ns] series with no timezone information.
    """
    s = pd.to_datetime(s, errors="coerce")
    # Avoid deprecated is_datetime64tz_dtype; inspect dtype instead
    if isinstance(s.dtype, pd.DatetimeTZDtype):  # type: ignore[attr-defined]
        s = s.dt.tz_localize(None)
    return s


# ---------------------------------------------------------------------------
# Calendar / cyclic features
# ---------------------------------------------------------------------------


def add_calendar_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    add_cyclic: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add basic calendar features derived from ``date_col``:

    - ``year``  (int32)
    - ``month`` (int16)
    - ``doy``   (day-of-year, int16)

    Optionally, cyclic encodings of day-of-year are added:

    - ``doy_sin``
    - ``doy_cos``

    Parameters
    ----------
    df : DataFrame
        Input table.
    date_col : str
        Name of the date/datetime column.
    add_cyclic : bool, default False
        Whether to append sine/cosine encodings of day-of-year.
    inplace : bool, default False
        If True, mutate ``df`` in-place and return it. Otherwise, work
        on a copy.

    Returns
    -------
    DataFrame
        Dataframe with added calendar (and optionally cyclic) columns.
    """
    out = df if inplace else df.copy()

    out[date_col] = ensure_datetime_naive(out[date_col])
    out = out.dropna(subset=[date_col])

    out["year"] = out[date_col].dt.year.astype("int32", copy=False)
    out["month"] = out[date_col].dt.month.astype("int16", copy=False)
    out["doy"] = out[date_col].dt.dayofyear.astype("int16", copy=False)

    if add_cyclic:
        add_cyclic_doy(out, doy_col="doy", prefix="doy", inplace=True)

    return out


def add_time_features(
    df: pd.DataFrame,
    date_col: str,
    *,
    add_cyclic: bool = False,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper around :func:`add_calendar_features`.

    Historically, MissClimatePy used the name ``add_time_features``. For
    clarity, the new name is ``add_calendar_features``, but this function
    is kept as a thin wrapper so existing code and tests still work.

    Parameters
    ----------
    df : DataFrame
        Input table.
    date_col : str
        Name of the date/datetime column.
    add_cyclic : bool, default False
        Whether to append cyclic encodings of day-of-year.
    inplace : bool, default False
        If True, mutate ``df`` in-place.

    Returns
    -------
    DataFrame
        Same as :func:`add_calendar_features`.
    """
    return add_calendar_features(df, date_col, add_cyclic=add_cyclic, inplace=inplace)


def add_cyclic_doy(
    df: pd.DataFrame,
    *,
    doy_col: str = "doy",
    prefix: str = "doy",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add sine/cosine cyclic encodings for a day-of-year column.

    Parameters
    ----------
    df : DataFrame
        Input table which must contain ``doy_col``.
    doy_col : str, default "doy"
        Name of the column containing the day-of-year (1..366).
    prefix : str, default "doy"
        Prefix for the new columns, resulting in f"{prefix}_sin" and
        f"{prefix}_cos".
    inplace : bool, default False
        If True, mutate ``df`` in-place. Otherwise work on a copy.

    Returns
    -------
    DataFrame
        Dataframe with two new columns: ``f"{prefix}_sin"`` and
        ``f"{prefix}_cos"``.
    """
    out = df if inplace else df.copy()

    if doy_col not in out.columns:
        raise ValueError(f"Column '{doy_col}' not found for cyclic encoding.")

    doy_values = pd.to_numeric(out[doy_col], errors="coerce").to_numpy()
    # Use 365.25 to keep leap years approximately consistent
    two_pi = 2.0 * np.pi
    phase = two_pi * doy_values / 365.25

    sin_name = f"{prefix}_sin"
    cos_name = f"{prefix}_cos"

    out[sin_name] = np.sin(phase)
    out[cos_name] = np.cos(phase)

    return out


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def validate_required_columns(
    df: pd.DataFrame,
    required: Sequence[str],
    context: Optional[str] = None,
) -> None:
    """
    Raise a ValueError if any required columns are missing.

    Parameters
    ----------
    df : DataFrame
        Input table.
    required : sequence of str
        Column names that must be present.
    context : str or None, default None
        Optional string to prepend to the error message (e.g., the
        calling function name).
    """
    missing = [c for c in required if c not in df.columns]
    if not missing:
        return

    prefix = f"[{context}] " if context else ""
    raise ValueError(
        f"{prefix}missing required columns {missing}. "
        f"Available columns include: {list(df.columns)[:12]}..."
    )


# ---------------------------------------------------------------------------
# Build XYZT-ready feature table
# ---------------------------------------------------------------------------


def build_xyzt_features(
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
    Prepare a long-format climate table for XYZT modelling.

    Steps
    -----
    1. Validate core columns.
    2. Normalize ``date_col`` to naive datetimes and optionally clip to
       ``[start, end]``.
    3. Add calendar features (year, month, doy, optional cyclic).
    4. Determine the list of feature columns to be used by models.

    Parameters
    ----------
    data : DataFrame
        Input table with at least the columns specified below.
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, datetime, latitude, longitude,
        altitude, and the target variable.
    start, end : str or None, default None
        Optional inclusive boundaries to clip the analysis period.
    add_cyclic : bool, default False
        Whether to include cyclic encodings of day-of-year.
    feature_cols : sequence of str or None, default None
        Explicit list of feature columns. If None, the default set is:

        - [lat_col, lon_col, alt_col, "year", "month", "doy"]
        - plus ["doy_sin", "doy_cos"] if ``add_cyclic=True``.

    Returns
    -------
    prepared_df : DataFrame
        Dataframe containing at least:

        - id_col, date_col, lat_col, lon_col, alt_col, target_col
        - all chosen feature columns.

        The function does **not** drop rows with missing features or
        target; downstream routines decide how to handle them.
    features_list : list of str
        Names of feature columns actually used.
    """
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
        context="build_xyzt_features",
    )

    df = data.copy()

    # Normalize and clip dates
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if start is not None or end is not None:
        lo = pd.to_datetime(start) if start is not None else df[date_col].min()
        hi = pd.to_datetime(end) if end is not None else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # Add calendar features in-place
    df = add_calendar_features(df, date_col=date_col, add_cyclic=add_cyclic, inplace=True)

    # Decide which feature columns to use
    if feature_cols is None:
        feats: List[str] = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # Ensure we keep the minimal set of columns needed by downstream code
    keep = sorted(
        set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats)
    )
    df = df[keep]

    return df, feats


# ---------------------------------------------------------------------------
# Station selection helpers
# ---------------------------------------------------------------------------


def select_station_ids(
    obj: Union[pd.DataFrame, Sequence[Hashable]],
    *,
    id_col: Optional[str] = None,
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Hashable]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Hashable], bool]] = None,
) -> List[Hashable]:
    """
    Select a subset of station identifiers using flexible filters.

    This function supports two calling styles:

    1) Passing a DataFrame + ``id_col``::

        select_station_ids(df, id_col="station", prefix="15")

    2) Passing an explicit sequence of ids::

        ids = ["15001", "15002", "32001"]
        select_station_ids(ids, prefix="15")

    Filters are combined with **OR** semantics:

    - Any id matching a given prefix is included.
    - Any id explicitly listed in ``station_ids`` is included.
    - Any id matching the regular expression is included.
    - Any id for which ``custom_filter(id)`` is True is included.

    If no filters are provided, all ids are returned.
    If filters are provided but none of them matches any id, the function
    falls back to returning **all** ids (useful for MDR-like experiments).

    Parameters
    ----------
    obj : DataFrame or sequence
        Either a DataFrame containing the station id column or a sequence
        of station identifiers.
    id_col : str or None, default None
        Name of the station id column when ``obj`` is a DataFrame.
    prefix : iterable of str or None, default None
        One or more prefixes; ids whose string representation starts with any of
        these prefixes will be selected.
    station_ids : iterable or None, default None
        Explicit list of ids to include (intersected with the ids present in
        ``obj`` when a DataFrame is used).
    regex : str or None, default None
        Regular expression; ids whose string representation matches the pattern
        will be included.
    custom_filter : callable or None, default None
        Function ``f(id) -> bool``. If provided, any id for which the function
        returns True will be included.

    Returns
    -------
    list
        List of selected ids, without duplicates, preserving the order in
        the original sequence.
    """
    # Determine the base list of ids
    if isinstance(obj, pd.DataFrame):
        if id_col is None:
            raise ValueError(
                "select_station_ids: 'id_col' must be provided when passing a DataFrame."
            )
        if id_col not in obj.columns:
            raise ValueError(
                f"select_station_ids: column '{id_col}' not found in DataFrame."
            )
        all_ids: List[Hashable] = list(pd.unique(obj[id_col].dropna()))
    else:
        all_ids = list(obj)

    # No filters → return everything
    if prefix is None and station_ids is None and regex is None and custom_filter is None:
        return list(all_ids)

    # Normalize prefixes
    if prefix is not None and isinstance(prefix, str):
        prefix_iter: Optional[List[str]] = [prefix]
    else:
        prefix_iter = list(prefix) if prefix is not None else None

    # Pre-compile regex if needed
    pat = None
    if regex is not None:
        import re

        pat = re.compile(regex)

    selected: List[Hashable] = []

    for sid in all_ids:
        s = str(sid)
        keep = False

        if prefix_iter is not None:
            if any(s.startswith(str(p)) for p in prefix_iter):
                keep = True

        if station_ids is not None and not keep:
            # Only consider ids that exist in the base list
            if sid in station_ids:
                keep = True

        if pat is not None and not keep:
            if pat.match(s):
                keep = True

        if custom_filter is not None and not keep:
            if custom_filter(sid):
                keep = True

        if keep:
            selected.append(sid)

    # Fallback: if no id matched, return all ids
    if not selected:
        selected = list(all_ids)

    # Deduplicate preserving order
    seen = set()
    out: List[Hashable] = []
    for sid in selected:
        if sid in seen:
            continue
        seen.add(sid)
        out.append(sid)

    return out


def filter_by_min_station_rows(
    df: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    min_station_rows: int,
    station_ids: Optional[Iterable[Hashable]] = None,
) -> List[Hashable]:
    """
    Filter station ids by a minimum number of **observed** target rows.

    This helper is meant to be used in MDR-like experiments to ensure that
    only stations with enough non-missing data are evaluated.

    Parameters
    ----------
    df : DataFrame
        Long-format table containing at least ``id_col`` and ``target_col``.
    id_col : str
        Station identifier column.
    target_col : str
        Target variable column. Only non-null values are counted.
    min_station_rows : int
        Minimum number of non-null target rows required for a station to be
        kept.
    station_ids : iterable or None, default None
        Optional subset of station ids to filter. If None, all unique ids
        in ``df[id_col]`` are considered.

    Returns
    -------
    list
        Station ids that meet or exceed the required number of observed rows.
    """
    validate_required_columns(
        df,
        [id_col, target_col],
        context="filter_by_min_station_rows",
    )

    if station_ids is None:
        candidate_ids = df[id_col].dropna().unique().tolist()
    else:
        candidate_ids = list(station_ids)

    # Count observed (non-NaN) target values per station
    obs_counts = (
        df.loc[~df[target_col].isna(), [id_col, target_col]]
        .groupby(id_col)[target_col]
        .size()
        .astype(int)
    )

    keep: List[Hashable] = []
    threshold = int(min_station_rows)

    for sid in candidate_ids:
        if int(obs_counts.get(sid, 0)) >= threshold:
            keep.append(sid)

    return keep


__all__ = [
    "ensure_datetime_naive",
    "add_calendar_features",
    "add_time_features",
    "add_cyclic_doy",
    "build_xyzt_features",
    "validate_required_columns",
    "select_station_ids",
    "filter_by_min_station_rows",
]
