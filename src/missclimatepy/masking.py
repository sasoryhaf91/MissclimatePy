# SPDX-License-Identifier: MIT
"""
missclimatepy.masking
=====================

Missing-data exploration and masking utilities for MissClimatePy.

This module provides small, composable tools to:

- quantify missingness per station over a target period,
- profile gaps (consecutive missing runs) per station,
- generate a station × date missingness matrix (daily cadence),
- optionally apply deterministic masking to simulate missingness.

Design notes
------------

* Column-name agnostic: callers must pass the names for station, date,
  and target columns.
* All operations are based on **daily** time steps by construction:
  functions that "fill ranges" assume daily cadence.
* Datetime columns are parsed once and made tz-naive via the shared
  :func:`missclimatepy.features.ensure_datetime_naive` helper.
* Random masking is reproducible via a `random_state` seed.

Typical uses
------------

These utilities are meant for:

- exploratory analysis of station-wise coverage and gap structure,
- constructing reproducible missingness patterns for MDR experiments,
- preparing station subsets for subsequent imputation/evaluation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .features import ensure_datetime_naive, validate_required_columns


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _daily_date_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Inclusive daily date range [start, end]."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if pd.isna(start) or pd.isna(end):
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    if end < start:
        start, end = end, start
    return pd.date_range(start, end, freq="D")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def percent_missing_between(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Percentage of missing values per station in a fixed date window.

    The denominator is the full number of calendar days in [start, end],
    regardless of whether a station has observations for every day.
    This matches the typical definition of coverage over a fixed
    evaluation period.

    Parameters
    ----------
    df : DataFrame
        Long-format table with at least (id_col, date_col, target_col).
    id_col, date_col, target_col : str
        Column names for station id, timestamp, and target variable.
    start, end : str
        Inclusive boundaries for the evaluation window.

    Returns
    -------
    DataFrame
        Columns:

        - ``station``         : station identifier (from ``id_col``),
        - ``total_days``      : number of calendar days in [start, end],
        - ``observed_days``   : distinct days with at least one non-null
                                observation for ``target_col``,
        - ``missing_days``    : ``total_days - observed_days``,
        - ``coverage``        : ``observed_days / total_days``,
        - ``percent_missing`` : 100 * ``missing_days / total_days``.

        Sorted by ``percent_missing`` descending (worst coverage first).
    """
    validate_required_columns(df, [id_col, date_col, target_col], context="percent_missing_between")

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    window = _daily_date_range(pd.to_datetime(start), pd.to_datetime(end))
    if window.empty:
        raise ValueError("Empty date window in percent_missing_between.")

    # For each station, count distinct observed days within [start, end].
    in_win = (work[date_col] >= window[0]) & (work[date_col] <= window[-1])
    observed = (
        work.loc[in_win & ~work[target_col].isna(), [id_col, date_col]]
        .drop_duplicates()
        .groupby(id_col)[date_col]
        .nunique()
        .rename("observed_days")
        .astype(int)
    )

    stations = work[id_col].dropna().unique()
    total_days = int(len(window))

    out = (
        pd.DataFrame({id_col: stations})
        .merge(observed, on=id_col, how="left")
        .fillna({"observed_days": 0})
    )

    out["total_days"] = total_days
    out["missing_days"] = out["total_days"] - out["observed_days"]
    out["coverage"] = out["observed_days"] / out["total_days"]
    out["percent_missing"] = 100.0 * (out["missing_days"] / out["total_days"])

    out = out.sort_values("percent_missing", ascending=False).reset_index(drop=True)
    out = out.rename(columns={id_col: "station"})

    return out[
        [
            "station",
            "total_days",
            "observed_days",
            "missing_days",
            "coverage",
            "percent_missing",
        ]
    ]


def gap_profile_by_station(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Profile consecutive missing runs per station over each station's own span.

    For each station:

    1. Build the inclusive daily date range from its min(date) to max(date).
    2. Mark a day as **observed** if ``target_col`` is non-null at least once
       on that day.
    3. Run-length encode the resulting boolean series and summarize only the
       missing runs.

    Returns
    -------
    DataFrame
        Columns:

        - ``station`` : station identifier,
        - ``n_gaps``  : number of missing runs,
        - ``mean_gap``: mean length of missing runs (days),
        - ``max_gap`` : maximum gap length (days).

        Stations with no gaps have ``n_gaps = 0``, ``mean_gap = 0.0``,
        ``max_gap = 0``.
    """
    validate_required_columns(df, [id_col, date_col, target_col], context="gap_profile_by_station")

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    out_rows = []

    for sid, sub in work.groupby(id_col):
        dmin = sub[date_col].min()
        dmax = sub[date_col].max()
        full = _daily_date_range(dmin, dmax)

        if full.empty:
            out_rows.append({"station": sid, "n_gaps": 0, "mean_gap": 0.0, "max_gap": 0})
            continue

        # Set of days with at least one non-null observation for target
        observed_days = set(
            sub.loc[~sub[target_col].isna(), date_col].dt.normalize().unique()
        )

        # Boolean valid vector aligned to 'full'
        valid = np.fromiter(
            (d.normalize() in observed_days for d in full),
            dtype=bool,
            count=len(full),
        )

        if valid.size == 0:
            out_rows.append({"station": sid, "n_gaps": 0, "mean_gap": 0.0, "max_gap": 0})
            continue

        # Run-length encoding on 'valid'
        changes = np.diff(valid.astype(np.int8), prepend=valid[0])
        group_ids = np.cumsum(changes != 0)

        lengths = np.bincount(group_ids)
        first_indices = np.r_[0, np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1]
        group_state = valid[first_indices]

        missing_lengths = lengths[~group_state] if lengths.size and (~group_state).any() else np.array([], dtype=int)

        if missing_lengths.size == 0:
            out_rows.append({"station": sid, "n_gaps": 0, "mean_gap": 0.0, "max_gap": 0})
        else:
            out_rows.append(
                {
                    "station": sid,
                    "n_gaps": int(missing_lengths.size),
                    "mean_gap": float(np.mean(missing_lengths)),
                    "max_gap": int(np.max(missing_lengths)),
                }
            )

    if not out_rows:
        return pd.DataFrame(
            columns=["station", "n_gaps", "mean_gap", "max_gap"], dtype=object
        )

    return (
        pd.DataFrame(out_rows)[["station", "n_gaps", "mean_gap", "max_gap"]]
        .sort_values("station")
        .reset_index(drop=True)
    )


def missing_matrix(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    sort_by_coverage: bool = True,
    as_uint8: bool = True,
) -> pd.DataFrame:
    """
    Build a station × date matrix with 1 for observed and 0 for missing.

    Parameters
    ----------
    df : DataFrame
        Long-format table with at least (id_col, date_col, target_col).
    id_col, date_col, target_col : str
        Column names.
    start, end : str or None
        Optional inclusive window to clip dates. If provided, all stations are
        evaluated on the **same** [start, end] interval. If not provided,
        each station uses its own [min(date), max(date)] span.
    sort_by_coverage : bool, default True
        If True, sort stations by descending coverage (more observed first).
    as_uint8 : bool, default True
        If True, return the underlying matrix as uint8 (0/1) to reduce memory.

    Returns
    -------
    DataFrame
        Index = station, columns = dates (daily), values in {0, 1}.
    """
    validate_required_columns(df, [id_col, date_col, target_col], context="missing_matrix")

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    if start is not None and end is not None:
        win = _daily_date_range(pd.to_datetime(start), pd.to_datetime(end))
        if win.empty:
            raise ValueError("Empty date window in missing_matrix.")

        observed = (
            work.loc[
                (work[date_col] >= win[0])
                & (work[date_col] <= win[-1])
                & ~work[target_col].isna(),
                [id_col, date_col],
            ]
            .drop_duplicates()
        )
        observed["val"] = 1
        mat = (
            observed.pivot_table(
                index=id_col,
                columns=date_col,
                values="val",
                fill_value=0,
                aggfunc="max",
            )
            .reindex(columns=win, fill_value=0)
        )
    else:
        # station-specific span: for each station, we normalize dates to
        # daily resolution and then pivot.
        work = work.assign(val=(~work[target_col].isna()).astype(np.int8))
        day = work.copy()
        day[date_col] = day[date_col].dt.normalize()
        observed = day.groupby([id_col, date_col])["val"].max().reset_index()
        mat = observed.pivot(index=id_col, columns=date_col, values="val").fillna(0)

    if as_uint8:
        mat = mat.astype("uint8")

    if sort_by_coverage and not mat.empty:
        coverage = mat.mean(axis=1)
        mat = mat.loc[coverage.sort_values(ascending=False).index]

    mat.index.name = "station"
    return mat


def describe_missing(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-stop summary of missingness per station combining coverage and gaps.

    If ``start`` and ``end`` are provided, coverage is computed over that fixed
    window. Otherwise, coverage uses each station's own min–max span.

    Returns
    -------
    DataFrame
        Columns:

        - ``station``         : station identifier,
        - ``total_days``      : calendar days in the coverage window,
        - ``observed_days``   : days with at least one non-null observation,
        - ``missing_days``    : ``total_days - observed_days``,
        - ``coverage``        : ``observed_days / total_days``,
        - ``percent_missing`` : 100 * ``missing_days / total_days``,
        - ``n_gaps``          : number of missing runs,
        - ``mean_gap``        : mean gap length (days),
        - ``max_gap``         : maximum gap length (days).
    """
    if start is not None and end is not None:
        cov = percent_missing_between(
            df,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            start=start,
            end=end,
        )
    else:
        # Coverage over each station's native span
        validate_required_columns(df, [id_col, date_col, target_col], context="describe_missing")
        work = df[[id_col, date_col, target_col]].copy()
        work[date_col] = ensure_datetime_naive(work[date_col])
        work = work.dropna(subset=[date_col])

        rows = []
        for sid, sub in work.groupby(id_col):
            dmin = sub[date_col].min()
            dmax = sub[date_col].max()
            full = _daily_date_range(dmin, dmax)
            if full.empty:
                rows.append(
                    {
                        "station": sid,
                        "total_days": 0,
                        "observed_days": 0,
                        "missing_days": 0,
                        "coverage": 0.0,
                        "percent_missing": 100.0,
                    }
                )
                continue

            obs_days = set(
                sub.loc[~sub[target_col].isna(), date_col].dt.normalize().unique()
            )
            observed_days = len(obs_days)
            total_days = len(full)
            missing_days = total_days - observed_days
            coverage = observed_days / total_days

            rows.append(
                {
                    "station": sid,
                    "total_days": total_days,
                    "observed_days": observed_days,
                    "missing_days": missing_days,
                    "coverage": coverage,
                    "percent_missing": 100.0 * (missing_days / total_days),
                }
            )

        cov = pd.DataFrame(rows)

    gaps = gap_profile_by_station(
        df, id_col=id_col, date_col=date_col, target_col=target_col
    )
    out = cov.merge(gaps, on="station", how="left")

    out[["n_gaps", "mean_gap", "max_gap"]] = out[["n_gaps", "mean_gap", "max_gap"]].fillna(
        {"n_gaps": 0, "mean_gap": 0.0, "max_gap": 0}
    )

    return (
        out[
            [
                "station",
                "total_days",
                "observed_days",
                "missing_days",
                "coverage",
                "percent_missing",
                "n_gaps",
                "mean_gap",
                "max_gap",
            ]
        ]
        .sort_values(["percent_missing", "station"], ascending=[False, True])
        .reset_index(drop=True)
    )


def apply_random_mask_by_station(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,  # kept for symmetry, not used internally yet
    target_col: str,
    percent_to_mask: float,
    random_state: int = 42,
    only_with_observation: bool = True,
) -> pd.DataFrame:
    """
    Deterministically mask a percentage of target values per station.

    This helper is convenient for MDR-like experiments where you want to
    control the included proportion of valid rows in training or simulate
    additional missingness for evaluation.

    Parameters
    ----------
    df : DataFrame
        Input table.
    id_col, date_col, target_col : str
        Column names (``date_col`` is kept for symmetry with other helpers).
    percent_to_mask : float
        Percentage in [0, 100]. For each station, this fraction of *eligible*
        rows will be set to NaN in ``target_col``.
    random_state : int, default 42
        Seed for reproducibility (applied at the global level).
    only_with_observation : bool, default True
        If True, sample the mask only among rows that are currently *observed*
        (non-NaN). If False, rows already missing can also be selected
        (no-ops).

    Returns
    -------
    DataFrame
        A copy of the input with masked values applied to ``target_col``.
    """
    if percent_to_mask < 0 or percent_to_mask > 100:
        raise ValueError("percent_to_mask must be in [0, 100].")

    validate_required_columns(df, [id_col, target_col], context="apply_random_mask_by_station")
    out = df.copy()

    # Global RNG; we derive station-specific RNGs from it to keep
    # reproducibility but allow per-station independence if needed.
    master_rng = np.random.RandomState(random_state)

    for _, idx in out.groupby(id_col).groups.items():
        idx = pd.Index(idx)

        if only_with_observation:
            eligible = idx[out.loc[idx, target_col].notna().values]
        else:
            eligible = idx

        if eligible.size == 0:
            continue

        k = int(np.floor(eligible.size * (percent_to_mask / 100.0)))
        if k <= 0:
            continue

        # Station-level deterministic draw
        local_seed = master_rng.randint(0, 2**31 - 1)
        local_rng = np.random.RandomState(local_seed)
        chosen = local_rng.choice(eligible.values, size=k, replace=False)

        out.loc[chosen, target_col] = np.nan

    return out


__all__ = [
    "percent_missing_between",
    "gap_profile_by_station",
    "missing_matrix",
    "describe_missing",
    "apply_random_mask_by_station",
]
