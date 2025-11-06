# src/missclimatepy/masking.py
# SPDX-License-Identifier: MIT
"""
Missing-data exploration and masking utilities for MissclimatePy.

This module provides small, composable tools to:
- quantify missingness per station over a target period,
- profile gaps (consecutive missing runs) per station,
- generate a 0/1 missingness matrix (stations × dates),
- optionally apply deterministic masking to simulate missingness.

All functions are column-name agnostic: callers must pass the names for
station id, date, and target columns.

Design notes
------------
- We never rely on deprecated pandas timezone dtype checks.
- Date columns are parsed once and made tz-naive if needed.
- We treat the imputation problem as *daily* by construction. Functions that
  build “full ranges” assume daily cadence.
- All random masking is reproducible via a `random_state` seed.

Key functions
-------------
percent_missing_between     : % missing per station between start and end (inclusive).
gap_profile_by_station      : number/mean/max length of consecutive missing runs.
missing_matrix              : stations × dates uint8 matrix (1=observed, 0=missing).
describe_missing            : consolidated per-station summary (coverage + gaps).
apply_random_mask_by_station: simulate missingness per station at a target rate.
"""

from __future__ import annotations

#from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _ensure_datetime_naive(series: pd.Series) -> pd.Series:
    """Parse datetimes and drop timezone if present (no deprecated dtype checks)."""
    s = pd.to_datetime(series, errors="coerce")
    if isinstance(s.dtype, pd.DatetimeTZDtype):  # no is_datetime64tz_dtype
        s = s.dt.tz_localize(None)
    return s


def _daily_date_range(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Inclusive daily date range [start, end]."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if pd.isna(start) or pd.isna(end):
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    if end < start:
        start, end = end, start
    return pd.date_range(start, end, freq="D")


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")


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
    Percentage of missing values per station between two dates (inclusive).

    The denominator is the full number of calendar days in [start, end],
    regardless of whether a station has observations for every day. This
    matches the intended definition for coverage over a fixed evaluation
    window.

    Parameters
    ----------
    df : DataFrame
        Long table with at least (id_col, date_col, target_col).
    id_col, date_col, target_col : str
        Column names.
    start, end : str
        Inclusive boundaries for the evaluation window.

    Returns
    -------
    DataFrame
        Columns: [station, total_days, observed_days, missing_days, coverage, percent_missing]
        Sorted by percent_missing descending (worst first).
    """
    _require_columns(df, [id_col, date_col, target_col])

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = _ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    window = _daily_date_range(pd.to_datetime(start), pd.to_datetime(end))
    if window.empty:
        raise ValueError("Empty date window for percent_missing_between.")

    # For each station, count distinct observed days within [start, end].
    # We consider a day "observed" for the station if the target is not NA for that date.
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
    return out[["station", "total_days", "observed_days", "missing_days", "coverage", "percent_missing"]]


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
      - Build the inclusive daily date range from its min(date) to max(date).
      - Mark a day as valid if target is non-null on that day (any record).
      - Compute run-length encoding and summarize missing runs only.

    Returns
    -------
    DataFrame
        [station, n_gaps, mean_gap, max_gap]
        where gaps are lengths (in days) of consecutive missing sequences.

    Notes
    -----
    - Uses numpy operations to avoid name collisions like the earlier
      "cannot insert valid, already exists" error.
    """
    _require_columns(df, [id_col, date_col, target_col])

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = _ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    out_rows = []
    for sid, sub in work.groupby(id_col):
        dmin = sub[date_col].min()
        dmax = sub[date_col].max()
        full = _daily_date_range(dmin, dmax)
        if full.empty:
            out_rows.append({"station": sid, "n_gaps": 0, "mean_gap": 0.0, "max_gap": 0})
            continue

        # Set of dates with at least one non-null observation for target
        observed_days = set(
            sub.loc[~sub[target_col].isna(), date_col].dt.normalize().unique()
        )
        # Boolean valid vector aligned to 'full'
        valid = np.fromiter((d in observed_days for d in full.normalize()), dtype=bool, count=len(full))

        # Run-length encoding on 'valid'
        # Identify change points
        if valid.size == 0:
            out_rows.append({"station": sid, "n_gaps": 0, "mean_gap": 0.0, "max_gap": 0})
            continue

        changes = np.diff(valid.astype(np.int8), prepend=valid[0])
        # Build group ids whenever the state changes
        group_ids = np.cumsum(changes != 0)

        # Aggregate lengths per (group_id, state)
        # We only keep groups where state == False (missing)
        # Using numpy bincounts per group id:
        # length per group:
        lengths = np.bincount(group_ids)
        # state per group (take the first element of each group)
        # To get state per group, take valid at the first index of each group.
        # First indices of groups are where group_ids changes:
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

    return pd.DataFrame(out_rows)[["station", "n_gaps", "mean_gap", "max_gap"]].sort_values("station").reset_index(drop=True)


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
        Long table with at least (id_col, date_col, target_col).
    id_col, date_col, target_col : str
        Column names.
    start, end : str or None
        Optional inclusive window to clip dates. If not provided, each station
        uses its own [min(date), max(date)] span.
    sort_by_coverage : bool
        If True, sort stations by descending coverage (more observed first).
    as_uint8 : bool
        If True, return matrix as uint8 to reduce memory.

    Returns
    -------
    DataFrame
        Index = station, columns = dates (daily), values in {0,1}.
    """
    _require_columns(df, [id_col, date_col, target_col])

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = _ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    if start is not None and end is not None:
        win = _daily_date_range(pd.to_datetime(start), pd.to_datetime(end))
        if win.empty:
            raise ValueError("Empty date window for missing_matrix.")
        # observed indicator per (station, date) in window
        observed = (
            work.loc[(work[date_col] >= win[0]) & (work[date_col] <= win[-1]) & ~work[target_col].isna(), [id_col, date_col]]
            .drop_duplicates()
        )
        observed["val"] = 1
        mat = (
            observed.pivot_table(index=id_col, columns=date_col, values="val", fill_value=0, aggfunc="max")
            .reindex(columns=win, fill_value=0)
        )
    else:
        # station-specific span
        work = work.assign(val=(~work[target_col].isna()).astype(np.int8))
        # For multiple rows per day, aggregate by max (any observed -> 1)
        day = work.copy()
        day[date_col] = day[date_col].dt.normalize()
        observed = (
            day.groupby([id_col, date_col])["val"].max().reset_index()
        )
        mat = observed.pivot(index=id_col, columns=date_col, values="val").fillna(0)

    # dtype & ordering
    if as_uint8:
        mat = mat.astype("uint8")

    if sort_by_coverage:
        coverage = mat.mean(axis=1)
        mat = mat.loc[coverage.sort_values(ascending=False).index]

    # Friendly station index name
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

    If `start` and `end` are provided, coverage is computed over that fixed
    window. Otherwise, coverage uses each station's own min–max span.

    Returns
    -------
    DataFrame
        [station, total_days, observed_days, missing_days, coverage, percent_missing,
         n_gaps, mean_gap, max_gap]
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
        # station-specific span coverage
        _require_columns(df, [id_col, date_col, target_col])
        work = df[[id_col, date_col, target_col]].copy()
        work[date_col] = _ensure_datetime_naive(work[date_col])
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
            obs_days = set(sub.loc[~sub[target_col].isna(), date_col].dt.normalize().unique())
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

    gaps = gap_profile_by_station(df, id_col=id_col, date_col=date_col, target_col=target_col)
    out = cov.merge(gaps, on="station", how="left")
    out[["n_gaps", "mean_gap", "max_gap"]] = out[["n_gaps", "mean_gap", "max_gap"]].fillna({"n_gaps": 0, "mean_gap": 0.0, "max_gap": 0})
    return out[
        ["station", "total_days", "observed_days", "missing_days", "coverage", "percent_missing", "n_gaps", "mean_gap", "max_gap"]
    ].sort_values(["percent_missing", "station"], ascending=[False, True]).reset_index(drop=True)


def apply_random_mask_by_station(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
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
        Column names.
    percent_to_mask : float
        Percentage in [0,100]. For each station, this fraction of *eligible*
        rows will be set to NaN in `target_col`.
    random_state : int
        Seed for reproducibility (applied per station).
    only_with_observation : bool
        If True, mask is sampled only among rows currently *observed* (non-NaN).
        If False, mask can also pick rows that are already missing (no-ops).

    Returns
    -------
    DataFrame
        A copy of the input with masked values applied to `target_col`.
    """
    if percent_to_mask < 0 or percent_to_mask > 100:
        raise ValueError("percent_to_mask must be in [0, 100].")

    _require_columns(df, [id_col, date_col, target_col])
    out = df.copy()

    rng = np.random.RandomState(random_state)
    for sid, sub_idx in out.groupby(id_col).groups.items():
        idx = pd.Index(sub_idx)
        if only_with_observation:
            eligible = idx[out.loc[idx, target_col].notna().values]
        else:
            eligible = idx  # may include already-missing rows

        if eligible.size == 0:
            continue

        k = int(np.floor(eligible.size * (percent_to_mask / 100.0)))
        if k <= 0:
            continue

        # station-level deterministic draw: advance RNG with station hash
        local_rng = np.random.RandomState(rng.randint(0, 2**31 - 1))
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
