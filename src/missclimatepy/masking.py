
# SPDX-License-Identifier: MIT
"""
missclimatepy.masking
=====================

Missing-data diagnostics and masking utilities for daily station records.

This module focuses on:

* Coverage diagnostics in a fixed date window (MDR-style).
* Gap profiling per station (run-lengths of missing stretches).
* Station × date missingness matrices (0/1).
* One-stop summaries combining coverage and gaps.
* Deterministic random masking for controlled experiments.

All functions operate on long-format ``pandas.DataFrame`` objects and are
schema-agnostic via explicit ``id_col``, ``date_col`` and ``target_col``
arguments.

Design principles
-----------------

* No side effects on the input; functions copy or derive new objects.
* Minimal dependencies: only ``numpy`` and ``pandas`` in addition to
  :mod:`missclimatepy.features`.
* Deterministic behaviour given explicit seeds for masking.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .features import ensure_datetime_naive, validate_required_columns


# ---------------------------------------------------------------------------
# Coverage in a fixed window
# ---------------------------------------------------------------------------


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
    This matches a typical definition of coverage over a fixed evaluation
    period.

    Parameters
    ----------
    df :
        Long-format table with at least (id_col, date_col, target_col).
    id_col, date_col, target_col :
        Column names for station id, timestamp, and target variable.
    start, end :
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
    validate_required_columns(df, [id_col, date_col, target_col])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "station",
                "total_days",
                "observed_days",
                "missing_days",
                "coverage",
                "percent_missing",
            ]
        )

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    if work.empty:
        return pd.DataFrame(
            columns=[
                "station",
                "total_days",
                "observed_days",
                "missing_days",
                "coverage",
                "percent_missing",
            ]
        )

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    if start_ts > end_ts:
        raise ValueError("start must be <= end")

    total_days = int((end_ts - start_ts).days) + 1
    if total_days <= 0:
        raise ValueError("Window [start, end] must contain at least one day.")

    # Restrict to the window
    mask_window = (work[date_col] >= start_ts) & (work[date_col] <= end_ts)
    work = work.loc[mask_window]

    # Distinct days with at least one non-null observation (per station)
    present = work.loc[~work[target_col].isna(), [id_col, date_col]]
    obs_counts = (
        present.drop_duplicates(subset=[id_col, date_col])
        .groupby(id_col)[date_col]
        .size()
        .astype(int)
    )

    stations = (
        work[id_col].dropna().unique().tolist()
        if not work.empty
        else df[id_col].dropna().unique().tolist()
    )
    if not stations:
        return pd.DataFrame(
            columns=[
                "station",
                "total_days",
                "observed_days",
                "missing_days",
                "coverage",
                "percent_missing",
            ]
        )

    rows = []
    for sid in stations:
        observed_days = int(obs_counts.get(sid, 0))
        missing_days = total_days - observed_days
        cov = observed_days / float(total_days)
        pct_miss = 100.0 * (missing_days / float(total_days))
        rows.append(
            {
                "station": sid,
                "total_days": total_days,
                "observed_days": observed_days,
                "missing_days": missing_days,
                "coverage": cov,
                "percent_missing": pct_miss,
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values("percent_missing", ascending=False).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Gap profiles (run-lengths of missing stretches)
# ---------------------------------------------------------------------------


def _run_length_encode_missing(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run-length encode a boolean mask where True marks missing entries.

    Returns
    -------
    values, lengths : np.ndarray, np.ndarray
        values[i] is the boolean for run i; lengths[i] is its length.
    """
    if mask.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=int)

    # Identify run starts
    diff = np.diff(mask.astype(int))
    run_starts = np.concatenate(([0], np.where(diff != 0)[0] + 1))
    run_vals = mask[run_starts]

    # Compute run lengths
    run_lengths = np.diff(np.concatenate((run_starts, [mask.size])))
    return run_vals, run_lengths


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
    validate_required_columns(df, [id_col, date_col, target_col])

    if df.empty:
        return pd.DataFrame(
            columns=["station", "n_gaps", "mean_gap", "max_gap"]
        )

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    if work.empty:
        return pd.DataFrame(
            columns=["station", "n_gaps", "mean_gap", "max_gap"]
        )

    rows: List[Dict] = []
    for sid, sdf in work.groupby(id_col):
        sdf = sdf.sort_values(date_col)

        lo = sdf[date_col].min()
        hi = sdf[date_col].max()
        full_dates = pd.date_range(lo, hi, freq="D")

        # One row per day, mark observed if any non-null
        daily = (
            sdf.loc[~sdf[target_col].isna(), [date_col]]
            .drop_duplicates()
            .set_index(date_col)
        )
        observed = pd.Series(
            False, index=full_dates, name="observed"
        )
        observed.loc[daily.index.intersection(observed.index)] = True

        missing_mask = ~observed.values
        vals, lens = _run_length_encode_missing(missing_mask)

        if lens.size == 0 or not (vals.any()):
            rows.append(
                {"station": sid, "n_gaps": 0, "mean_gap": 0.0, "max_gap": 0}
            )
            continue

        gap_lengths = lens[vals]
        n_gaps = int(gap_lengths.size)
        mean_gap = float(np.mean(gap_lengths))
        max_gap = int(np.max(gap_lengths))
        rows.append(
            {
                "station": sid,
                "n_gaps": n_gaps,
                "mean_gap": mean_gap,
                "max_gap": max_gap,
            }
        )

    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Station × date missingness matrix
# ---------------------------------------------------------------------------


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
    df :
        Long-format table with at least (id_col, date_col, target_col).
    id_col, date_col, target_col :
        Column names.
    start, end :
        Optional inclusive window to clip dates. If provided, all stations are
        evaluated on the **same** [start, end] interval. If not provided,
        each station uses its own min–max span but the output still has a
        common date index given by the union of all spans.
    sort_by_coverage :
        If True, sort stations by descending coverage (more observed first).
    as_uint8 :
        If True, return the underlying matrix as uint8 (0/1) to reduce memory.

    Returns
    -------
    DataFrame
        Index = station, columns = dates (daily), values in {0, 1}.
    """
    validate_required_columns(df, [id_col, date_col, target_col])

    if df.empty:
        return pd.DataFrame()

    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    if work.empty:
        return pd.DataFrame()

    if start is not None and end is not None:
        lo = pd.to_datetime(start)
        hi = pd.to_datetime(end)
    else:
        lo = work[date_col].min()
        hi = work[date_col].max()

    if lo > hi:
        raise ValueError("start must be <= end (or implied min <= max).")

    # Limit to [lo, hi]
    mask_window = (work[date_col] >= lo) & (work[date_col] <= hi)
    work = work.loc[mask_window]

    if work.empty:
        return pd.DataFrame()

    # Build a "present" indicator at daily resolution
    work["present"] = (~work[target_col].isna()).astype(int)
    pivot = (
        work.pivot_table(
            index=id_col,
            columns=date_col,
            values="present",
            aggfunc="max",
            fill_value=0,
        )
        .sort_index(axis=1)
    )

    if pivot.empty:
        return pd.DataFrame()

    # Sort by coverage (more observed first) if requested
    if sort_by_coverage:
        coverage = pivot.mean(axis=1)
        pivot = pivot.loc[coverage.sort_values(ascending=False).index]

    if as_uint8:
        pivot = pivot.astype("uint8")

    return pivot


# ---------------------------------------------------------------------------
# One-stop summary: coverage + gaps
# ---------------------------------------------------------------------------


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
    validate_required_columns(df, [id_col, date_col, target_col])

    if df.empty:
        return pd.DataFrame(
            columns=[
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
        )

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
        work = df[[id_col, date_col, target_col]].copy()
        work[date_col] = ensure_datetime_naive(work[date_col])
        work = work.dropna(subset=[date_col])
        if work.empty:
            cov = pd.DataFrame(
                columns=[
                    "station",
                    "total_days",
                    "observed_days",
                    "missing_days",
                    "coverage",
                    "percent_missing",
                ]
            )
        else:
            rows_cov: List[Dict] = []
            for sid, sdf in work.groupby(id_col):
                sdf = sdf.sort_values(date_col)
                lo = sdf[date_col].min()
                hi = sdf[date_col].max()
                total_days = int((hi - lo).days) + 1
                # Distinct dates with at least one non-null value
                present = sdf.loc[~sdf[target_col].isna(), [date_col]].drop_duplicates()
                observed_days = int(len(present))
                missing_days = total_days - observed_days
                cov_val = observed_days / float(total_days)
                pct_miss = 100.0 * (missing_days / float(total_days))
                rows_cov.append(
                    {
                        "station": sid,
                        "total_days": total_days,
                        "observed_days": observed_days,
                        "missing_days": missing_days,
                        "coverage": cov_val,
                        "percent_missing": pct_miss,
                    }
                )
            cov = pd.DataFrame(rows_cov)

    gaps = gap_profile_by_station(
        df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
    )

    out = cov.merge(gaps, on="station", how="left", validate="one_to_one")
    # For stations missing in gaps (e.g. empty input), fill gap stats with zeros
    for col, val in [("n_gaps", 0), ("mean_gap", 0.0), ("max_gap", 0)]:
        if col in out.columns:
            out[col] = out[col].fillna(val)

    # Column ordering
    cols = [
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
    out = out[cols]
    return out


# ---------------------------------------------------------------------------
# Deterministic random masking per station
# ---------------------------------------------------------------------------


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
    df :
        Input table.
    id_col, date_col, target_col :
        Column names (``date_col`` is kept for symmetry with other helpers).
    percent_to_mask :
        Percentage in [0, 100]. For each station, this fraction of *eligible*
        rows will be set to NaN in ``target_col``.
    random_state :
        Seed for reproducibility (applied at the global level).
    only_with_observation :
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

    validate_required_columns(df, [id_col, target_col])

    if df.empty:
        return df.copy()

    out = df.copy()
    rng = np.random.RandomState(random_state)

    for sid, sdf in out.groupby(id_col):
        # indices of eligible rows in the original DataFrame
        if only_with_observation:
            eligible_idx = sdf.index[~sdf[target_col].isna()].to_numpy()
        else:
            eligible_idx = sdf.index.to_numpy()

        n_eligible = len(eligible_idx)
        if n_eligible == 0:
            continue

        n_mask = int(np.floor(n_eligible * (percent_to_mask / 100.0)))
        if n_mask <= 0:
            continue

        chosen = rng.choice(eligible_idx, size=n_mask, replace=False)
        out.loc[chosen, target_col] = np.nan

    return out


__all__ = [
    "percent_missing_between",
    "gap_profile_by_station",
    "missing_matrix",
    "describe_missing",
    "apply_random_mask_by_station",
]
