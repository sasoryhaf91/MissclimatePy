# SPDX-License-Identifier: MIT
"""
missclimatepy.masking
=====================

Utilities to explore and quantify missingness at daily resolution.

Key functions
-------------
- percent_missing_between: missing % per station relative to an expected daily span.
- gap_profile_by_station : length of longest gap, number of gaps, mean gap length.
- missing_matrix         : 0/1 matrix for heatmap (date rows × station cols).
"""
from __future__ import annotations
from typing import Optional, Dict
#from typing import Tuple
#import numpy as np
import pandas as pd


def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_localize(None)
    return s


def percent_missing_between(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute missing percentage per station using an *expected daily span*.
    That is, pct_missing = (#expected_days - #valid_observed_days) / #expected_days.

    Parameters
    ----------
    df : DataFrame
        Long format with station, date, and target.
    id_col, date_col, target_col : str
        Column names for station id, date and target variable.
    start, end : str or None
        Optional inclusive analysis window. If None, uses global min/max date in df.

    Returns
    -------
    DataFrame with columns:
        station, start, end, expected_days, observed_valid, missing, pct_missing,
        first_obs, last_obs
    """
    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = _ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    span_lo = pd.to_datetime(start) if start else work[date_col].min()
    span_hi = pd.to_datetime(end) if end else work[date_col].max()

    # restrict to span
    work = work[(work[date_col] >= span_lo) & (work[date_col] <= span_hi)]

    # valid obs by station/day
    valid = work.dropna(subset=[target_col]).drop_duplicates([id_col, date_col])

    # expected daily days per station
    # (we avoid creating full daily ranges per station for memory; compute via length)
    expected_days = (span_hi - span_lo).days + 1

    agg = valid.groupby(id_col)[date_col].size().rename("observed_valid").to_frame()
    agg["expected_days"] = expected_days
    agg["missing"] = agg["expected_days"] - agg["observed_valid"]
    agg["pct_missing"] = agg["missing"] / agg["expected_days"]
    # first/last observed inside span
    bounds = valid.groupby(id_col)[date_col].agg(["min", "max"]).rename(
        columns={"min": "first_obs", "max": "last_obs"}
    )
    out = agg.join(bounds, how="left").reset_index().rename(columns={id_col: "station"})
    out.insert(1, "start", span_lo.normalize())
    out.insert(2, "end", span_hi.normalize())
    return out


def gap_profile_by_station(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Describe gap structure per station:
    longest gap length (days), number of gaps, and mean gap length.
    """
    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = _ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    def _gaps(sub: pd.DataFrame) -> Dict[str, float]:
        sub = sub.sort_values(date_col)
        if sub.empty:
            return {"longest_gap": 0, "n_gaps": 0, "mean_gap": 0.0}

        # build consecutive daily index so gaps are explicit
        full_idx = pd.date_range(sub[date_col].min(), sub[date_col].max(), freq="D")
        full = pd.DataFrame(index=full_idx)
        full["is_valid"] = False
        # mark valid days
        full.loc[sub[date_col].values, "is_valid"] = sub[target_col].notna().to_numpy()

        # run-length encode on the validity flag
        run_id = (full["is_valid"] != full["is_valid"].shift()).cumsum()
        # use as_index=False to avoid reset_index() name collisions
        groups = (
            full.assign(run_id=run_id)
                .groupby(["run_id", "is_valid"], as_index=False)
                .size()
        )
        gaps = groups.loc[groups["is_valid"] == False, "size"]  # noqa: E712
        if gaps.empty:
            return {"longest_gap": 0, "n_gaps": 0, "mean_gap": 0.0}
        return {
            "longest_gap": int(gaps.max()),
            "n_gaps": int(gaps.size),
            "mean_gap": float(gaps.mean()),
        }

    out = (
        work.groupby(id_col, group_keys=False)
            .apply(_gaps)
            .apply(pd.Series)
            .reset_index()
            .rename(columns={id_col: "station"})
    )
    return out


def missing_matrix(
    df: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    target_col: str,
    stations: Optional[list] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Produce a 0/1 matrix for heatmaps (rows = dates, columns = stations),
    where 1 = missing, 0 = observed.
    """
    work = df[[id_col, date_col, target_col]].copy()
    work[date_col] = _ensure_datetime_naive(work[date_col])
    work = work.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else work[date_col].min()
        hi = pd.to_datetime(end) if end else work[date_col].max()
        work = work[(work[date_col] >= lo) & (work[date_col] <= hi)]

    if stations is not None:
        work = work[work[id_col].isin(stations)]

    # pivot to observed flag, then invert
    obs = work.assign(obs=work[target_col].notna()).pivot_table(
        index=date_col, columns=id_col, values="obs", aggfunc="max"
    )
    miss = (~obs.fillna(False)).astype(int)
    return miss.sort_index()
