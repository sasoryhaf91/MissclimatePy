# SPDX-License-Identifier: MIT
"""
missclimatepy.imputate
======================

Final imputation routine that produces *completed* time series for selected
stations using ONLY spatial coordinates (latitude, longitude, altitude) and
calendar features (year, month, day-of-year; optionally cyclic sin/cos).

Design
------
• One RandomForestRegressor is trained per target station.
• Training pool = K nearest *eligible* donors (by haversine on lat/lon), or
  all other *eligible* donors if `k_neighbors=None`.
• Target station rows are **fully included** in training (100% of valid rows).
• Minimum Data Requirement (MDR) is enforced by **total row count** in the
  window, not by the number of valid rows in the target:
      effective_mdr = max(min_station_rows, 1826)
  Targets below MDR are **skipped**. Donors below MDR are **excluded**.

Selection of targets (OR semantics)
-----------------------------------
Use any combination of:
  - station_ids (explicit list)
  - prefix (string or list of strings; startswith)
  - regex (regular expression)
  - custom_filter (callable(id)->bool)
If none is provided, all stations in the window are considered.

Output
------
A long-format DataFrame spanning [start, end] at daily frequency for the
**selected & eligible** targets, with the original columns plus `source`
in {"observed","imputed"}.

Backward compatibility
----------------------
`include_target_pct` and `include_target_seed` are accepted but **ignored**
(the algorithm always uses full inclusion of target valid rows).

Example
-------
>>> out = impute_dataset(
...     df,
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     start="1981-01-01", end="2023-12-31",
...     k_neighbors=20,
...     rf_params=RFParams(n_estimators=300, max_depth=30, n_jobs=-1, random_state=42),
...     show_progress=True,
... )
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluate import (
    RFParams,
    _require_columns,
    _ensure_datetime_naive,
    _add_time_features,
    _build_neighbor_map_haversine,
)

try:  # optional progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Hard floor MDR (~5 years of daily data)
_HARD_FLOOR_MDR = 1826


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def impute_dataset(
    data: pd.DataFrame,
    *,
    # schema
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # period
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # selection of TARGET stations (OR semantics)
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    prefix: Optional[Iterable[str]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # donors (neighbors)
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    # model
    rf_params: Optional[Union[RFParams, Dict[str, Any]]] = None,
    # MDR
    min_station_rows: int = _HARD_FLOOR_MDR,
    # progress & logging
    show_progress: bool = False,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    # deprecated/ignored (kept for back-compat with existing calls/tests)
    include_target_pct: Optional[float] = None,   # ignored
    include_target_seed: Optional[int] = None,    # ignored
) -> pd.DataFrame:
    """
    Impute missing `target_col` values for the selected & MDR-eligible stations.
    The function returns a completed long-format table where:
      • Observed values are preserved.
      • Missing values are filled with model predictions.
      • A new `source` column indicates "observed" or "imputed".

    Parameters
    ----------
    data : DataFrame
        Long-format table with at least:
        [id_col, date_col, lat_col, lon_col, alt_col, target_col].
        Any extra columns are carried to the output.
    start, end : str or None
        Inclusive analysis window. If None, inferred from data.
    add_cyclic : bool
        If True, adds sin/cos transforms of day-of-year as features.
    feature_cols : list[str] or None
        Custom feature set. If None, defaults to:
        [lat_col, lon_col, alt_col, "year", "month", "doy"] (+ "doy_sin", "doy_cos" if `add_cyclic`).
    station_ids, prefix, regex, custom_filter :
        Target-station selectors (OR semantics). If none given, considers all.
    k_neighbors : int or None
        If provided (and neighbor_map is None), builds a KNN map (haversine on per-station
        median lat/lon) and trains only with those neighbors (excluding the target).
        If None, uses all *eligible* donors except the target.
    neighbor_map : dict or None
        Overrides `k_neighbors`. Dict {station_id -> list_of_neighbor_ids}.
    rf_params : RFParams | dict | None
        RandomForestRegressor hyperparameters. Missing fields use defaults.
    min_station_rows : int
        Soft MDR requested by the caller. Effective MDR is:
        `effective_mdr = max(min_station_rows, 1826)`.
        MDR is evaluated on **total row count** within the window (observed + missing).
    show_progress : bool
        If True, prints a compact line per station.
    progress_callback : callable(dict) -> None or None
        If provided, called once per station with payload (counts, timing, donors).
    log_csv : str or None
        If provided, appends one row per station to this CSV.
    flush_every : int
        Flush frequency for `log_csv`.
    include_target_pct, include_target_seed :
        Accepted but ignored (always full inclusion of target valid rows).

    Returns
    -------
    DataFrame
        Completed table for the selected & eligible targets, spanning [start, end],
        with original columns plus a `source` column ∈ {"observed","imputed"}.
    """
    # Validate schema
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # Dates & clip window
    df = data.copy()
    df[date_col] = _ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
    else:
        lo = df[date_col].min()
        hi = df[date_col].max()

    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # Features
    df_feat = _add_time_features(df, date_col, add_cyclic=add_cyclic)
    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # Window index
    full_index = pd.date_range(lo, hi, freq="D")

    # Global valid rows for training (features + target present)
    valid_mask_global = ~df_feat[feats + [target_col]].isna().any(axis=1)

    # ---------------- MDR by TOTAL ROW COUNT (observed + missing) ---------------- #
    total_counts = df_feat.groupby(id_col).size().astype(int)
    effective_mdr = max(int(min_station_rows), _HARD_FLOOR_MDR)
    eligible_ids = set(total_counts[total_counts >= effective_mdr].index.tolist())
    # ---------------------------------------------------------------------------- #

    # All station ids (stable)
    all_ids = df_feat[id_col].dropna().unique().tolist()
    all_ids.sort(key=lambda x: str(x))

    # Select targets (OR semantics)
    chosen: List[Union[str, int]] = []
    if station_ids is not None:
        chosen.extend(list(station_ids))
    if prefix is not None:
        if isinstance(prefix, str):
            prefix = [prefix]
        for p in prefix:
            chosen.extend([s for s in all_ids if str(s).startswith(str(p))])
    if regex is not None:
        import re
        pat = re.compile(regex)
        chosen.extend([s for s in all_ids if pat.match(str(s))])
    if custom_filter is not None:
        chosen.extend([s for s in all_ids if custom_filter(s)])
    if not chosen:
        chosen = all_ids  # default: impute all

    # Unique & stable
    seen = set()
    target_stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # Neighbor map limited to eligible donors
    if neighbor_map is not None:
        nmap = neighbor_map
        used_k = None
    elif k_neighbors is not None:
        df_for_neighbors = df_feat[df_feat[id_col].isin(list(eligible_ids))]
        if df_for_neighbors.empty:
            nmap = None
            used_k = None
        else:
            nmap = _build_neighbor_map_haversine(
                df_for_neighbors,
                id_col=id_col, lat_col=lat_col, lon_col=lon_col,
                k=int(k_neighbors), include_self=False
            )
            used_k = int(k_neighbors)
    else:
        nmap = None
        used_k = None

    # RF hyperparameters
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # Logging buffers
    header_written: Dict[str, bool] = {}
    pending_rows: List[Dict[str, Any]] = []

    iterator = tqdm(target_stations, desc="Imputing selected stations", unit="st") if show_progress else target_stations

    # Output blocks
    out_blocks: List[pd.DataFrame] = []

    # Preserve original column order + add `source`
    original_cols = list(df.columns)
    if "source" in original_cols:
        original_cols.remove("source")
    final_cols = original_cols + ["source"]

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()
        is_target = (df_feat[id_col] == sid)

        # Skip ineligible targets
        if sid not in eligible_ids:
            _emit_progress(progress_callback, show_progress, sid, t0, len(full_index),
                           0, used_k, donors=0, reason=f"target_below_mdr<{effective_mdr}",
                           total_rows=int(total_counts.get(sid, 0)))
            continue

        # Donors (eligible only, excluding target)
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            donor_ids = [nid for nid in neigh_ids if (nid != sid) and (nid in eligible_ids)]
        else:
            donor_ids = [nid for nid in all_ids if (nid != sid) and (nid in eligible_ids)]

        # Assemble training pool from eligible donors (valid rows only)
        if donor_ids:
            donor_mask = df_feat[id_col].isin(donor_ids)
            train_pool = df_feat.loc[donor_mask & valid_mask_global, feats + [target_col]]
        else:
            train_pool = pd.DataFrame(columns=feats + [target_col])

        # FULL inclusion of target valid rows
        target_valid_mask = is_target & valid_mask_global
        target_valid = df_feat.loc[target_valid_mask, feats + [target_col]]

        if train_pool.empty and target_valid.empty:
            _emit_progress(progress_callback, show_progress, sid, t0, len(full_index),
                           0, used_k, donors=0, reason="empty_train")
            continue

        # Align and stack
        if not train_pool.empty:
            target_valid = target_valid.loc[:, train_pool.columns]
            train_df = pd.concat([train_pool, target_valid], axis=0, ignore_index=True)
        else:
            train_df = target_valid.copy()

        # Fit model
        model = RandomForestRegressor(**rf_kwargs)
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        model.fit(X_train, y_train)

        # Predict across full window grid with station medoids for coords
        med = df.loc[df[id_col] == sid, [lat_col, lon_col, alt_col]].median(numeric_only=True)
        st_grid = pd.DataFrame({
            id_col: sid,
            date_col: full_index,
            lat_col: float(med.get(lat_col, np.nan)) if not med.empty else np.nan,
            lon_col: float(med.get(lon_col, np.nan)) if not med.empty else np.nan,
            alt_col: float(med.get(alt_col, np.nan)) if not med.empty else np.nan,
        })
        st_grid_feat = _add_time_features(st_grid, date_col, add_cyclic=add_cyclic)
        X_grid = st_grid_feat[feats].to_numpy(copy=False)
        y_hat = model.predict(X_grid)

        # Merge observed + fill with predictions
        st_obs = df[df[id_col] == sid][[id_col, date_col, target_col]]
        st_full = st_grid.merge(st_obs, on=[id_col, date_col], how="left", suffixes=("", "_obs"))

        is_obs = st_full[target_col].notna()
        st_full[target_col] = np.where(is_obs, st_full[target_col], y_hat).astype(float)
        st_full["source"] = np.where(is_obs, "observed", "imputed").astype("object")

        # Carry extra original columns (if any)
        extra_cols = [c for c in original_cols if c not in st_full.columns]
        if extra_cols:
            st_full = st_full.merge(
                df[df[id_col] == sid][[id_col, date_col] + extra_cols],
                on=[id_col, date_col],
                how="left",
                suffixes=("", "_orig"),
            )
            for c in extra_cols:
                if c not in st_full.columns:
                    st_full[c] = np.nan

        # Ensure final schema & order
        for c in original_cols:
            if c not in st_full.columns:
                st_full[c] = np.nan
        st_full = st_full[final_cols]

        out_blocks.append(st_full)

        # Progress / logs
        _emit_progress(progress_callback, show_progress, sid, t0, len(st_full),
                       len(train_df), used_k, donors=len(donor_ids), reason="ok")
        if log_csv is not None:
            _append_to_csv(
                {
                    "station": sid,
                    "rows_train": int(len(train_df)),
                    "rows_window": int(len(st_full)),
                    "used_k_neighbors": used_k,
                    "donors": len(donor_ids),
                    "seconds": float(pd.Timestamp.utcnow().timestamp() - t0),
                },
                path=log_csv,
                header_written=_header_state,
                buffer=_csv_buffer,
                flush_every=flush_every,
            )

    if not out_blocks:
        # No eligible targets selected → empty but well-formed output
        return pd.DataFrame(columns=final_cols)

    result = pd.concat(out_blocks, axis=0, ignore_index=True)
    result.sort_values([id_col, date_col], inplace=True, kind="mergesort")
    result.reset_index(drop=True, inplace=True)
    return result


# --------------------------------------------------------------------------- #
# CSV logging helpers
# --------------------------------------------------------------------------- #

_header_state: Dict[str, bool] = {}
_csv_buffer: List[Dict[str, Any]] = []


def _append_to_csv(
    row: Dict[str, Any],
    *,
    path: str,
    header_written: Dict[str, bool],
    buffer: List[Dict[str, Any]],
    flush_every: int,
) -> None:
    """Buffer CSV rows and flush every `flush_every` appends."""
    if path is None:
        return
    buffer.append(row)
    if len(buffer) >= int(flush_every):
        _flush_csv_rows(buffer, path=path, header_written=header_written)


def _flush_csv_rows(rows: List[Dict[str, Any]], *, path: str, header_written: Dict[str, bool]) -> None:
    """Flush buffered rows to CSV with header written only once."""
    if not rows:
        return
    tmp = pd.DataFrame(rows)
    write_header = not header_written.get(path, False)
    tmp.to_csv(path, mode="a", index=False, header=write_header)
    header_written[path] = True
    rows.clear()


# --------------------------------------------------------------------------- #
# Progress emission
# --------------------------------------------------------------------------- #

def _emit_progress(
    callback: Optional[Callable[[Dict[str, Any]], None]],
    show: bool,
    station: Union[str, int],
    t0: float,
    rows_window: int,
    rows_train: int,
    used_k: Optional[int],
    *,
    donors: int,
    reason: str,
    total_rows: Optional[int] = None,
) -> None:
    """Emit a compact progress payload and optional console line."""
    sec = float(pd.Timestamp.utcnow().timestamp() - t0)
    payload = {
        "station": station,
        "seconds": sec,
        "rows_window": int(rows_window),
        "rows_train": int(rows_train),
        "used_k_neighbors": used_k,
        "donors": int(donors),
        "reason": reason,
    }
    if total_rows is not None:
        payload["rows_total_in_window"] = int(total_rows)

    if callback:
        callback(payload)

    if show:
        base = (
            f"{station}: donors={payload['donors']:,} train={payload['rows_train']:,} "
            f"→ window={payload['rows_window']:,} (k={payload['used_k_neighbors']}) [{reason}]"
        )
        if total_rows is not None:
            base += f" total={total_rows}"
        tqdm.write(base)
