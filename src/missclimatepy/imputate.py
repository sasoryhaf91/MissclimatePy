# src/missclimatepy/imputate.py
# SPDX-License-Identifier: MIT
"""
missclimatepy.imputate
======================

Minimal long-format imputation for a *single target* variable using ONLY:
- Spatial coordinates: latitude, longitude, altitude
- Calendar features: year, month, day-of-year (and optional cyclic sin/cos)

For each *selected* station, a **RandomForestRegressor** is trained with:
1) Either the K nearest spatial neighbors (haversine over (lat, lon)) or
   *all other* stations if `k_neighbors=None`, and
2) **Full inclusion of the target station's valid rows** (no percentage control).

Key policy: MDR by *observed target rows*
-----------------------------------------
A station is *eligible for imputation* only if its **count of observed (non-NaN)
target rows within [start, end]** is ≥ `max(1826, min_station_rows or 0)`.
Stations below this threshold are returned as:
- full daily grid in the window,
- observed values preserved,
- remaining gaps left as NaN,
- `source` set to "observed" where present and NaN otherwise,
i.e., **no imputation** is performed.

Output (minimal tidy)
---------------------
Exactly these columns, in order:
[station, date, latitude, longitude, altitude, <target>, source]
with `source` ∈ {"observed", "imputed"}.

Persistence
-----------
Use the new parameters `save_path`, `save_format`, `save_index`, and `save_partitions`
to write results automatically in CSV or Parquet. If `save_partitions=True`, one file
per station is written (suffix `_station=<ID>` for CSV; directory partitioning for Parquet).

Example
-------
>>> from missclimatepy.imputate import impute_dataset
>>> from missclimatepy.evaluate import RFParams
>>> out = impute_dataset(
...     data=df,
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     start="1981-01-01", end="2023-12-31",
...     k_neighbors=20,
...     rf_params=RFParams(n_estimators=300, max_depth=30, n_jobs=-1, random_state=42),
...     min_station_rows=None,   # MDR uses 1826 by default
...     show_progress=True,
...     save_path="outputs/tmin_imputed.parquet",  # autosave
...     save_format="auto",                        # infer from extension
...     save_partitions=False,                     # single file
... )
>>> out.columns.tolist()
['station','date','latitude','longitude','altitude','tmin','source']
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union, Literal

#import os
from pathlib import Path

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
        return x  # transparent iterator


# ------------------------- internal: persistence helpers ------------------------- #

def _infer_format_from_path(path: str) -> Literal["csv", "parquet"]:
    p = path.lower()
    if p.endswith(".parquet"):
        return "parquet"
    if p.endswith(".csv") or p.endswith(".csv.gz") or p.endswith(".csv.bz2") or p.endswith(".csv.xz") or p.endswith(".csv.zip"):
        return "csv"
    # Default to CSV if unclear (user can override with save_format)
    return "csv"  # conservative


def _ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, dest: Path, *, index: bool) -> None:
    _ensure_parent_dir(dest)
    # compression inferred by pandas from suffix; dtype-safe already
    df.to_csv(dest, index=index)


def _write_parquet(df: pd.DataFrame, dest: Path, *, index: bool) -> None:
    _ensure_parent_dir(dest)
    df.to_parquet(dest, index=index)


def _write_partitions(
    df: pd.DataFrame,
    *,
    station_col: str,
    base_path: Path,
    fmt: Literal["csv", "parquet"],
    index: bool,
) -> None:
    """
    Write one file (CSV/Parquet) per station. For Parquet, writes into
    base_path / f"station={sid}/part.parquet" (Hive-like partitions).
    For CSV, writes base_path with suffix *_station=<sid>.csv[.gz].
    """
    if fmt == "parquet":
        # Partitioned folders: base/station=<sid>/part.parquet
        for sid, sdf in df.groupby(station_col, sort=False):
            part_dir = base_path / f"station={sid}"
            _ensure_parent_dir(part_dir / "part.parquet")
            sdf.to_parquet(part_dir / "part.parquet", index=index)
    else:
        # CSV per station with suffix
        base = str(base_path)
        # Keep any compression extension if present
        # e.g., "out.csv.gz" → "out_station=S123.csv.gz"
        for sid, sdf in df.groupby(station_col, sort=False):
            if base.lower().endswith(".csv"):
                out_name = base[:-4] + f"_station={sid}.csv"
            elif any(base.lower().endswith(ext) for ext in (".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")):
                # split at ".csv" and keep the rest (e.g., ".gz")
                pos = base.lower().rfind(".csv")
                out_name = base[:pos] + f"_station={sid}" + base[pos:]
            else:
                # no extension -> add .csv
                out_name = base + f"_station={sid}.csv"
            dest = Path(out_name)
            _ensure_parent_dir(dest)
            sdf.to_csv(dest, index=index)


def _save_result_df(
    df: pd.DataFrame,
    *,
    path: Optional[str],
    fmt: Literal["csv", "parquet", "auto"],
    index: bool,
    partition: bool,
    station_col: str,
) -> None:
    """
    Save the imputed DataFrame if `path` is provided. No-op if `path` is None.
    """
    if path is None:
        return

    base = Path(path)
    # Decide format
    if fmt == "auto":
        fmt_resolved = _infer_format_from_path(str(base))
    else:
        fmt_resolved = fmt

    if fmt_resolved not in ("csv", "parquet"):
        raise ValueError(f"Unsupported save_format '{fmt}'. Use 'csv', 'parquet', or 'auto'.")

    # Basic schema validation (columns are required by contract)
    required = {"station", "date", "latitude", "longitude", "altitude", "source"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Result schema missing required columns for saving. "
            f"Expected to include: {sorted(required)}; got: {list(df.columns)[:10]}..."
        )

    if partition:
        _write_partitions(df, station_col=station_col, base_path=base, fmt=fmt_resolved, index=index)
        return

    # Single file
    if fmt_resolved == "parquet":
        _write_parquet(df, base, index=index)
    else:
        _write_csv(df, base, index=index)


# ------------------------------ public imputation API ------------------------------ #

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
    # window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection (optional, OR semantics)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # neighbors
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    # MDR (minimum observed rows within the window)
    min_station_rows: Optional[int] = None,
    # model
    rf_params: Optional[Union[RFParams, Dict]] = None,
    # logging
    show_progress: bool = False,
    # persistence
    save_path: Optional[str] = None,
    save_format: Literal["csv", "parquet", "auto"] = "auto",
    save_index: bool = False,
    save_partitions: bool = False,
) -> pd.DataFrame:
    """
    Impute the target time series for *selected* stations over [start, end]
    and return a minimal long-format DataFrame. Optionally save the result.

    MDR (eligibility)
    -----------------
    A station is imputed only if the number of **observed (non-NaN) target rows**
    within [start, end] is ≥ `max(1826, min_station_rows or 0)`. Otherwise,
    the station is returned observed-only (no imputation), preserving gaps.

    Parameters
    ----------
    data : DataFrame
        Long-format input with at least:
        [id_col, date_col, lat_col, lon_col, alt_col, target_col].
        The target column may contain NaNs (gaps).
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, coordinates and target variable.
    start, end : str or None
        Inclusive analysis window. If None, inferred from `data`.
    add_cyclic : bool
        If True, adds sin/cos of day-of-year to the feature set.
    feature_cols : list[str] or None
        Custom feature list. Default: [lat, lon, alt, year, month, doy] (+ cyc).
    prefix, station_ids, regex, custom_filter :
        Optional filters to select which stations to process (OR semantics).
        If none provided, all stations are considered.
    k_neighbors : int or None
        If provided and `neighbor_map` is None, build a KNN haversine map over
        station medians and train with those neighbors (excluding the target).
        If None, train with *all other* stations.
    neighbor_map : dict or None
        Overrides `k_neighbors`. Dict {station_id -> list_of_neighbor_ids}.
    min_station_rows : int or None
        Optional stricter MDR. Actual threshold is:
        `MDR_MIN = max(1826, min_station_rows or 0)`.
        MDR counts **observed target rows** within [start, end].
    rf_params : RFParams | dict | None
        RandomForest hyperparameters. Missing fields use defaults.
    show_progress : bool
        If True, prints compact progress lines (requires `tqdm`).

    save_path : str or None
        If provided, write the resulting DataFrame to this path.
        - If `save_format="auto"`, the format is inferred by extension:
          *.parquet → Parquet; *.csv[.gz|.bz2|.xz|.zip] → CSV.
        - Parent directories are created if needed.
    save_format : {"csv","parquet","auto"}
        File format to write. Use "auto" to infer from extension.
    save_index : bool
        Whether to write the DataFrame index.
    save_partitions : bool
        If True, write **one file per station**.
        - Parquet: base_dir / "station=<ID>/part.parquet"
        - CSV: base_name → base_name + "_station=<ID>.csv[.gz]"

    Returns
    -------
    DataFrame
        Minimal long-format table with **exactly** the columns:
        [station, date, latitude, longitude, altitude, <target_col>, source]
        where `source` ∈ {"observed", "imputed"} for processed stations.

    Notes
    -----
    - Only the target variable is modeled and returned.
    - Coordinates in the output are station-level medoids (median per station).
    - Determinism depends on `rf_params.random_state`.
    - Saving is best-effort and raises a clear ValueError on unsupported formats.
    """
    # ---- basic validation & windowing ---------------------------------------
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    df = data.copy()
    df[date_col] = _ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    lo = pd.to_datetime(start) if start else df[date_col].min()
    hi = pd.to_datetime(end) if end else df[date_col].max()
    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # ---- feature engineering -------------------------------------------------
    df = _add_time_features(df, date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)
    # Ensure unique feature names to avoid duplicate-column issues downstream
    feats = list(dict.fromkeys(feats))

    # ---- station selection (OR semantics; default: all) ----------------------
    all_ids = df[id_col].dropna().unique().tolist()
    chosen: List[Union[str, int]] = []

    if prefix is not None:
        if isinstance(prefix, str):
            prefix = [prefix]
        for p in prefix:
            chosen.extend([s for s in all_ids if str(s).startswith(str(p))])

    if station_ids is not None:
        chosen.extend(list(station_ids))

    if regex is not None:
        import re
        pat = re.compile(regex)
        chosen.extend([s for s in all_ids if pat.match(str(s))])

    if custom_filter is not None:
        chosen.extend([s for s in all_ids if custom_filter(s)])

    if not chosen:
        chosen = all_ids

    # Stable order, unique
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # ---- MDR filter (by *observed* target rows in window) -------------------
    base_mdr = 1826
    user_mdr = int(min_station_rows) if (min_station_rows is not None) else 0
    MDR_MIN = max(base_mdr, user_mdr)

    observed_mask = ~df[target_col].isna()
    obs_counts = df.loc[observed_mask].groupby(id_col, sort=False)[target_col].size().astype(int)

    eligible = [s for s in stations if int(obs_counts.get(s, 0)) >= MDR_MIN]
    if show_progress and len(eligible) != len(stations):
        skipped = [s for s in stations if s not in eligible]
        tqdm.write(
            f"MDR (observed≥{MDR_MIN}) filter: {len(stations)} → {len(eligible)} stations "
            f"(skipped {len(skipped)})"
        )
    stations = eligible

    if not stations:
        empty = pd.DataFrame(columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"])
        # still honor save_path if user wants an empty scaffold written
        _save_result_df(
            empty,
            path=save_path,
            fmt=save_format,
            index=save_index,
            partition=save_partitions,
            station_col=id_col,
        )
        return empty

    # ---- neighbor map / training pool ---------------------------------------
    if neighbor_map is not None:
        nmap = neighbor_map
    elif k_neighbors is not None:
        nmap = _build_neighbor_map_haversine(
            df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k=int(k_neighbors), include_self=False
        )
    else:
        nmap = None

    # RF hyperparameters
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # Canonical date grid & station medoids for output stability
    full_dates = pd.date_range(lo, hi, freq="D")
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median(numeric_only=True)
        .rename(columns={lat_col: "latitude", lon_col: "longitude", alt_col: "altitude"})
    )

    # Global mask of valid rows for training (features + target present)
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # ---- main loop -----------------------------------------------------------
    iterator = tqdm(stations, desc="Imputing stations", unit="st") if show_progress else stations
    out_blocks: List[pd.DataFrame] = []

    for sid in iterator:
        # Base grid (full window) + station medoid metadata
        lat0 = float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan
        lon0 = float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan
        alt0 = float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan

        grid = pd.DataFrame({
            date_col: full_dates,
            id_col: sid,
            "latitude": lat0,
            "longitude": lon0,
            "altitude": alt0,
        })

        # Attach observed target (restricted to window)
        st_obs = df.loc[df[id_col] == sid, [date_col, target_col]]
        merged = grid.merge(st_obs, on=date_col, how="left")

        # Recompute features on the full grid
        merged = _add_time_features(merged, date_col, add_cyclic=add_cyclic)

        # Build training pool: neighbors or all-other stations (exclude target)
        is_target_mask = (df[id_col] == sid)
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            pool_mask = df[id_col].isin(neigh_ids) & (~is_target_mask) & valid_mask_global
        else:
            pool_mask = (~is_target_mask) & valid_mask_global

        train_pool = df.loc[pool_mask, feats + [target_col]]

        # FULL inclusion of the target station's valid rows (no percentage control)
        st_valid = df.loc[is_target_mask & valid_mask_global, feats + [target_col]]
        train_df = pd.concat([train_pool, st_valid], axis=0, ignore_index=True)

        if train_df.empty:
            # Guard: return observed-only rows as minimal block
            y_obs = merged[target_col].to_numpy()
            out = pd.DataFrame({
                id_col: sid,
                "date": merged[date_col].to_numpy(),
                "latitude": merged["latitude"].astype(float).to_numpy(),
                "longitude": merged["longitude"].astype(float).to_numpy(),
                "altitude": merged["altitude"].astype(float).to_numpy(),
                target_col: y_obs,
                "source": np.where(np.isnan(y_obs), np.nan, "observed").astype("object"),
            })
            out_blocks.append(out)
            if show_progress:
                tqdm.write(f"Station {sid}: empty training set → observed-only.")
            continue

        # Fit and predict
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        model = RandomForestRegressor(**rf_kwargs)
        model.fit(X_train, y_train)

        X_full = merged[feats].to_numpy(copy=False)
        y_hat = model.predict(X_full)

        # Observed wins; fill remaining with predictions
        y_obs = merged[target_col].to_numpy()
        mask_nan = np.isnan(y_obs)
        filled = y_obs.copy()
        filled[mask_nan] = y_hat[mask_nan]

        # Minimal long-format block
        out = pd.DataFrame({
            id_col: sid,
            "date": merged[date_col].to_numpy(),
            "latitude": merged["latitude"].astype(float).to_numpy(),
            "longitude": merged["longitude"].astype(float).to_numpy(),
            "altitude": merged["altitude"].astype(float).to_numpy(),
            target_col: filled,
            "source": np.where(mask_nan, "imputed", "observed").astype("object"),
        })
        out_blocks.append(out)

        if show_progress:
            n_obs = int((out["source"] == "observed").sum())
            n_imp = int((out["source"] == "imputed").sum())
            k_text = len(nmap.get(sid, [])) if nmap is not None else "all"
            tqdm.write(
                f"Station {sid}: window={len(out):,}  observed={n_obs:,}  imputed={n_imp:,}  "
                f"k={k_text}"
            )

    # Stack, sort, return *minimal* schema only
    result = pd.concat(out_blocks, axis=0, ignore_index=True)
    result = result.sort_values([id_col, "date"], kind="mergesort").reset_index(drop=True)

    # Ensure exact column order
    result = result[[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]]

    # --------- optional saving ---------
    _save_result_df(
        result,
        path=save_path,
        fmt=save_format,
        index=save_index,
        partition=save_partitions,
        station_col=id_col,
    )

    return result
