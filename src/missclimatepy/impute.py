# src/missclimatepy/impute.py
# SPDX-License-Identifier: MIT
"""
missclimatepy.impute
====================

Minimal long-format imputation for a *single target* variable using ONLY:
- Spatial coordinates: latitude, longitude, altitude
- Calendar features: year, month, day-of-year (and optional cyclic sin/cos)

For each *selected* station, a **RandomForestRegressor** is trained with:
1) Either the K nearest spatial neighbors (haversine over (lat, lon)) or
   *all other* stations if ``k_neighbors=None``, and
2) A controllable fraction of the target station's own valid rows
   (via ``include_target_pct``), from 0% (LOSO) to 100%.

Station eligibility (min_station_rows)
--------------------------------------
A station is imputed only if the number of **observed (non-NaN) target rows**
within [start, end] is ≥ ``min_station_rows`` when this parameter is provided
and > 0. If ``min_station_rows`` is ``None`` or 0, *no minimum* is enforced
(all selected stations are eligible).

Information visibility (include_target_pct)
-------------------------------------------
For each eligible station, the fraction of its own observed rows that is used
in the training set is controlled by ``include_target_pct``:

- ``include_target_pct is None or >= 100``:
    All observed rows of the station are included in the training set
    (original behavior).
- ``0 <= include_target_pct < 100``:
    Only that percentage (floored) of the station's observed rows is included
    in the training set. Rows are selected deterministically in chronological
    order (oldest to newest). Remaining observed rows are kept in the output as
    ``source="observed"`` but are *not* seen by the model during training.
- ``include_target_pct == 0``:
    Extreme LOSO scenario where the model never sees the target station's
    history and relies only on neighbors (when available).

Output (minimal tidy)
---------------------
Exactly these columns, in order:

    [station, date, latitude, longitude, altitude, <target>, source]

with ``source ∈ {"observed", "imputed"}``.

Persistence
-----------
Use the parameters ``save_path``, ``save_format``, ``save_index`` and
``save_partitions`` to write results automatically in CSV or Parquet. If
``save_partitions=True``, one file per station is written (suffix
``_station=<ID>`` for CSV; directory partitioning for Parquet).

Example
-------
>>> from missclimatepy.impute import impute_dataset
>>> from missclimatepy.evaluate import RFParams
>>> out = impute_dataset(
...     data=df,
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     start="1981-01-01", end="2023-12-31",
...     k_neighbors=20,
...     rf_params=RFParams(n_estimators=300, max_depth=30, n_jobs=-1, random_state=42),
...     min_station_rows=365,
...     include_target_pct=50.0,
...     show_progress=True,
...     save_path="outputs/tmin_imputed.parquet",
...     save_format="auto",
...     save_partitions=False,
... )
>>> out.columns.tolist()
['station', 'date', 'latitude', 'longitude', 'altitude', 'tmin', 'source']
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union, Literal
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
    """
    Infer file format from the output path extension.

    - *.parquet           -> "parquet"
    - *.csv[.gz|.bz2|...] -> "csv"
    - otherwise           -> "csv" (conservative default)
    """
    p = path.lower()
    if p.endswith(".parquet"):
        return "parquet"
    if (
        p.endswith(".csv")
        or p.endswith(".csv.gz")
        or p.endswith(".csv.bz2")
        or p.endswith(".csv.xz")
        or p.endswith(".csv.zip")
    ):
        return "csv"
    # Default to CSV if unclear (user can override with save_format)
    return "csv"


def _ensure_parent_dir(path: Path) -> None:
    """
    Create parent directories for `path` if they do not exist.
    """
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, dest: Path, *, index: bool) -> None:
    """
    Write a DataFrame to CSV (compression inferred from suffix).
    """
    _ensure_parent_dir(dest)
    df.to_csv(dest, index=index)


def _write_parquet(df: pd.DataFrame, dest: Path, *, index: bool) -> None:
    """
    Write a DataFrame to Parquet.
    """
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
    Write one file (CSV/Parquet) per station.

    - Parquet: base_path / f"station=<sid>/part.parquet" (Hive-like layout).
    - CSV:     base_name + "_station=<sid>.csv[.gz]" for each station.
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
            lower = base.lower()
            if lower.endswith(".csv"):
                out_name = base[:-4] + f"_station={sid}.csv"
            elif any(lower.endswith(ext) for ext in (".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")):
                # split at ".csv" and keep the rest (e.g., ".gz")
                pos = lower.rfind(".csv")
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

    This helper:
    - infers format if `fmt="auto"`,
    - validates basic schema (required columns),
    - writes a single file or one file per station.
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
        raise ValueError(
            f"Unsupported save_format '{fmt}'. Use 'csv', 'parquet', or 'auto'."
        )

    # Basic schema validation:
    # - station_col is the id column (e.g., "station" or a custom id_col)
    # - latitude/longitude/altitude/source are canonical in the output
    required = {station_col, "date", "latitude", "longitude", "altitude", "source"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Result schema missing required columns for saving. "
            f"Expected to include: {sorted(required)}; got: {list(df.columns)[:10]}..."
        )

    if partition:
        _write_partitions(
            df,
            station_col=station_col,
            base_path=base,
            fmt=fmt_resolved,
            index=index,
        )
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
    # how much of the target station history the model can see
    include_target_pct: Optional[float] = None,
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

    Station eligibility (min_station_rows)
    --------------------------------------
    A station is imputed only if the number of **observed (non-NaN) target rows**
    within [start, end] is ≥ ``min_station_rows`` when this parameter is
    provided and > 0. If ``min_station_rows`` is ``None`` or 0, *no minimum*
    is enforced (all selected stations are eligible).

    Information visibility (include_target_pct)
    -------------------------------------------
    For each eligible station, the fraction of its own observed rows that is
    used in the training set is controlled by ``include_target_pct``:

    - ``include_target_pct is None or >= 100``:
        All observed rows of the station are included in the training set
        (original behavior).
    - ``0 <= include_target_pct < 100``:
        Only that percentage (floored) of the station's observed rows is
        included in the training set. Rows are selected deterministically in
        chronological order (oldest to newest).
    - ``include_target_pct == 0``:
        No local rows are used in training (extreme LOSO); imputation is
        driven solely by neighboring stations when available.

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
        Custom feature list. Default: [lat, lon, alt, year, month, doy] (+ cyclic).
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
        Minimum number of observed target rows within the window required for
        a station to be imputed. If None or 0, no minimum is enforced.
    include_target_pct : float in [0, 100] or None
        Fraction of each station's *observed* target rows allowed into the
        training set. Values ≥ 100 are treated as full inclusion.

    rf_params : RFParams | dict | None
        RandomForest hyperparameters. Missing fields use defaults.
    show_progress : bool
        If True, prints compact progress lines (requires `tqdm`).

    save_path : str or None
        If provided, write the resulting DataFrame to this path.
        - If ``save_format="auto"``, the format is inferred by extension:
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
        where ``source ∈ {"observed", "imputed"}``.
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

    # ---- MDR filter (by *observed* target rows in window, user-controlled) ---
    if min_station_rows is not None and min_station_rows > 0:
        observed_mask = ~df[target_col].isna()
        obs_counts = (
            df.loc[observed_mask]
            .groupby(id_col, sort=False)[target_col]
            .size()
            .astype(int)
        )
        eligible = [s for s in stations if int(obs_counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress and len(eligible) != len(stations):
            skipped = [s for s in stations if s not in eligible]
            tqdm.write(
                f"MDR filter (observed≥{min_station_rows}): "
                f"{len(stations)} → {len(eligible)} stations (skipped {len(skipped)})"
            )
        stations = eligible

    if not stations:
        empty = pd.DataFrame(
            columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]
        )
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
            df,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=int(k_neighbors),
            include_self=False,
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

        grid = pd.DataFrame(
            {
                date_col: full_dates,
                id_col: sid,
                "latitude": lat0,
                "longitude": lon0,
                "altitude": alt0,
            }
        )

        # Attach observed target (restricted to window)
        st_obs = df.loc[df[id_col] == sid, [date_col, target_col]]
        merged = grid.merge(st_obs, on=date_col, how="left")

        # Recompute features on the full grid
        merged = _add_time_features(merged, date_col, add_cyclic=add_cyclic)

        # Build training pool: neighbors or all-other stations (exclude target)
        is_target_mask = df[id_col] == sid
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            pool_mask = df[id_col].isin(neigh_ids) & (~is_target_mask) & valid_mask_global
        else:
            pool_mask = (~is_target_mask) & valid_mask_global

        train_pool = df.loc[pool_mask, feats + [target_col]]

        # Fallback: si no hay vecinos válidos, usar todas las demás estaciones
        # con valores observados del target dentro de la ventana.
        if train_pool.empty:
            alt_pool_mask = (~is_target_mask) & (~df[target_col].isna()) & valid_mask_global
            train_pool = df.loc[alt_pool_mask, feats + [target_col]]

        # Target-station valid rows (features + target present + date para ordenar)
        st_valid = df.loc[
            is_target_mask & valid_mask_global,
            [date_col] + feats + [target_col],
        ]

        # -------- include_target_pct control (deterministic, by date) ---------
        if include_target_pct is not None:
            if include_target_pct < 0:
                raise ValueError("include_target_pct must be >= 0 or None.")
            if include_target_pct >= 100.0:
                # full inclusion (original behavior)
                pass
            else:
                n_local = len(st_valid)
                if n_local > 0:
                    st_valid = st_valid.sort_values(date_col)
                    k_sel = int(np.floor(n_local * (include_target_pct / 100.0)))
                    if k_sel > 0:
                        st_valid = st_valid.iloc[:k_sel, :]
                    else:
                        # k_sel == 0 → no local rows used in training (extreme LOSO)
                        st_valid = st_valid.iloc[0:0, :]

        # Drop date_col: the model only sees features + target
        if not st_valid.empty:
            st_valid_for_train = st_valid[feats + [target_col]]
        else:
            st_valid_for_train = st_valid  # empty DataFrame

        # Concatenate training pool + selected target-station rows
        train_df = pd.concat([train_pool, st_valid_for_train], axis=0, ignore_index=True)

        if train_df.empty:
            # Guard: no training information at all → return observed-only rows
            y_obs = merged[target_col].to_numpy()
            mask_nan = np.isnan(y_obs)

            source = np.empty(len(y_obs), dtype="object")
            source[mask_nan] = np.nan
            source[~mask_nan] = "observed"

            out = pd.DataFrame(
                {
                    id_col: sid,
                    "date": merged[date_col].to_numpy(),
                    "latitude": merged["latitude"].astype(float).to_numpy(),
                    "longitude": merged["longitude"].astype(float).to_numpy(),
                    "altitude": merged["altitude"].astype(float).to_numpy(),
                    target_col: y_obs,
                    "source": source,
                }
            )
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

        # Build 'source' as object array:
        if merged[target_col].isna().all():
            # Caso LOSO puro: la estación no tiene ningún valor observado,
            # por lo que todas las filas son imputadas.
            source = np.full(len(merged), "imputed", dtype="object")
        else:
            source = np.empty(len(y_obs), dtype="object")
            source[mask_nan] = "imputed"
            source[~mask_nan] = "observed"

        # Minimal long-format block
        out = pd.DataFrame(
            {
                id_col: sid,
                "date": merged[date_col].to_numpy(),
                "latitude": merged["latitude"].astype(float).to_numpy(),
                "longitude": merged["longitude"].astype(float).to_numpy(),
                "altitude": merged["altitude"].astype(float).to_numpy(),
                target_col: filled,
                "source": source,
            }
        )
        out_blocks.append(out)

        if show_progress:
            n_obs = int((out["source"] == "observed").sum())
            n_imp = int((out["source"] == "imputed").sum())
            k_text = len(nmap.get(sid, [])) if nmap is not None else "all"
            tqdm.write(
                f"Station {sid}: window={len(out):,}  observed={n_obs:,}  "
                f"imputed={n_imp:,}  k={k_text}, "
                f"include_target_pct={include_target_pct if include_target_pct is not None else 100}"
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


__all__ = [
    "impute_dataset",
]
