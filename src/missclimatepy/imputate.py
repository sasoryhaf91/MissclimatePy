# SPDX-License-Identifier: MIT
"""
missclimatepy.imputate
======================

Minimal long-format imputation for a *single target* variable using ONLY:
- Spatial coordinates: latitude, longitude, altitude
- Calendar features: year, month, day-of-year (and optional cyclic sin/cos)

For each *selected* station, we train **one RandomForestRegressor** using:
1) Either the K nearest spatial neighbors (by haversine over (lat, lon)) or
   *all other* stations if `k_neighbors=None`, and
2) An optional controlled inclusion (`include_target_pct`) of the target
   station's own valid rows (to adapt locally).

The function returns a **minimal tidy table** with exactly:
[station, date, latitude, longitude, altitude, <target>, source]
where `source` ∈ {"observed", "imputed"}.

Key policies
------------
- **MDR (Minimum Data Requirement)** per station:
  A station is eligible only if its **original row count in [start, end]**
  is >= max(1826, min_station_rows if provided). Esta política usa el
  número de filas originales en la ventana (independiente del missing del target).
- **Single-target discipline**:
  Se entrena y se imputa exclusivamente `target_col`. Ninguna otra variable
  entra al objetivo de entrenamiento ni a la salida.
- **Schema-agnostic**:
  Todos los nombres de columna son parámetros de la función.

Typical usage
-------------
>>> from missclimatepy.imputate import impute_dataset
>>> from missclimatepy.evaluate import RFParams
>>> out = impute_dataset(
...     data=df,
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     start="1981-01-01", end="2023-12-31",
...     k_neighbors=20, include_target_pct=80.0,
...     rf_params=RFParams(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42),
...     show_progress=True,
... )
>>> out.columns
Index(['station','date','latitude','longitude','altitude','tmin','source'], dtype='object')

Reproducibility & scope
-----------------------
- Determinism via `rf_params.random_state` y muestreo con `include_target_seed`.
- Sin dependencias a datos externos. No requiere covariables climáticas adicionales.
- Pensado para *gap-filling* de series diarias con cobertura ≥ ~5 años por estación.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

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

try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # fallback: transparent iterator


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
    # neighbors & optional leakage from target
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,     # 0..95 (observed rows from the target added to train)
    include_target_seed: int = 42,
    # station MDR
    min_station_rows: Optional[int] = None,
    # model
    rf_params: Optional[Union[RFParams, Dict]] = None,
    # logging
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Impute the target time series for *selected* stations over [start, end]
    and return a minimal long-format DataFrame.

    Eligibility (MDR)
    -----------------
    A station is **processed** only if its original number of rows within
    [start, end] is >= max(1826, min_station_rows if provided). Stations
    not meeting MDR are silently skipped.

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
        Optional filters to select which stations to impute (OR semantics).
        If none provided, all stations are considered.
    k_neighbors : int or None
        If provided and `neighbor_map` is None, build a KNN haversine map over
        station medians and train with those neighbors (excluding the target).
        If None, train with *all other* stations.
    neighbor_map : dict or None
        Overrides `k_neighbors`. Dict {station_id -> list_of_neighbor_ids}.
    include_target_pct : float
        Percent (0..95) of the target station’s valid rows to include in training.
        Use 0 for strict LOSO-style exclusion; use a positive value to adapt locally.
    include_target_seed : int
        RNG seed for sampling target rows when `include_target_pct` > 0.
    min_station_rows : int or None
        Optional hint for MDR. Actual threshold is:
        MDR_MIN = max(1826, min_station_rows if provided).
    rf_params : RFParams | dict | None
        RandomForest hyperparameters. Missing fields use defaults.
    show_progress : bool
        If True, prints compact progress lines (requires `tqdm`).

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
    - Determinism depends on `rf_params.random_state` and `include_target_seed`.
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

    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # ---- MDR filter (per station, based on original rows in window) ----------
    base_mdr = 1826
    user_mdr = int(min_station_rows) if (min_station_rows is not None) else 0
    MDR_MIN = max(base_mdr, user_mdr)

    rows_per_station = df.groupby(id_col, sort=False)[date_col].size().astype(int)
    stations = [s for s in stations if int(rows_per_station.get(s, 0)) >= MDR_MIN]

    if not stations:
        return pd.DataFrame(columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"])

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
        .median()
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

        # Recompute features on the full grid (fast; deterministic)
        merged = _add_time_features(merged, date_col, add_cyclic=add_cyclic)

        # Build training pool: neighbors or all-other stations
        is_target_mask = (df[id_col] == sid)
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            pool_mask = df[id_col].isin(neigh_ids) & (~is_target_mask) & valid_mask_global
        else:
            pool_mask = (~is_target_mask) & valid_mask_global

        train_pool = df.loc[pool_mask, feats + [target_col]]

        # Optional inclusion (leakage) from the target’s valid rows
        st_valid = df.loc[is_target_mask & valid_mask_global, feats + [target_col]]
        pct = max(0.0, min(float(include_target_pct), 95.0))
        if pct > 0.0 and not st_valid.empty:
            n_take = int(np.ceil(len(st_valid) * (pct / 100.0)))
            leakage = st_valid.sample(n=n_take, random_state=int(include_target_seed))
            train_df = pd.concat([train_pool, leakage], axis=0, ignore_index=True)
        else:
            train_df = train_pool

        if train_df.empty:
            # No training data → cannot impute; return observed-only rows as minimal block
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
                tqdm.write(f"Station {sid}: empty training pool → observed-only (no imputation).")
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
                f"k={k_text}  incl={pct:.1f}%"
            )

    # Stack, sort, return *minimal* schema only
    result = pd.concat(out_blocks, axis=0, ignore_index=True)
    result = result.sort_values([id_col, "date"], kind="mergesort").reset_index(drop=True)

    # Ensure exact column order
    result = result[[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]]
    return result

