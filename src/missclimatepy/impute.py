# SPDX-License-Identifier: MIT
"""
missclimatepy.impute
====================

Local XYZT-based imputation of daily climate station series.

This module implements a station-wise imputer that:

- uses only spatial coordinates (latitude, longitude, altitude) and
  calendar-based features (year, month, day-of-year, optional cyclic
  sin/cos) as predictors;
- calibrates one local model per *target* station, using either all other
  stations or only a K-nearest-neighbors pool;
- optionally includes a fraction of the target station's observed rows
  in the training pool (``include_target_pct``);
- returns a **complete daily series** for each *imputed* station over a
  requested window, with a clear provenance flag for each value.

Returned schema
---------------

The output is a long-format table with one row per (station, date)
combination over the constructed grid, for stations that **were actually
imputed**:

    [id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]

where:

- ``source`` ∈ {"observed", "imputed"}.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:  # optional UX dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from .features import (
    ensure_datetime_naive,
    add_calendar_features,
    select_station_ids,
    validate_required_columns,
)
from .neighbors import build_neighbor_map
from .models import make_model


def impute_dataset(
    data: pd.DataFrame,
    *,
    # column names (generic schema)
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # temporal window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # feature configuration
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # selection of target stations
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[callable] = None,
    # minimum information per station (to even try imputation)
    min_station_rows: Optional[int] = None,
    # neighborhood & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    # model
    model_kind: str = "rf",
    model_params: Optional[Dict[str, Any]] = None,
    # logging / UX
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Impute daily station series using a local XYZT model per station.

    Only stations that actually obtain a non-empty training pool and
    satisfy ``min_station_rows`` (if provided) are returned.
    """
    # ------------------------------------------------------------------
    # 1. Basic validation & datetime handling
    # ------------------------------------------------------------------
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
        context="impute_dataset",
    )

    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if df.empty:
        return pd.DataFrame(
            columns=[id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
        )

    # Global window for clipping
    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]
    else:
        lo = None
        hi = None

    if df.empty:
        return pd.DataFrame(
            columns=[id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
        )

    # ------------------------------------------------------------------
    # 2. Calendar features & feature list
    # ------------------------------------------------------------------
    df = add_calendar_features(df, date_col=date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feats: List[str] = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # rows with valid features (for training)
    valid_feat_mask = ~df[feats].isna().any(axis=1)
    observed_mask_for_train = valid_feat_mask & df[target_col].notna()

    # Station medoids for coordinates
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median()
    )

    # ------------------------------------------------------------------
    # 3. Choose stations to impute and apply min_station_rows filter
    # ------------------------------------------------------------------
    stations = select_station_ids(
        df,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )

    if min_station_rows is not None:
        obs_counts = (
            df.loc[df[target_col].notna(), [id_col, target_col]]
            .groupby(id_col)[target_col]
            .size()
            .astype(int)
        )
        before = len(stations)
        stations = [
            s for s in stations
            if int(obs_counts.get(s, 0)) >= int(min_station_rows)
        ]
        if show_progress:
            tqdm.write(
                f"Filtered by min_station_rows(observed)={int(min_station_rows)}: "
                f"{before} -> {len(stations)} stations"
            )

    if not stations:
        if show_progress:
            tqdm.write("No stations left to impute after filtering.")
        return pd.DataFrame(
            columns=[id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
        )

    # ------------------------------------------------------------------
    # 4. Neighbor map (if requested)
    # ------------------------------------------------------------------
    if neighbor_map is not None:
        nmap = dict(neighbor_map)
        used_k = None
    elif k_neighbors is not None:
        nmap = build_neighbor_map(
            df,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=int(k_neighbors),
            include_self=False,
        )
        used_k = int(k_neighbors)
    else:
        nmap = None
        used_k = None

    # ------------------------------------------------------------------
    # 5. Iterate over stations and build complete series
    # ------------------------------------------------------------------
    rows_out: List[pd.DataFrame] = []

    iterator = tqdm(stations, desc="Imputing stations", unit="st") if show_progress else stations

    rng_leak = np.random.RandomState(int(include_target_seed))

    for sid in iterator:
        st_mask = (df[id_col] == sid)
        st_df = df.loc[st_mask].copy()

        if st_df.empty:
            if show_progress:
                tqdm.write(f"Station {sid}: no rows in window, skipped.")
            continue

        # Observed rows (for leakage and overwriting)
        st_obs = st_df.loc[st_df[target_col].notna(), [date_col, target_col]]
        n_obs = int(len(st_obs))

        # Station-specific window:
        if lo is not None or hi is not None:
            window_start = lo if lo is not None else st_df[date_col].min()
            window_end = hi if hi is not None else st_df[date_col].max()
        else:
            window_start = st_df[date_col].min()
            window_end = st_df[date_col].max()

        if pd.isna(window_start) or pd.isna(window_end):
            if show_progress:
                tqdm.write(f"Station {sid}: invalid date range, skipped.")
            continue

        grid_dates = pd.date_range(window_start, window_end, freq="D")
        if grid_dates.empty:
            if show_progress:
                tqdm.write(f"Station {sid}: empty date grid, skipped.")
            continue

        # Base training pool: neighbors or all-other stations (excluding target)
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_base_mask = df[id_col].isin(neigh_ids) & observed_mask_for_train
        else:
            train_base_mask = (~st_mask) & observed_mask_for_train

        train_pool = df.loc[train_base_mask, feats + [target_col]]

        # Optional leakage from target station
        train_df = train_pool

        pct = max(0.0, min(float(include_target_pct), 95.0))
        if pct > 0.0 and n_obs > 0:
            st_obs_full = df.loc[
                st_mask & observed_mask_for_train,
                feats + [target_col],
            ]
            if not st_obs_full.empty:
                n_total = int(len(st_obs_full))
                n_take = int(np.ceil(n_total * (pct / 100.0)))
                n_take = max(1, min(n_take, n_total))
                chosen_idx = rng_leak.choice(
                    st_obs_full.index.to_numpy(),
                    size=n_take,
                    replace=False,
                )
                leak_df = st_obs_full.loc[chosen_idx, feats + [target_col]]
                train_df = pd.concat([train_pool, leak_df], axis=0, copy=False)

        # Si no hay pool de entrenamiento, OMITIMOS la estación
        if train_df.empty:
            if show_progress:
                tqdm.write(
                    f"Station {sid}: empty training pool "
                    f"(neighbors={used_k if used_k is not None else 'all/none'}) "
                    f"-> skipped (not imputed)."
                )
            continue

        # ------------------------------------------------------------------
        # Imputation
        # ------------------------------------------------------------------
        # Build grid and broadcast coordinates
        grid = pd.DataFrame(
            {
                id_col: sid,
                date_col: grid_dates,
            }
        )

        if sid in medoids.index:
            grid[lat_col] = float(medoids.loc[sid, lat_col])
            grid[lon_col] = float(medoids.loc[sid, lon_col])
            grid[alt_col] = float(medoids.loc[sid, alt_col])
        else:
            grid[lat_col] = np.nan
            grid[lon_col] = np.nan
            grid[alt_col] = np.nan

        # Calendar features for the grid
        grid = add_calendar_features(grid, date_col=date_col, add_cyclic=add_cyclic)

        n_grid = len(grid)
        values = np.full(n_grid, np.nan, dtype=float)
        source = np.full(n_grid, "imputed", dtype=object)  # será corregido a 'observed'

        # Observed-by-date mapping (siempre aplicado)
        if not st_obs.empty:
            obs_series = (
                st_obs.dropna(subset=[target_col])
                .drop_duplicates(subset=[date_col], keep="last")
                .set_index(date_col)[target_col]
            )
            observed_dates = obs_series.index
            is_observed = grid[date_col].isin(observed_dates)
            if is_observed.any():
                mapped = grid.loc[is_observed, date_col].map(obs_series)
                values[is_observed.to_numpy()] = mapped.to_numpy()
                source[is_observed.to_numpy()] = "observed"

        # Entrenamos y predecimos sobre toda la malla
        model = make_model(model_kind=model_kind, model_params=model_params)
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_grid = grid[feats].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_grid).astype(float)

        # Sólo sobrescribimos donde no había observado
        is_not_observed = (source != "observed")
        values[is_not_observed] = y_hat[is_not_observed]

        # Adjuntamos al grid
        grid[target_col] = values
        grid["source"] = source

        cols_out = [id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
        rows_out.append(grid[cols_out].copy())

        if show_progress:
            n_imp = int((grid["source"] == "imputed").sum())
            tqdm.write(
                f"[impute] Station {sid}: "
                f"grid={n_grid:,}  obs={n_obs:,}  imputed={n_imp:,}"
            )

    if not rows_out:
        if show_progress:
            tqdm.write("No stations were actually imputed.")
        return pd.DataFrame(
            columns=[id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
        )

    out = pd.concat(rows_out, axis=0, ignore_index=True)
    out = out.sort_values(by=[id_col, date_col]).reset_index(drop=True)
    return out


__all__ = ["impute_dataset"]


