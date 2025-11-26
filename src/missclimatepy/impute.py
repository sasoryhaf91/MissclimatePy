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
- returns a **complete daily series** for each evaluated station over a
  requested window, with a clear provenance flag for each value.

Design
------

For each target station:

1. Build a daily date grid:

   - If ``start`` / ``end`` are provided, use the global window
     ``[start, end]``.
   - Otherwise, use the station's own span
     ``[min(date), max(date)]`` in the input data.

2. Attach station coordinates (median lat/lon/alt over the input rows)
   and derive calendar features.

3. Define a training pool:

   - If ``neighbor_map`` is provided, use the listed neighbors.
   - Else if ``k_neighbors`` is provided, use a Haversine KNN map.
   - Else, default to "all other stations".

   The pool always requires non-null target and non-null feature values.

4. Optionally leak a fraction ``include_target_pct`` of the target
   station's own observed rows into the training pool.

5. Train a model via :func:`missclimatepy.models.make_model` and
   predict on the full date grid.

6. Overwrite predicted values on dates where we have an original
   observation from the input. These are flagged as ``source='observed'``,
   whereas truly imputed values are flagged as ``source='imputed'``.

If either:

- the station has fewer than ``min_station_rows`` observed rows, or
- the training pool is empty,

then no model is trained and the station is *passed through*:

- Observed dates keep their original value and ``source='observed'``;
- Non-observed dates are returned as ``NaN`` with ``source='missing'``.

Returned schema
---------------

The output is a long-format table with one row per
(station, date) combination over the constructed grid:

    [id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]

where:

- ``source`` ∈ {"observed", "imputed", "missing"}.

This function is intended for reconstruction / completion of station
records; model evaluation and MDR-style experiments are handled by
:mod:`missclimatepy.evaluate`.
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
    # minimum information per station (controls whether we train a model)
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

    Parameters
    ----------
    data : DataFrame
        Long-format table with at least
        (id_col, date_col, lat_col, lon_col, alt_col, target_col).
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, coordinates and target.
    start, end : str or None
        Optional inclusive window for reconstruction. If provided, each
        station's output spans the same [start, end] interval. If not
        provided, each station spans [min(date), max(date)] in the input.
    add_cyclic : bool
        If True, add sin/cos transforms of day-of-year to the features.
    feature_cols : sequence of str or None
        Custom feature list. If None, defaults to
        [lat, lon, alt, year, month, doy] (+ cyclic terms if requested).
    prefix, station_ids, regex, custom_filter :
        Optional filters to choose which stations to *impute* (OR semantics).
        If none match, all stations in ``data`` are used.
    min_station_rows : int or None
        Minimum number of observed (non-NaN) target values a station must
        have in the working window in order to train a model. Stations
        below this threshold are passed through without imputation:
        observed dates remain observed; other dates are returned as
        ``NaN`` / ``source='missing'``.
    k_neighbors : int or None
        If provided and ``neighbor_map`` is None, a Haversine KNN map is
        built on lat/lon and each station's training pool uses only its
        neighbors (excluding itself).
    neighbor_map : dict or None
        Precomputed neighborhood map {station_id -> [neighbor_ids]}.
        Overrides ``k_neighbors`` if given.
    include_target_pct : float
        Fraction (0..95) of the target station's own observed rows to
        include in the training pool. A value of 0.0 prevents leakage
        from the target station; a small positive value allows slight
        local adaptation.
    include_target_seed : int
        Seed for the random sampler that chooses which target rows to
        include when ``include_target_pct > 0``.
    model_kind : str
        Identifier passed to :func:`missclimatepy.models.make_model`
        ("rf", "etr", "gbrt", "hgbt", "linreg", "ridge", "lasso",
         "knn", "svr", "mlp", "xgb", ...).
    model_params : dict or None
        Hyperparameters forwarded to :func:`make_model`. Any missing
        entries fall back to model-specific defaults.
    show_progress : bool
        If True, display a tqdm progress bar over stations and brief
        diagnostic messages about imputation decisions.

    Returns
    -------
    DataFrame
        Long-format table with one row per (station, date) in the
        constructed grid. Columns:

        - ``id_col``        : station identifier,
        - ``lat_col``       : latitude (median per station),
        - ``lon_col``       : longitude (median per station),
        - ``alt_col``       : altitude (median per station),
        - ``date_col``      : daily timestamp,
        - ``target_col``    : observed or imputed value (float),
        - ``source``        : {"observed", "imputed", "missing"}.
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
        # Default XYZT feature set
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
    # 3. Choose stations to impute
    # ------------------------------------------------------------------
    stations = select_station_ids(
        df,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
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
            # No data at all for this station in the working window
            if show_progress:
                tqdm.write(f"Station {sid}: no rows in window, skipped.")
            continue

        # Observed rows (for counting / leakage / overwriting)
        st_obs = st_df.loc[st_df[target_col].notna(), [date_col, target_col]]
        n_obs = int(len(st_obs))

        # Station-specific window:
        if lo is not None or hi is not None:
            # Global window [lo, hi] already applied to df
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

        # ------------------------------------------------------------------
        # Training pool for this station
        # ------------------------------------------------------------------
        # Base pool: neighbors or all-other stations (excluding target)
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

        # Decide whether to impute or pass-through
        do_impute = True
        if min_station_rows is not None and n_obs < int(min_station_rows):
            do_impute = False
            if show_progress:
                tqdm.write(
                    f"Station {sid}: n_obs={n_obs} < min_station_rows={int(min_station_rows)} "
                    f"-> no model, pass-through."
                )
        if train_df.empty:
            do_impute = False
            if show_progress:
                tqdm.write(
                    f"Station {sid}: empty training pool "
                    f"(neighbors={used_k if used_k is not None else 'all/none'}) "
                    f"-> no model, pass-through."
                )

        # ------------------------------------------------------------------
        # Imputation / pass-through
        # ------------------------------------------------------------------
        # Initialize container arrays
        n_grid = len(grid)
        values = np.full(n_grid, np.nan, dtype=float)
        source = np.full(n_grid, "missing", dtype=object)

        # Observed-by-date mapping (always applied)
        if not st_obs.empty:
            # unique by date; last occurrence wins if duplicated
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

        # If we decided to impute, fill remaining positions
        if do_impute:
            model = make_model(model_kind=model_kind, model_params=model_params)
            X_train = train_df[feats].to_numpy(copy=False)
            y_train = train_df[target_col].to_numpy(copy=False)
            X_grid = grid[feats].to_numpy(copy=False)

            if X_train.size > 0:
                model.fit(X_train, y_train)
                y_hat = model.predict(X_grid).astype(float)

                # Only overwrite positions that are not already observed
                is_not_observed = (source != "observed")
                values[is_not_observed] = y_hat[is_not_observed]
                source[is_not_observed] = "imputed"

        # Attach to grid and keep only public columns
        grid[target_col] = values
        grid["source"] = source

        cols_out = [id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
        rows_out.append(grid[cols_out].copy())

    if not rows_out:
        return pd.DataFrame(
            columns=[id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
        )

    out = pd.concat(rows_out, axis=0, ignore_index=True)

    # Sort for usability
    out = out.sort_values(by=[id_col, date_col]).reset_index(drop=True)

    return out


__all__ = [
    "impute_dataset",
]

