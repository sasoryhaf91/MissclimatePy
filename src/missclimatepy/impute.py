# SPDX-License-Identifier: MIT
"""
missclimatepy.impute
====================

Spatial–temporal imputation of daily climate station records using XYZT
features:

    X = (lat, lon, alt, calendar features)

This module provides a single high-level function:

- :func:`impute_dataset` – fills missing values in a long-format station × date
  table using a configurable regression model trained on:

    - Spatial coordinates (latitude, longitude, altitude), and
    - Calendar-derived features (year, month, day-of-year, optional cyclic
      sin/cos of day-of-year).

Design goals
------------

- No dependence on internal covariates that are often missing themselves:
  only station metadata + calendar structure.
- Reuse the same XYZT feature design and model factory (:func:`make_model`)
  as :mod:`missclimatepy.evaluate`.
- Flexible configuration:

  * Choice of model via ``model_kind`` and ``model_params``.
  * Optional K-nearest-neighbor restriction via ``k_neighbors`` or
    a precomputed ``neighbor_map``.
  * Station filters (prefix, explicit ids, regex, custom predicate).
  * Minimum number of observed rows per station (``min_station_rows``).
  * Control over the fraction of the target station's valid rows that enter
    training (``include_target_pct``).

Output
------

The function returns a copy of the input dataframe (possibly clipped to a
date window) with:

- The target column ``target_col`` where missing values have been filled
  whenever possible.
- An additional column ``source_col`` (default "source") with values:

  - "observed" – original non-NaN values from the input table.
  - "imputed" – values predicted by the model where the original target
    was NaN and features were valid.
  - "missing" – rows where the target remains NaN because no model was
    trained or features were incomplete.

Example
-------

>>> import pandas as pd
>>> from missclimatepy.impute import impute_dataset
>>>
>>> df = pd.DataFrame({
...     "station": ["S1"] * 4 + ["S2"] * 4,
...     "date": pd.date_range("2020-01-01", periods=4, freq="D").tolist() * 2,
...     "latitude": [10.0] * 4 + [11.0] * 4,
...     "longitude": [-100.0] * 4 + [-101.0] * 4,
...     "altitude": [2000.0] * 8,
...     "tmin": [1.0, 2.0, None, 4.0, 0.5, None, 1.5, 2.0],
... })
>>>
>>> out = impute_dataset(
...     df,
...     id_col="station",
...     date_col="date",
...     lat_col="latitude",
...     lon_col="longitude",
...     alt_col="altitude",
...     target_col="tmin",
...     model_kind="rf",
...     model_params={"n_estimators": 50, "random_state": 42},
... )
>>> out[["station", "date", "tmin", "source"]].head()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # optional progress bar
    from tqdm.auto import tqdm  # type: ignore[import]
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # fallback: transparent iterator

from .features import (
    ensure_datetime_naive,
    add_time_features,
    validate_required_columns,
)
from .models import make_model
from .neighbors import build_neighbor_map


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _prepare_xyzt(
    data: pd.DataFrame,
    *,
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    start: Optional[str],
    end: Optional[str],
    add_cyclic: bool,
    feature_cols: Optional[Sequence[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess once:

    - Ensure naive datetime in ``date_col``.
    - Optionally clip to [start, end].
    - Add calendar features (year, month, doy, optional sin/cos).
    - Decide the list of feature columns to use.

    Returns
    -------
    df : DataFrame
        Preprocessed view containing at least id/date/coords/target/features.
    feats : list[str]
        Names of feature columns used for modelling.
    """
    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    df = add_time_features(df, date_col=date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # mantén sólo las columnas necesarias (id, fecha, coords, target, features)
    keep = sorted(
        set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats)
    )
    df = df[keep]
    return df, feats


def _select_stations(
    df: pd.DataFrame,
    *,
    id_col: str,
    prefix: Optional[Iterable[str]],
    station_ids: Optional[Iterable[Union[str, int]]],
    regex: Optional[str],
    custom_filter: Optional[Callable[[Union[str, int]], bool]],
) -> List[Union[str, int]]:
    """
    Apply OR semantics over filters to select station ids.
    """
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

    # make unique, preserving order
    seen = set()
    out: List[Union[str, int]] = []
    for s in chosen:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _apply_min_station_rows(
    df: pd.DataFrame,
    *,
    id_col: str,
    target_col: str,
    stations: List[Union[str, int]],
    min_station_rows: Optional[int],
) -> List[Union[str, int]]:
    """
    Filter station ids by minimum count of observed (non-NaN) target values.
    """
    if min_station_rows is None:
        return stations

    counts = (
        df.loc[~df[target_col].isna(), [id_col, target_col]]
        .groupby(id_col)[target_col]
        .size()
        .astype(int)
    )
    return [s for s in stations if int(counts.get(s, 0)) >= int(min_station_rows)]


def _build_neighbors_if_needed(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k_neighbors: Optional[int],
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]],
) -> Tuple[Optional[Dict[Union[str, int], List[Union[str, int]]]], Optional[int]]:
    """
    Decide which neighbor map to use.

    Returns
    -------
    (nmap, used_k)
    """
    if neighbor_map is not None:
        return neighbor_map, None
    if k_neighbors is None:
        return None, None
    nmap = build_neighbor_map(
        df,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        k=int(k_neighbors),
        include_self=False,
    )
    return nmap, int(k_neighbors)


def _sample_target_rows_for_training(
    target_valid: pd.DataFrame,
    *,
    target_col: str,
    include_target_pct: float,
    seed: int,
) -> pd.DataFrame:
    """
    From the target station's valid rows (non-NaN target and features),
    sample a percentage to be included in the training set.

    This function **does not** create a test set: unused valid rows remain
    as purely "observed" and are not fed to the regressor.

    Parameters
    ----------
    target_valid : DataFrame
        Rows for a single station with valid features and non-NaN target.
    target_col : str
        Name of the target column.
    include_target_pct : float
        Percentage (0..100). 0 => no rows included in training.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    inc_df : DataFrame
        Subset of target_valid to be added to the training pool.
    """
    pct = float(include_target_pct)
    if pct <= 0.0 or target_valid.empty:
        return target_valid.iloc[0:0]  # empty with same schema

    pct = max(0.0, min(pct, 100.0))
    n_total = len(target_valid)
    n_take = int(np.floor(n_total * (pct / 100.0)))

    if n_take <= 0:
        return target_valid.iloc[0:0]

    rng = np.random.RandomState(int(seed))
    chosen_idx = rng.choice(target_valid.index.to_numpy(), size=n_take, replace=False)
    inc_df = target_valid.loc[chosen_idx]
    return inc_df


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def impute_dataset(
    data: pd.DataFrame,
    *,
    # column names
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # period
    start: Optional[str] = None,
    end: Optional[str] = None,
    # feature config
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    min_station_rows: Optional[int] = None,
    # neighborhood & leakage
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 100.0,
    include_target_seed: int = 42,
    # model
    model_kind: str = "rf",
    model_params: Optional[Dict[str, Any]] = None,
    # output
    source_col: str = "source",
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Impute missing values of a daily climate variable using XYZT regression.

    For each target station:

    1. A training pool is built from:

       - All other stations, or only those specified by a neighbor map
         (either precomputed or built with Haversine KNN), and
       - A subset of the target station's *observed* rows, controlled by
         ``include_target_pct``.

    2. A regression model is instantiated with :func:`make_model` using
       ``model_kind`` and ``model_params`` and fit on the training pool.

    3. The fitted model is applied to all rows of the target station where
       ``target_col`` is NaN but all selected features are non-NaN. These
       rows become "imputed".

    Parameters
    ----------
    data : DataFrame
        Long-format table containing at least

            [id_col, date_col, lat_col, lon_col, alt_col, target_col].

    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, geographic coordinates, and
        the target variable to impute.
    start, end : str or None
        Optional inclusive time window. If provided, imputation is performed
        only on rows in [start, end]; other rows are dropped from the output.
    add_cyclic : bool
        If True, add sine/cosine of day-of-year as extra features.
    feature_cols : sequence of str or None
        Explicit list of feature columns to use. If None, defaults to:

            [lat_col, lon_col, alt_col, "year", "month", "doy"] (+ sin/cos).

    prefix, station_ids, regex, custom_filter :
        Optional filters to select which stations are **imputed** (OR semantics).
        Unselected stations are passed through unchanged, except for the
        addition of ``source_col``.
    min_station_rows : int or None
        Minimum number of observed (non-NaN) target rows required for a
        station to be imputed. Stations with fewer observations are left
        unchanged.

    k_neighbors : int or None
        If provided and ``neighbor_map`` is None, builds a KNN map over
        station centroids (lat/lon) and restricts the training pool for each
        target station to its K nearest neighbors.
    neighbor_map : dict or None
        Custom neighbor map {station_id -> [neighbor_ids,...]}. If provided,
        ``k_neighbors`` is ignored.
    include_target_pct : float
        Percentage (0..100) of the target station's *observed* rows to add
        to the training pool. Use 0.0 for strict LOSO-like imputation
        (training only on other stations). Default 100.0 uses all observed
        rows.
    include_target_seed : int
        Seed used when sampling target rows for training.

    model_kind : str
        Identifier understood by :func:`make_model` (e.g. "rf", "etr", "ridge",
        "mlp", "xgb" if xgboost is installed, etc.).
    model_params : dict or None
        Hyperparameters passed to :func:`make_model`. Missing keys fall back
        to model-specific defaults.

    source_col : str
        Name of the column indicating the origin of each value
        ("observed"/"imputed"/"missing").
    show_progress : bool
        If True, display a per-station progress bar using tqdm (when
        available).

    Returns
    -------
    DataFrame
        Copy of the input (possibly time-windowed) with ``target_col``
        imputed where possible and an extra ``source_col`` column.
    """
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
        context="impute_dataset",
    )

    # Preprocess and build XYZT features
    df_xyzt, feats = _prepare_xyzt(
        data,
        id_col=id_col,
        date_col=date_col,
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        target_col=target_col,
        start=start,
        end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
    )

    if df_xyzt.empty:
        out = df_xyzt.copy()
        out[source_col] = np.where(
            out[target_col].notna(),
            "observed",
            "missing",
        )
        return out

    # Validity mask for features + target (for training use)
    valid_features_mask = ~df_xyzt[feats].isna().any(axis=1)
    valid_global_mask = valid_features_mask & ~df_xyzt[target_col].isna()

    # Decide which stations to process
    stations_all = _select_stations(
        df_xyzt,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )
    stations = _apply_min_station_rows(
        df_xyzt,
        id_col=id_col,
        target_col=target_col,
        stations=stations_all,
        min_station_rows=min_station_rows,
    )

    # Neighbor map (if any)
    nmap, used_k = _build_neighbors_if_needed(
        df_xyzt,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        k_neighbors=k_neighbors,
        neighbor_map=neighbor_map,
    )

    # Instantiate model template (we'll re-create per station to ensure
    # independent random states etc.)
    # We call make_model inside the loop so that every station starts
    # from a fresh estimator.

    # Prepare output copy with source flags
    out = df_xyzt.copy()
    out[source_col] = np.where(
        out[target_col].notna(),
        "observed",
        "missing",
    )

    iterator = tqdm(stations, desc="Imputing stations", unit="st") if show_progress else stations

    for sid in iterator:
        # Rows for this station
        station_mask = out[id_col] == sid
        if not station_mask.any():
            continue

        # Observed rows (valid target) for this station (candidates for training)
        station_valid_global = valid_global_mask & station_mask
        target_valid = out.loc[station_valid_global].copy()

        # Missing rows that we may try to impute (NaN target but features valid)
        station_missing = station_mask & out[target_col].isna() & valid_features_mask

        if not station_missing.any():
            # Nothing to impute for this station
            continue

        # If we lack observed data for this station, we may still try to
        # impute from neighbors only (LOSO), provided there is a non-empty
        # training pool.
        # Build training pool from neighbors / all-other stations
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = out[id_col].isin(neigh_ids) & valid_global_mask
        else:
            # all other stations
            train_pool_mask = (~station_mask) & valid_global_mask

        train_pool = out.loc[train_pool_mask].copy()

        # Optionally add a subset of the target station's own observed rows
        if include_target_pct > 0.0 and not target_valid.empty:
            inc_target_df = _sample_target_rows_for_training(
                target_valid,
                target_col=target_col,
                include_target_pct=include_target_pct,
                seed=include_target_seed,
            )
            if not inc_target_df.empty:
                train_pool = pd.concat([train_pool, inc_target_df], axis=0, copy=False)

        if train_pool.empty:
            # No training data → cannot impute this station
            if show_progress:
                tqdm.write(f"Station {sid}: empty training pool (skipped)")
            continue

        # Fit model and impute
        model = make_model(model_kind=model_kind, model_params=model_params or {})
        X_train = train_pool[feats].to_numpy(copy=False)
        y_train = train_pool[target_col].to_numpy(copy=False)
        model.fit(X_train, y_train)

        missing_idx = out.index[station_missing]
        X_missing = out.loc[missing_idx, feats].to_numpy(copy=False)
        try:
            y_hat = model.predict(X_missing)
        except Exception as exc:  # pragma: no cover - defensive
            if show_progress:
                tqdm.write(f"Station {sid}: prediction failed ({exc!r})")
            continue

        out.loc[missing_idx, target_col] = y_hat
        out.loc[missing_idx, source_col] = "imputed"

    return out


__all__ = ["impute_dataset"]
