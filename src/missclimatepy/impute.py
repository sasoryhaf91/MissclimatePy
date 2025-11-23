# SPDX-License-Identifier: MIT
"""
missclimatepy.impute
====================

Low-level engine for full-series imputation in MissClimatePy.

This module provides :func:`impute_dataset`, which performs single-target
imputation over a long-format climate dataset using purely spatial–temporal
predictors::

    [latitude, longitude, altitude, year, month, day-of-year, (doy_sin, doy_cos)]

Key ideas
---------

- Space–time only:
  All models use (x, y, z, t) features only. No internal covariates from the
  climate network are used as predictors, by design.

- Station-wise logic:
  Imputation is performed "per station", but models are trained globally
  (using all other stations or only K nearest neighbors). For each station,
  only the target variable values that are missing are imputed; observed
  values are preserved and labeled as such.

- Flexible model backend:
  The main backend is a RandomForestRegressor (via RFParams), but several
  alternatives are available through ``model_kind``::

      "rf"     : RandomForestRegressor
      "knn"    : KNeighborsRegressor
      "linear" : LinearRegression
      "mlp"    : MLPRegressor (ANN)
      "svd"    : TruncatedSVD + LinearRegression
      "mcm"    : Mean Climatology Model (pure temporal baseline)

  All of them share the same feature space (x, y, z, t).

- Mean Climatology Model (MCM):
  When ``model_kind="mcm"``, imputation is performed using a deterministic
  climatological baseline built from the station's own temporal record
  (mean per day-of-year, per month or global, depending on ``mcm_mode``).
  In addition, MCM is used as a fallback when a machine-learning model
  cannot be trained (e.g. empty training pool).

Returned format
---------------

The imputation result is a minimal long-format table with columns::

    [id_col, date_col, lat_col, lon_col, alt_col, target_col, "source"]

where ``source`` is either ``"observed"`` (original value) or ``"imputed"``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .features import (
    ensure_datetime_naive,
    add_calendar_features,
    validate_required_columns,
    default_feature_names,
)
from .neighbors import build_neighbor_map
from .metrics import compute_mcm_baseline
from .evaluate import RFParams, _make_model, _select_stations

try:  # optional progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # transparent iterator


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_rf_params(
    rf_params: Optional[Union[RFParams, Mapping[str, Any]]]
) -> RFParams:
    """
    Normalize RF hyperparameters into an RFParams instance.

    Parameters
    ----------
    rf_params :
        - None        → default RFParams()
        - RFParams    → returned as-is
        - dict/mapping→ merged into default RFParams

    Returns
    -------
    RFParams
    """
    if rf_params is None:
        return RFParams()
    if isinstance(rf_params, RFParams):
        return rf_params
    base = asdict(RFParams())
    base.update(dict(rf_params))
    return RFParams(**base)


# ---------------------------------------------------------------------------
# Public imputation routine
# ---------------------------------------------------------------------------


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
    # temporal window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection
    station_ids: Optional[Iterable[Union[int, str]]] = None,
    prefix: Optional[Iterable[str] | str] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[int, str]], bool]] = None,
    # neighbors / spatial context
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[int, str], List[Union[int, str]]]] = None,
    # target leakage / MDR-style experiments
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    min_station_rows: Optional[int] = None,
    # model backend
    rf_params: Optional[Union[RFParams, Mapping[str, Any]]] = None,
    model_kind: str = "rf",
    model_params: Optional[Mapping[str, Any]] = None,
    # MCM configuration (used when model_kind="mcm" or when falling back)
    mcm_mode: str = "doy",
    mcm_min_samples: int = 1,
    # logging
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Impute a single target variable over a long-format climate dataset.

    The function operates station by station. For each selected station, only
    rows where ``target_col`` is missing are imputed; observed values are
    preserved. Models are trained globally using other stations (LOSO-like)
    or only the K nearest neighbors, depending on ``k_neighbors`` and
    ``neighbor_map``.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format table with at least the columns
        [id_col, date_col, lat_col, lon_col, alt_col, target_col].
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, coordinates and target variable.
    start, end : str or None, optional
        Inclusive temporal window for imputation. If None, inferred from data.
    add_cyclic : bool, default False
        If True, adds sin/cos of day-of-year (``doy_sin``, ``doy_cos``) as features.
    feature_cols : sequence of str or None, optional
        Custom feature set. If None, defaults to
        [``lat_col``, ``lon_col``, ``alt_col``, ``"year"``, ``"month"``, ``"doy"``]
        plus cyclic harmonics when ``add_cyclic=True``.
    station_ids, prefix, regex, custom_filter :
        Filters to select which stations to impute (OR semantics). If no
        filter is provided, all stations in ``data[id_col]`` are considered.
    k_neighbors : int or None, default 20
        If provided and ``neighbor_map`` is None, a neighbor map is built using
        haversine BallTree over station centroids, and only those neighbors
        (excluding the target) are used in the training pool. If None, all
        other stations constitute the training pool.
    neighbor_map : dict or None, default None
        Explicit neighbor mapping ``{station_id -> list_of_neighbor_ids}``.
        Overrides ``k_neighbors`` when provided.
    include_target_pct : float, default 0.0
        Percentage (0–100) of *valid* target-station rows (within the window)
        to include in the training set, as a controlled form of target leakage.
        This is mainly intended for MDR-style experiments. Values above 95
        are clipped.
    include_target_seed : int, default 42
        Random seed used when sampling target-station rows for leakage.
    min_station_rows : int or None, default None
        Minimum number of rows a station must have in the time window to be
        imputed. Stations below this threshold are skipped (their original
        rows are not returned by this function).
    rf_params : RFParams | dict | None, default None
        Hyperparameters for the RandomForestRegressor when ``model_kind="rf"``.
    model_kind : {"rf", "knn", "linear", "mlp", "svd", "mcm"}, default "rf"
        Main backend used for imputation. All backends operate on the same
        feature space. If "mcm", a Mean Climatology Model is used and no
        neighbor information is required.
    model_params : mapping or None, default None
        Extra parameters specific to the chosen backend.
    mcm_mode : {"doy", "month", "global"}, default "doy"
        Temporal grouping used by the MCM imputer and fallback.
    mcm_min_samples : int, default 1
        Minimum number of observations required per group to compute a local
        MCM mean. Groups with fewer samples fall back to the global mean.
    show_progress : bool, default False
        If True, display a progress bar / per-station messages.

    Returns
    -------
    pandas.DataFrame
        Minimal long-format table with columns::

            [id_col, date_col, lat_col, lon_col, alt_col, target_col, "source"]

        where ``source`` is either ``"observed"`` or ``"imputed"``.
    """
    # ------------------------------------------------------------------
    # 1) Basic schema validation and datetime / window handling
    # ------------------------------------------------------------------
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
        context="impute_dataset",
    )

    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
    else:
        lo = df[date_col].min()
        hi = df[date_col].max()

    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # Early exit if window is empty
    if df.empty:
        return pd.DataFrame(
            columns=[id_col, date_col, lat_col, lon_col, alt_col, target_col, "source"]
        )

    # ------------------------------------------------------------------
    # 2) Feature engineering
    # ------------------------------------------------------------------
    df = add_calendar_features(df, date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feats = default_feature_names(
            lat_col=lat_col,
            lon_col=lon_col,
            alt_col=alt_col,
            add_cyclic=add_cyclic,
        )
    else:
        # Preserve order, remove duplicates
        feats = list(dict.fromkeys(feature_cols))

    # Masks
    feature_ok_mask = ~df[feats].isna().any(axis=1)
    valid_train_mask = feature_ok_mask & df[target_col].notna()

    # ------------------------------------------------------------------
    # 3) Station selection + MDR filter
    # ------------------------------------------------------------------
    all_stations = _select_stations(
        df,
        id_col=id_col,
        station_ids=station_ids,
        prefix=prefix,
        regex=regex,
        custom_filter=custom_filter,
    )

    if min_station_rows is not None:
        counts = df[[id_col]].groupby(id_col).size().astype(int)
        before = len(all_stations)
        all_stations = [
            sid for sid in all_stations if int(counts.get(sid, 0)) >= int(min_station_rows)
        ]
        if show_progress:
            tqdm.write(
                f"Filtered by min_station_rows={min_station_rows}: "
                f"{before} → {len(all_stations)} stations"
            )

    if not all_stations:
        return pd.DataFrame(
            columns=[id_col, date_col, lat_col, lon_col, alt_col, target_col, "source"]
        )

    # ------------------------------------------------------------------
    # 4) Neighbor map (haversine BallTree) – not needed for pure MCM
    # ------------------------------------------------------------------
    model_kind_lower = model_kind.lower()

    if model_kind_lower == "mcm":
        nmap = None
    else:
        if neighbor_map is not None:
            nmap = neighbor_map
        elif k_neighbors is not None:
            nmap = build_neighbor_map(
                df,
                id_col=id_col,
                lat_col=lat_col,
                lon_col=lon_col,
                k=int(k_neighbors),
                include_self=False,
            )
        else:
            nmap = None

    # ------------------------------------------------------------------
    # 5) Model setup
    # ------------------------------------------------------------------
    rf_norm = _normalize_rf_params(rf_params)
    n_features = len(feats)
    pct = max(0.0, min(float(include_target_pct), 95.0))

    # ------------------------------------------------------------------
    # 6) Main loop over stations
    # ------------------------------------------------------------------
    results: List[pd.DataFrame] = []

    iterator = tqdm(all_stations, desc="Imputing stations", unit="st") if show_progress else all_stations

    for sid in iterator:
        is_target_station = df[id_col] == sid
        station_df = df.loc[is_target_station].copy()

        if station_df.empty:
            # Should not happen with previous filters, but keep safe.
            continue

        # Identify missing rows for this station (with valid features)
        missing_mask_station = station_df[target_col].isna() & feature_ok_mask.loc[is_target_station].values
        n_missing = int(missing_mask_station.sum())

        # If nothing to impute and not in MCM-only mode: just mark as observed
        if n_missing == 0 and model_kind_lower != "mcm":
            station_out = station_df[[id_col, date_col, lat_col, lon_col, alt_col, target_col]].copy()
            station_out["source"] = "observed"
            results.append(station_out)
            if show_progress:
                tqdm.write(f"{sid}: no missing values → nothing imputed")
            continue

        # ------------------------------------------------------------------
        # MCM-only backend: purely temporal baseline per station
        # ------------------------------------------------------------------
        if model_kind_lower == "mcm":
            dates = station_df[date_col].values
            values = station_df[target_col].values

            # If absolutely no observed values, we cannot build climatology
            if np.all(np.isnan(values)):
                if show_progress:
                    tqdm.write(f"{sid}: no observed values → cannot impute with MCM")
                station_out = station_df[[id_col, date_col, lat_col, lon_col, alt_col, target_col]].copy()
                station_out["source"] = np.where(
                    station_out[target_col].isna(), "missing", "observed"
                )
                results.append(station_out)
                continue

            mcm_pred = compute_mcm_baseline(
                dates=dates,
                values=values,
                mode=mcm_mode,
                min_samples=int(mcm_min_samples),
            )

            station_out = station_df[[id_col, date_col, lat_col, lon_col, alt_col, target_col]].copy()
            is_missing = station_out[target_col].isna()
            station_out.loc[is_missing, target_col] = mcm_pred[is_missing].values
            station_out["source"] = np.where(is_missing, "imputed", "observed")
            results.append(station_out)

            if show_progress:
                tqdm.write(f"{sid}: MCM imputation done (n_missing={n_missing})")

            continue

        # ------------------------------------------------------------------
        # Machine-learning backends (RF, KNN, Linear, MLP, SVD)
        # ------------------------------------------------------------------

        # Training pool: neighbors or all-other stations
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target_station) & valid_train_mask
        else:
            train_pool_mask = (~is_target_station) & valid_train_mask

        train_pool = df.loc[train_pool_mask, feats + [target_col]]

        # Controlled leakage from the target station
        station_valid_for_leak = df.loc[is_target_station & valid_train_mask, feats + [target_col]]

        if pct > 0.0 and not station_valid_for_leak.empty:
            n_take = int(np.ceil(len(station_valid_for_leak) * (pct / 100.0)))
            leakage = station_valid_for_leak.sample(
                n=n_take,
                random_state=int(include_target_seed),
            )
            train_df = pd.concat([train_pool, leakage], axis=0, ignore_index=True)
        else:
            train_df = train_pool

        # If training set is empty, fall back to MCM
        if train_df.empty:
            if show_progress:
                tqdm.write(
                    f"{sid}: empty training pool for model_kind={model_kind_lower} → "
                    f"falling back to MCM"
                )

            dates = station_df[date_col].values
            values = station_df[target_col].values

            if np.all(np.isnan(values)):
                station_out = station_df[[id_col, date_col, lat_col, lon_col, alt_col, target_col]].copy()
                station_out["source"] = np.where(
                    station_out[target_col].isna(), "missing", "observed"
                )
                results.append(station_out)
                continue

            mcm_pred = compute_mcm_baseline(
                dates=dates,
                values=values,
                mode=mcm_mode,
                min_samples=int(mcm_min_samples),
            )
            station_out = station_df[[id_col, date_col, lat_col, lon_col, alt_col, target_col]].copy()
            is_missing = station_out[target_col].isna()
            station_out.loc[is_missing, target_col] = mcm_pred[is_missing].values
            station_out["source"] = np.where(is_missing, "imputed", "observed")
            results.append(station_out)
            continue

        # Train model
        model = _make_model(
            model_kind=model_kind_lower,
            rf_params=rf_norm,
            model_params=model_params,
            n_features=n_features,
        )
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)

        # Predict only for rows with missing target and valid features
        station_missing_for_model = station_df[target_col].isna() & feature_ok_mask.loc[is_target_station].values
        if int(station_missing_for_model.sum()) == 0:
            station_out = station_df[[id_col, date_col, lat_col, lon_col, alt_col, target_col]].copy()
            station_out["source"] = "observed"
            results.append(station_out)
            if show_progress:
                tqdm.write(f"{sid}: no missing rows with valid features → nothing imputed")
            continue

        X_missing = station_df.loc[station_missing_for_model, feats].to_numpy(copy=False)
        y_imputed = model.predict(X_missing)

        station_out = station_df[[id_col, date_col, lat_col, lon_col, alt_col, target_col]].copy()
        is_missing = station_out[target_col].isna()
        station_out.loc[station_missing_for_model, target_col] = y_imputed
        station_out["source"] = np.where(is_missing, "imputed", "observed")
        results.append(station_out)

        if show_progress:
            tqdm.write(
                f"{sid}: imputed {int(is_missing.sum())} values "
                f"with model_kind={model_kind_lower} k={k_neighbors}"
            )

    # ------------------------------------------------------------------
    # 7) Assemble final table
    # ------------------------------------------------------------------
    if not results:
        return pd.DataFrame(
            columns=[id_col, date_col, lat_col, lon_col, alt_col, target_col, "source"]
        )

    out = pd.concat(results, axis=0, ignore_index=True)

    # Sort for reproducibility
    out.sort_values(
        by=[id_col, date_col],
        inplace=True,
        kind="mergesort",
    )
    out.reset_index(drop=True, inplace=True)

    return out


__all__ = [
    "impute_dataset",
]
