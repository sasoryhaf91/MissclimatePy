# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise evaluation of XYZT models for daily climate data.

This module implements a *local* evaluation scheme where, for each target
station, we:

1. Build XYZT features from:
   - X: latitude, longitude, altitude
   - T: calendar variables (year, month, day-of-year, optional cyclic sin/cos)
2. Define a training pool based on:
   - all other stations, or
   - a K-nearest-neighbor set in (lat, lon).
3. Optionally include a controlled fraction of the target station's valid
   rows into the training pool (``include_target_pct``) using a
   **precipitation-friendly** stratified scheme (month × dry/wet).
4. Fit a regression model chosen via :func:`missclimatepy.models.make_model`
   (Random Forest by default) on the training pool.
5. Predict on the held-out rows of the target station (test set).
6. Compare the model against a **Mean Climatology Model (MCM)** baseline
   based on day-of-year (``baseline="mcm_doy"``).

Metrics are computed at three temporal scales:

- Daily: direct comparison on the test rows.
- Monthly: after aggregating to calendar months.
- Annual: after aggregating to calendar years.

At each scale we report:

- MAE, RMSE, R², and KGE (Kling–Gupta efficiency).

For the baseline, the same metrics are returned with the ``_mcm`` suffix.

The main public entry point is :func:`evaluate_stations`, which is wrapped
by the high-level API function :func:`missclimatepy.api.evaluate_xyzt`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:  # optional pretty progress
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # type: ignore[return-value]


from .features import (
    ensure_datetime_naive,
    add_time_features,
    validate_required_columns,
)
from .metrics import compute_metrics, aggregate_and_compute
from .models import make_model
from .neighbors import build_neighbor_map


StationID = Union[str, int]


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _select_stations(
    df: pd.DataFrame,
    *,
    id_col: str,
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[StationID]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[StationID], bool]] = None,
) -> List[StationID]:
    """
    Apply OR-combined station filters and return a unique list of IDs.
    """
    all_ids: List[StationID] = df[id_col].dropna().unique().tolist()
    chosen: List[StationID] = []

    # Prefix filter
    if prefix is not None:
        if isinstance(prefix, str):
            prefix = [prefix]
        for p in prefix:
            p_str = str(p)
            chosen.extend([sid for sid in all_ids if str(sid).startswith(p_str)])

    # Explicit station IDs
    if station_ids is not None:
        chosen.extend(list(station_ids))

    # Regex filter
    if regex is not None:
        import re

        pat = re.compile(regex)
        chosen.extend([sid for sid in all_ids if pat.match(str(sid))])

    # Custom boolean filter
    if custom_filter is not None:
        chosen.extend([sid for sid in all_ids if custom_filter(sid)])

    if not chosen:
        chosen = all_ids

    # Deduplicate preserving order
    seen = set()
    out: List[StationID] = []
    for sid in chosen:
        if sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _clip_include_pct(pct: float) -> float:
    """
    Clip include_target_pct to [0, 95].
    """
    return float(max(0.0, min(float(pct), 95.0)))


def _stratified_target_split(
    target_valid: pd.DataFrame,
    *,
    target_col: str,
    pct: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified split of target station rows into:

    - inc_target_df: rows to include in training (leakage).
    - test_df: held-out rows for evaluation.

    Stratification is performed by (month, dry/wet) where:

    - month = calendar month extracted beforehand.
    - dry  = target == 0.0
    - wet  = target > 0.0
    """
    n_total = len(target_valid)
    if pct <= 0.0 or n_total <= 1:
        # Pure LOSO-like: no leakage from target
        empty = target_valid.iloc[0:0].copy()
        return empty, target_valid

    n_take = int(np.ceil(n_total * (pct / 100.0)))
    # Ensure at least one row remains for testing
    n_take = min(n_take, n_total - 1)

    tmp = target_valid.copy()
    tmp["_month_"] = tmp["month"].to_numpy()
    tmp["_wet_"] = (tmp[target_col].to_numpy() > 0.0)

    strata: Dict[Tuple[int, bool], np.ndarray] = {}
    for key, sub in tmp.groupby(["_month_", "_wet_"]):
        strata[key] = sub.index.to_numpy()

    inc_indices: List[int] = []
    rng = np.random.RandomState(int(seed))

    for key, idx_arr in strata.items():
        if len(inc_indices) >= n_take:
            break
        n_stratum = idx_arr.size
        if n_stratum == 0:
            continue
        # proportional allocation; may overshoot, we trim later
        n_stratum_take = int(round(n_take * (n_stratum / n_total)))
        n_stratum_take = max(0, min(n_stratum_take, n_stratum))
        if n_stratum_take > 0:
            chosen = rng.choice(idx_arr, size=n_stratum_take, replace=False)
            inc_indices.extend(chosen.tolist())

    if len(inc_indices) > n_take:
        inc_indices = rng.choice(np.array(inc_indices), size=n_take, replace=False).tolist()
    elif len(inc_indices) < n_take:
        remaining = n_take - len(inc_indices)
        all_idx = tmp.index.to_numpy()
        mask_chosen = np.isin(all_idx, np.array(inc_indices))
        pool_left = all_idx[~mask_chosen]
        if pool_left.size > 0:
            extra = rng.choice(pool_left, size=min(remaining, pool_left.size), replace=False)
            inc_indices.extend(extra.tolist())

    inc_index = pd.Index(sorted(set(inc_indices)))
    inc_df = tmp.loc[inc_index].drop(columns=["_month_", "_wet_"])
    test_df = tmp.drop(index=inc_index).drop(columns=["_month_", "_wet_"])

    if test_df.empty:
        # Move one row back from inclusion to test
        move_idx = inc_index[-1]
        move_row = tmp.loc[[move_idx]].drop(columns=["_month_", "_wet_"])
        inc_df = inc_df.drop(index=move_idx)
        test_df = pd.concat([test_df, move_row], axis=0)

    return inc_df, test_df


def _baseline_mcm_doy(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target_col: str,
) -> np.ndarray:
    """
    Mean Climatology Model (MCM) baseline based on day-of-year:

    - For each doy, compute mean(target) on the *training* rows.
    - For each test row, predict the corresponding doy mean.
    - If a doy has no training occurrences, fall back to the global mean.
    """
    if train_df.empty:
        return np.full(len(test_df), np.nan, dtype=float)

    means_by_doy = (
        train_df.groupby("doy")[target_col]
        .mean()
        .astype(float)
    )
    global_mean = float(train_df[target_col].mean())

    doy_series = test_df["doy"]
    baseline = doy_series.map(means_by_doy)
    baseline = baseline.fillna(global_mean)
    return baseline.to_numpy(dtype=float)


def _compute_all_metrics_for_pair(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_col: str = "y_obs",
    yhat_col: str = "y_mod",
    agg_for_metrics: str = "sum",
) -> Dict[str, float]:
    """
    Compute daily, monthly and annual metrics (MAE, RMSE, R2, KGE)
    for a given prediction dataframe.
    """
    y_true = df_pred[y_col].to_numpy()
    y_hat = df_pred[yhat_col].to_numpy()

    daily = compute_metrics(y_true, y_hat, include_kge=True)

    monthly, _ = aggregate_and_compute(
        df_pred,
        date_col=date_col,
        y_col=y_col,
        yhat_col=yhat_col,
        freq="M",
        agg=agg_for_metrics,
        include_kge=True,
    )
    annual, _ = aggregate_and_compute(
        df_pred,
        date_col=date_col,
        y_col=y_col,
        yhat_col=yhat_col,
        freq="YS",
        agg=agg_for_metrics,
        include_kge=True,
    )

    out: Dict[str, float] = {
        "MAE_d": float(daily["MAE"]),
        "RMSE_d": float(daily["RMSE"]),
        "R2_d": float(daily["R2"]) if np.isfinite(daily["R2"]) else float("nan"),
        "KGE_d": float(daily["KGE"]) if np.isfinite(daily["KGE"]) else float("nan"),
        "MAE_m": float(monthly["MAE"]),
        "RMSE_m": float(monthly["RMSE"]),
        "R2_m": float(monthly["R2"]) if np.isfinite(monthly["R2"]) else float("nan"),
        "KGE_m": float(monthly["KGE"]) if np.isfinite(monthly["KGE"]) else float("nan"),
        "MAE_y": float(annual["MAE"]),
        "RMSE_y": float(annual["RMSE"]),
        "R2_y": float(annual["R2"]) if np.isfinite(annual["R2"]) else float("nan"),
        "KGE_y": float(annual["KGE"]) if np.isfinite(annual["KGE"]) else float("nan"),
    }
    return out


# --------------------------------------------------------------------------- #
# Public evaluator
# --------------------------------------------------------------------------- #


def evaluate_stations(
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
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[StationID]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[StationID], bool]] = None,
    min_station_rows: Optional[int] = None,
    # neighbors & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[StationID, List[StationID]]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    # model
    model_kind: str = "rf",
    model_params: Optional[Mapping[str, Any]] = None,
    # metrics
    agg_for_metrics: str = "sum",
    baseline: str = "mcm_doy",
    # UX / logging
    show_progress: bool = False,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    # optional saving
    save_report_path: Optional[str] = None,
    save_preds_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a XYZT model per station using only coordinates + calendar
    features, and compare against an MCM baseline.

    Parameters
    ----------
    data : DataFrame
        Long-format table with at least
        (id_col, date_col, lat_col, lon_col, alt_col, target_col).
    start, end : str or None
        Optional inclusive window for analysis. If None, use full span.
    add_cyclic : bool
        Whether to add sin/cos of day-of-year.
    feature_cols : sequence of str or None
        Custom feature columns. If None, defaults to coordinates and
        calendar features: [lat, lon, alt, year, month, doy, (doy_sin, doy_cos)].
    prefix, station_ids, regex, custom_filter :
        Optional filters to select which stations to *evaluate* (OR logic).
    min_station_rows : int or None
        Minimum number of observed target rows required per station within the
        [start, end] window. Stations below this threshold are skipped.
    k_neighbors : int or None
        If provided and `neighbor_map` is None, build a KNN neighbor map in
        Haversine distance (lat/lon) using :func:`missclimatepy.neighbors.build_neighbor_map`.
        If None and `neighbor_map` is also None, all other stations are used.
    neighbor_map : dict or None
        Precomputed mapping {station_id -> [neighbor_ids,...]}.
        Overrides `k_neighbors` if provided.
    include_target_pct : float
        Percentage (0..95) of valid target rows *from the target station* to
        include in the training set via a month × dry/wet stratified sampler.
        0.0 emulates a LOSO-like setting (no leakage from target).
    include_target_seed : int
        Seed for the stratified sampler when `include_target_pct > 0`.
    model_kind : str
        Identifier of the regression model; passed to :func:`make_model`.
    model_params : mapping or None
        Optional hyperparameters for :func:`make_model`.
    agg_for_metrics : {"sum","mean","median"}
        Aggregation used for monthly/annual metrics.
    baseline : {"mcm_doy"}
        Baseline model to compare against. Currently only "mcm_doy" is
        implemented (mean climatology by day-of-year).
    show_progress : bool
        If True, show a per-station progress bar / log.
    log_csv : str or None
        If provided, append per-station rows to this CSV file every
        `flush_every` stations.
    save_report_path, save_preds_path : str or None
        Optional output paths. If ends with ".csv", writes CSV;
        if ends with ".parquet", writes Parquet; otherwise defaults to CSV.
    parquet_compression : str
        Compression codec for Parquet output.

    Returns
    -------
    report : DataFrame
        One row per station with metrics and metadata:
        [station, n_rows, seconds, rows_train, rows_test,
         MAE_d, RMSE_d, R2_d, KGE_d,
         MAE_m, RMSE_m, R2_m, KGE_m,
         MAE_y, RMSE_y, R2_y, KGE_y,
         MAE_d_mcm, RMSE_d_mcm, R2_d_mcm, KGE_d_mcm,
         MAE_m_mcm, RMSE_m_mcm, R2_m_mcm, KGE_m_mcm,
         MAE_y_mcm, RMSE_y_mcm, R2_y_mcm, KGE_y_mcm,
         used_k_neighbors, include_target_pct,
         latitude, longitude, altitude]
    preds : DataFrame
        Per-row test predictions across all stations:
        [station, date, latitude, longitude, altitude, y_obs, y_mod]
    """
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
        context="evaluate_stations",
    )

    # --- Preprocess: datetime + clip window ---------------------------------
    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # Add calendar features
    df = add_time_features(df, date_col=date_col, add_cyclic=add_cyclic)

    # Determine feature columns
    if feature_cols is None:
        feats: List[str] = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    validate_required_columns(df, feats, context="evaluate_stations(features)")

    # Keep only necessary columns
    keep_cols = sorted(
        set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats)
    )
    df = df[keep_cols]

    # Global validity mask: rows with non-NaN in all features + target
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # --- Station selection ---------------------------------------------------
    stations = _select_stations(
        df,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )

    # Apply min_station_rows on observed (not global valid) rows
    if min_station_rows is not None:
        obs_counts = (
            df.loc[~df[target_col].isna(), [id_col, target_col]]
            .groupby(id_col)[target_col]
            .size()
            .astype(int)
        )
        before = len(stations)
        stations = [
            sid for sid in stations if int(obs_counts.get(sid, 0)) >= int(min_station_rows)
        ]
        if show_progress and before != len(stations):
            tqdm.write(
                f"Filtered by min_station_rows(observed)={int(min_station_rows)}: "
                f"{before} -> {len(stations)} stations"
            )

    # --- Neighbor map --------------------------------------------------------
    used_k: Optional[int]
    if neighbor_map is not None:
        nmap = neighbor_map
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

    # --- Station medoids for report ------------------------------------------
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median()
        .rename(
            columns={
                lat_col: "latitude",
                lon_col: "longitude",
                alt_col: "altitude",
            }
        )
    )

    # --- Iteration over stations ---------------------------------------------
    rows_report: List[Dict[str, Any]] = []
    preds_list: List[pd.DataFrame] = []
    pending_log: List[Dict[str, Any]] = []
    header_written: Dict[str, bool] = {}

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    pct = _clip_include_pct(include_target_pct)

    def _append_log_rows():
        if log_csv and pending_log:
            tmp = pd.DataFrame(pending_log)
            first = not header_written.get(log_csv, False)
            tmp.to_csv(log_csv, mode="a", index=False, header=first)
            header_written[log_csv] = True
            pending_log.clear()

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()
        is_target = df[id_col] == sid

        # Valid rows for this station (candidates for test/leakage)
        target_valid = df.loc[is_target & valid_mask_global].copy()

        if target_valid.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            coords = medoids.loc[sid] if sid in medoids.index else None
            row = {
                "station": sid,
                "n_rows": 0,
                "seconds": float(sec),
                "rows_train": 0,
                "rows_test": 0,
                # model metrics
                "MAE_d": np.nan,
                "RMSE_d": np.nan,
                "R2_d": np.nan,
                "KGE_d": np.nan,
                "MAE_m": np.nan,
                "RMSE_m": np.nan,
                "R2_m": np.nan,
                "KGE_m": np.nan,
                "MAE_y": np.nan,
                "RMSE_y": np.nan,
                "R2_y": np.nan,
                "KGE_y": np.nan,
                # baseline metrics
                "MAE_d_mcm": np.nan,
                "RMSE_d_mcm": np.nan,
                "R2_d_mcm": np.nan,
                "KGE_d_mcm": np.nan,
                "MAE_m_mcm": np.nan,
                "RMSE_m_mcm": np.nan,
                "R2_m_mcm": np.nan,
                "KGE_m_mcm": np.nan,
                "MAE_y_mcm": np.nan,
                "RMSE_y_mcm": np.nan,
                "R2_y_mcm": np.nan,
                "KGE_y_mcm": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": pct,
                "latitude": float(coords["latitude"]) if coords is not None else np.nan,
                "longitude": float(coords["longitude"]) if coords is not None else np.nan,
                "altitude": float(coords["altitude"]) if coords is not None else np.nan,
            }
            rows_report.append(row)
            pending_log.append(row)
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid rows (skipped)")
            if log_csv and len(pending_log) >= flush_every:
                _append_log_rows()
            continue

        # Training pool: neighbors or all-other stations
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target)
        else:
            train_pool_mask = ~is_target
        train_pool = df.loc[train_pool_mask & valid_mask_global].copy()

        # Stratified inclusion of target rows
        inc_target_df, test_df = _stratified_target_split(
            target_valid,
            target_col=target_col,
            pct=pct,
            seed=include_target_seed,
        )
        train_df = pd.concat([train_pool, inc_target_df], axis=0)

        if train_df.empty or test_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            coords = medoids.loc[sid] if sid in medoids.index else None
            row = {
                "station": sid,
                "n_rows": int(len(test_df)),
                "seconds": float(sec),
                "rows_train": int(len(train_df)),
                "rows_test": int(len(test_df)),
                "MAE_d": np.nan,
                "RMSE_d": np.nan,
                "R2_d": np.nan,
                "KGE_d": np.nan,
                "MAE_m": np.nan,
                "RMSE_m": np.nan,
                "R2_m": np.nan,
                "KGE_m": np.nan,
                "MAE_y": np.nan,
                "RMSE_y": np.nan,
                "R2_y": np.nan,
                "KGE_y": np.nan,
                "MAE_d_mcm": np.nan,
                "RMSE_d_mcm": np.nan,
                "R2_d_mcm": np.nan,
                "KGE_d_mcm": np.nan,
                "MAE_m_mcm": np.nan,
                "RMSE_m_mcm": np.nan,
                "R2_m_mcm": np.nan,
                "KGE_m_mcm": np.nan,
                "MAE_y_mcm": np.nan,
                "RMSE_y_mcm": np.nan,
                "R2_y_mcm": np.nan,
                "KGE_y_mcm": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": pct,
                "latitude": float(coords["latitude"]) if coords is not None else np.nan,
                "longitude": float(coords["longitude"]) if coords is not None else np.nan,
                "altitude": float(coords["altitude"]) if coords is not None else np.nan,
            }
            rows_report.append(row)
            pending_log.append(row)
            if show_progress:
                tqdm.write(f"Station {sid}: empty train or test (skipped)")
            if log_csv and len(pending_log) >= flush_every:
                _append_log_rows()
            continue

        # --- Fit model --------------------------------------------------------
        model = make_model(model_kind=model_kind, model_params=model_params)

        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[feats].to_numpy(copy=False)
        y_test = test_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # Per-row prediction table
        coords_cols = {
            "latitude": test_df[lat_col].astype(float).values,
            "longitude": test_df[lon_col].astype(float).values,
            "altitude": test_df[alt_col].astype(float).values,
        }
        pred_df = pd.DataFrame(
            {
                "station": sid,
                date_col: test_df[date_col].values,
                "y_obs": y_test,
                "y_mod": y_hat,
                **coords_cols,
            }
        )

        # Baseline MCM (currently only doy-based)
        if baseline == "mcm_doy":
            y_hat_mcm = _baseline_mcm_doy(train_df, test_df, target_col=target_col)
            base_df = pred_df.copy()
            base_df["y_mod"] = y_hat_mcm
        else:
            raise ValueError(
                f"Unsupported baseline '{baseline}'. Currently only 'mcm_doy' is implemented."
            )

        # Metrics for model & baseline
        model_metrics = _compute_all_metrics_for_pair(
            pred_df,
            date_col=date_col,
            y_col="y_obs",
            yhat_col="y_mod",
            agg_for_metrics=agg_for_metrics,
        )
        base_metrics = _compute_all_metrics_for_pair(
            base_df,
            date_col=date_col,
            y_col="y_obs",
            yhat_col="y_mod",
            agg_for_metrics=agg_for_metrics,
        )

        sec = pd.Timestamp.utcnow().timestamp() - t0
        coords = medoids.loc[sid] if sid in medoids.index else None

        row: Dict[str, Any] = {
            "station": sid,
            "n_rows": int(len(pred_df)),
            "seconds": float(sec),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "used_k_neighbors": used_k,
            "include_target_pct": pct,
            "latitude": float(coords["latitude"]) if coords is not None else np.nan,
            "longitude": float(coords["longitude"]) if coords is not None else np.nan,
            "altitude": float(coords["altitude"]) if coords is not None else np.nan,
        }

        # Merge model metrics
        row.update(model_metrics)
        # Merge baseline metrics with _mcm suffix
        for k, v in base_metrics.items():
            row[f"{k}_mcm"] = v

        rows_report.append(row)
        preds_list.append(pred_df)
        pending_log.append(row)

        if show_progress:
            tqdm.write(
                f"{sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={model_metrics['MAE_d']:.3f}  "
                f"RMSE_d={model_metrics['RMSE_d']:.3f}  "
                f"R2_d={model_metrics['R2_d']:.3f}"
            )

        if log_csv and len(pending_log) >= flush_every:
            _append_log_rows()

    # Flush remaining log rows
    if log_csv and pending_log:
        _append_log_rows()

    # Assemble outputs
    report = pd.DataFrame(rows_report)
    if not report.empty and "RMSE_d" in report.columns:
        report = report.sort_values("RMSE_d", ascending=True).reset_index(drop=True)

    if preds_list:
        preds = pd.concat(preds_list, axis=0, ignore_index=True)
    else:
        preds = pd.DataFrame(
            columns=["station", date_col, "latitude", "longitude", "altitude", "y_obs", "y_mod"]
        )

    # Optional saving
    def _save(df_out: pd.DataFrame, path: Optional[str]) -> None:
        if not path:
            return
        ext = str(path).lower()
        if ext.endswith(".csv"):
            df_out.to_csv(path, index=False)
        elif ext.endswith(".parquet"):
            df_out.to_parquet(path, index=False, compression=parquet_compression)
        else:
            df_out.to_csv(path, index=False)

    _save(report, save_report_path)
    _save(preds, save_preds_path)

    return report, preds


__all__ = ["evaluate_stations"]

