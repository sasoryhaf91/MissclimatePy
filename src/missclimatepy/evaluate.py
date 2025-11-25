# src/missclimatepy/evaluate.py
# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise evaluation for climate-data imputation using ONLY spatial
coordinates (latitude, longitude, altitude) and calendar features
(year, month, day-of-year, optional cyclic sin/cos). One local model is
trained per *target* station, using either all other stations or the
K nearest neighbors (by haversine distance on lat/lon), with optional
controlled inclusion ("leakage") of a fraction of the target station's
own valid rows.

Key design choices
------------------
- Generic schema: the caller provides the column names (no enforced renaming).
- Local models: one regression model per station; training pool can be neighbors.
- Controlled inclusion: include 0..95% of the target station's valid rows into
  training to adapt locally; 0% means strict LOSO-like exclusion.
- Precipitation-friendly sampling: when ``include_target_pct > 0``, the rows
  taken from the target station are drawn using a *month × dry/wet* stratified
  scheme to avoid oversampling dry days.
- Metrics: daily, monthly and yearly MAE/RMSE/R²/KGE.
- Baseline: a Mean Climatology Model (MCM) based on day-of-year means computed
  from the *training* data only (with fallback to global mean).
- Outputs:
  (1) station-level report with metrics, sizes, medoid coordinates; and
  (2) full per-row predictions:
      [station, date, latitude, longitude, altitude, y_obs, y_mod]

This module also exposes lightweight internal helpers that are reused by
``missclimatepy.impute``:

- :func:`_require_columns`
- :func:`_ensure_datetime_naive`
- :func:`_add_time_features`
- :func:`_build_neighbor_map_haversine`
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .models import make_model

try:
    from sklearn.neighbors import BallTree

    _HAS_SK_BALLTREE = True
except Exception:  # pragma: no cover
    _HAS_SK_BALLTREE = False

try:  # optional progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # transparent iterator


# --------------------------------------------------------------------------- #
# Internal helpers (also reused by impute.py)
# --------------------------------------------------------------------------- #


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    """
    Raise a ValueError if any of the requested columns is missing.

    This is intentionally strict and used early to provide clearer error
    messages to end users.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns include: {list(df.columns)[:12]}..."
        )


def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    """
    Ensure a naive (timezone-free) datetime64[ns] series.

    - Coerces parseable values with ``errors='coerce'``.
    - Drops timezones if present (tz-aware series are converted to naive).
    """
    s = pd.to_datetime(s, errors="coerce")
    # Avoid deprecated is_datetime64tz_dtype; check dtype class instead
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_localize(None)
    return s


def _add_time_features(
    df: pd.DataFrame,
    date_col: str,
    add_cyclic: bool = False,
) -> pd.DataFrame:
    """
    Add standard calendar features derived from ``date_col``:

    - year  (int32)
    - month (int16)
    - doy   (day of year, int16)

    If ``add_cyclic=True``, also adds:
    - doy_sin, doy_cos : sin/cos transforms of day-of-year on a 365.25-day cycle.
    """
    out = df.copy()
    out[date_col] = _ensure_datetime_naive(out[date_col])
    out["year"] = out[date_col].dt.year.astype("int32", copy=False)
    out["month"] = out[date_col].dt.month.astype("int16", copy=False)
    out["doy"] = out[date_col].dt.dayofyear.astype("int16", copy=False)
    if add_cyclic:
        two_pi = 2.0 * np.pi
        doy_arr = out["doy"].to_numpy()
        out["doy_sin"] = np.sin(two_pi * doy_arr / 365.25)
        out["doy_cos"] = np.cos(two_pi * doy_arr / 365.25)
    return out


def _rmse_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Simple RMSE implementation that stays robust to empty inputs.
    """
    if y_true.size == 0:
        return np.nan
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def _kge(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Kling–Gupta Efficiency (KGE).

    Returns NaN for degenerate or too-short series.
    """
    if y_true.size < 2 or y_pred.size < 2:
        return np.nan

    mu_o = float(np.mean(y_true))
    mu_p = float(np.mean(y_pred))
    if mu_o == 0.0:
        return np.nan

    std_o = float(np.std(y_true, ddof=1))
    std_p = float(np.std(y_pred, ddof=1))
    if std_o == 0.0 or std_p == 0.0:
        return np.nan

    r = float(np.corrcoef(y_true, y_pred)[0, 1])
    alpha = std_p / std_o
    beta = mu_p / mu_o

    return float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))


def _safe_metrics_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R², KGE robustamente; NaN en casos degenerados.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "KGE": np.nan}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse_manual(y_true, y_pred)

    if y_true.size < 2 or float(np.var(y_true)) == 0.0:
        r2 = np.nan
    else:
        r2 = float(r2_score(y_true, y_pred))

    kge = _kge(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "KGE": kge}


_FREQ_ALIAS = {"M": "MS", "A": "YS", "Y": "YS", "Q": "QS"}


def _freq_alias(freq: str) -> str:
    """
    Normalize frequency short-hands to start-period codes:

    - "M" → "MS" (month start)
    - "Y"/"A" → "YS" (year start)
    - "Q" → "QS" (quarter start)
    """
    return _FREQ_ALIAS.get(freq, freq)


def _agg_op(agg: str) -> str:
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg_for_metrics must be one of {'sum','mean','median'}.")
    return agg


def _aggregate_and_score(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    yhat_col: str,
    freq: str = "M",
    agg: str = "sum",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Aggregate predictions to a given temporal frequency and compute metrics.

    Parameters
    ----------
    df_pred : DataFrame
        Per-row predictions including columns [date_col, y_col, yhat_col].
    date_col : str
        Name of the datetime column.
    y_col, yhat_col : str
        Names of observed and predicted columns.
    freq : str
        Pandas offset alias (e.g., "M", "YS", "Q"). Short forms are normalized.
    agg : {"sum","mean","median"}
        Aggregation used for both observed and predicted series.

    Returns
    -------
    metrics : dict
        Dictionary with keys {"MAE","RMSE","R2","KGE"}.
    agg_df : DataFrame
        Aggregated table at the requested frequency.
    """
    freq = _freq_alias(freq=freq)
    op = _agg_op(agg)

    df = df_pred[[date_col, y_col, yhat_col]].dropna(subset=[date_col]).copy()
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "KGE": np.nan}, df

    df = df.set_index(date_col).sort_index()
    agg_df = df.resample(freq).agg({y_col: op, yhat_col: op}).dropna()
    if agg_df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "KGE": np.nan}, agg_df

    m = _safe_metrics_all(agg_df[y_col].values, agg_df[yhat_col].values)
    return m, agg_df


def _preprocess_once(
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
    Single-pass preprocessing:

    - Convert ``date_col`` to naive datetime.
    - Optionally clip data to [start, end].
    - Add calendar features.
    - Assemble and validate the feature list.

    Returns
    -------
    prepared_df : DataFrame
        Subset of columns needed for evaluation.
    features_list : list[str]
        Names of feature columns used for modeling.
    """
    df = data.copy()
    df[date_col] = _ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    df = _add_time_features(df, date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # Keep only the necessary columns (and ensure uniqueness)
    keep = sorted(set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats))
    df = df[keep]
    return df, feats


def _build_neighbor_map_haversine(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k: int,
    include_self: bool = False,
) -> Dict[Union[str, int], List[Union[str, int]]]:
    """
    Construct a neighbor map {station -> [neighbors...]} using Haversine KNN
    over per-station (median) centroids on (latitude, longitude).

    Parameters
    ----------
    df : DataFrame
        Prepared dataframe containing at least [id_col, lat_col, lon_col].
    id_col, lat_col, lon_col : str
        Column names for station id and geographic coordinates.
    k : int
        Number of neighbors to retain per station.
    include_self : bool
        If True, the station itself may appear in its neighbor list
        (and will be trimmed to exactly k). If False, self is removed.

    Returns
    -------
    dict
        Mapping from station id to list of neighbor station ids.
    """
    if not _HAS_SK_BALLTREE:
        raise ImportError("scikit-learn BallTree is required for Haversine neighbors.")

    centroids = (
        df.groupby(id_col)[[lat_col, lon_col]]
        .median()
        .reset_index()
    )

    lat_rad = np.deg2rad(centroids[lat_col].to_numpy())
    lon_rad = np.deg2rad(centroids[lon_col].to_numpy())
    mat = np.column_stack([lat_rad, lon_rad])

    tree = BallTree(mat, metric="haversine")
    query_k = int(k) + (1 if include_self else 0)
    _, ind = tree.query(mat, k=query_k)

    ids = centroids[id_col].tolist()
    neighbor_map: Dict[Union[str, int], List[Union[str, int]]] = {}

    for row_i, sid in enumerate(ids):
        row_idx = ind[row_i].tolist()
        neigh_ids = [ids[j] for j in row_idx]
        if not include_self:
            neigh_ids = [nid for nid in neigh_ids if nid != sid]
        neighbor_map[sid] = neigh_ids[: int(k)]

    return neighbor_map


def _append_rows_to_csv(
    rows: List[Dict],
    path: str,
    *,
    header_written_flag: Dict[str, bool],
) -> None:
    """
    Append a list of dict rows to CSV in one shot, writing the header only once.
    """
    if not rows or path is None:
        return
    tmp = pd.DataFrame(rows)
    first = not header_written_flag.get(path, False)
    tmp.to_csv(path, mode="a", index=False, header=first)
    header_written_flag[path] = True
    rows.clear()


# --------------------------------------------------------------------------- #
# Public evaluator
# --------------------------------------------------------------------------- #


def evaluate_stations(
    data: pd.DataFrame,
    *,
    # column names (generic schema)
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
    # selection
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    min_station_rows: Optional[int] = None,
    # neighborhood & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,       # 0 = exclude; 1..95 = include %
    include_target_seed: int = 42,
    # model & metrics
    model_kind: str = "rf",
    model_params: Optional[Dict[str, object]] = None,
    agg_for_metrics: str = "sum",
    # logging / UX
    show_progress: bool = False,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    # outputs
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one model per target station using either all other stations or
    only its K nearest neighbors as the training pool. Optionally include
    1..95% of the target station's valid rows in training.

    The inclusion of target rows is handled with a *precipitation-friendly*
    scheme: when ``include_target_pct > 0``, the target rows are sampled in a
    month × dry/wet stratified manner.

    Metrics are computed at three temporal scales (daily, monthly, annual) for
    both the chosen model and a Mean Climatology Model (MCM) based on day-of-
    year means from the training data.

    Returns
    -------
    (report_df, predictions_df)
        report_df : one row per evaluated station with metrics and metadata:
            [
              station, n_rows, seconds,
              rows_train, rows_test,
              used_k_neighbors, include_target_pct,
              MAE_d, RMSE_d, R2_d, KGE_d,
              MAE_m, RMSE_m, R2_m, KGE_m,
              MAE_y, RMSE_y, R2_y, KGE_y,
              MCM_MAE_d, MCM_RMSE_d, MCM_R2_d, MCM_KGE_d,
              MCM_MAE_m, MCM_RMSE_m, MCM_R2_m, MCM_KGE_m,
              MCM_MAE_y, MCM_RMSE_y, MCM_R2_y, MCM_KGE_y,
              latitude, longitude, altitude,
              model_kind
            ]

        predictions_df : per-row predictions:
            [station, date, latitude, longitude, altitude, y_obs, y_mod]
    """
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # 1) Preprocess once (datetime, clip, time features, features list)
    df, feats = _preprocess_once(
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

    # 2) Global valid mask for features+target (used for training / test splits)
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # 3) Determine stations to evaluate (OR over filters; default = all)
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

    # Unique station ids in stable order
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # 4) Optional minimum rows filter (based on observed target, NOT valid_mask)
    if min_station_rows is not None:
        obs_counts = (
            df.loc[~df[target_col].isna(), [id_col, target_col]]
            .groupby(id_col)[target_col]
            .size()
            .astype(int)
        )
        before = len(stations)
        stations = [
            s for s in stations if int(obs_counts.get(s, 0)) >= int(min_station_rows)
        ]
        if show_progress and before != len(stations):
            tqdm.write(
                f"Filtered by min_station_rows(observed)={int(min_station_rows)}: "
                f"{before} -> {len(stations)} stations"
            )

    # 5) Neighbor map
    used_k: Optional[int]
    if neighbor_map is not None:
        nmap = neighbor_map
        used_k = None  # user-provided list lengths may vary
    elif k_neighbors is not None:
        nmap = _build_neighbor_map_haversine(
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

    # 6) Station medoids for report
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

    # 7) Iterate over stations and fit/predict
    header_flag: Dict[str, bool] = {}
    pending_rows: List[Dict] = []
    rows_report: List[Dict] = []
    all_preds: List[pd.DataFrame] = []

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        # --- Target station valid rows (candidates for test and/or leakage) ---
        is_target = df[id_col] == sid
        target_valid = df.loc[is_target & valid_mask_global].copy()

        if target_valid.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid,
                "n_rows": 0,
                "seconds": sec,
                "rows_train": 0,
                "rows_test": 0,
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
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
                "MCM_MAE_d": np.nan,
                "MCM_RMSE_d": np.nan,
                "MCM_R2_d": np.nan,
                "MCM_KGE_d": np.nan,
                "MCM_MAE_m": np.nan,
                "MCM_RMSE_m": np.nan,
                "MCM_R2_m": np.nan,
                "MCM_KGE_m": np.nan,
                "MCM_MAE_y": np.nan,
                "MCM_RMSE_y": np.nan,
                "MCM_R2_y": np.nan,
                "MCM_KGE_y": np.nan,
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
                "model_kind": model_kind,
            }
            rows_report.append(row)
            pending_rows.append(row)
            if log_csv and len(pending_rows) >= flush_every:
                _append_rows_to_csv(
                    pending_rows, log_csv, header_written_flag=header_flag
                )
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid rows (skipped)")
            continue

        # --- Training pool: neighbors or all-other stations -------------------
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target)
        else:
            train_pool_mask = ~is_target
        train_pool = df.loc[train_pool_mask & valid_mask_global].copy()

        # --- Stratified inclusion from target station -------------------------
        pct = max(0.0, min(float(include_target_pct), 95.0))
        n_total = len(target_valid)

        if pct <= 0.0 or n_total <= 1:
            inc_target_df = target_valid.iloc[0:0]  # empty with same schema
            test_df = target_valid
        else:
            n_take = int(np.ceil(n_total * (pct / 100.0)))
            n_take = min(n_take, n_total - 1)  # at least one row must remain for test

            # Month × dry/wet strata: dry = target == 0, wet = target > 0
            target_valid["_month_"] = target_valid["month"].to_numpy()
            target_valid["_wet_"] = (target_valid[target_col].to_numpy() > 0.0)

            strata_groups: Dict[Tuple[int, bool], np.ndarray] = {}
            for key, sub in target_valid.groupby(["_month_", "_wet_"]):
                strata_groups[key] = sub.index.to_numpy()

            inc_indices: List[int] = []
            rng = np.random.RandomState(int(include_target_seed))

            for key, idx_arr in strata_groups.items():
                if len(inc_indices) >= n_take:
                    break
                n_stratum = idx_arr.size
                n_stratum_take = int(round(n_take * (n_stratum / n_total)))
                n_stratum_take = max(0, min(n_stratum_take, n_stratum))
                if n_stratum_take > 0:
                    chosen_idx = rng.choice(idx_arr, size=n_stratum_take, replace=False)
                    inc_indices.extend(chosen_idx.tolist())

            if len(inc_indices) > n_take:
                inc_indices = rng.choice(
                    np.array(inc_indices), size=n_take, replace=False
                ).tolist()
            elif len(inc_indices) < n_take:
                remaining = n_take - len(inc_indices)
                all_idx = target_valid.index.to_numpy()
                mask_chosen = np.isin(all_idx, np.array(inc_indices))
                pool_left = all_idx[~mask_chosen]
                if pool_left.size > 0:
                    extra = rng.choice(
                        pool_left, size=min(remaining, pool_left.size), replace=False
                    )
                    inc_indices.extend(extra.tolist())

            inc_index = pd.Index(sorted(set(inc_indices)))
            inc_target_df = target_valid.loc[inc_index].drop(columns=["_month_", "_wet_"])
            test_df = target_valid.drop(index=inc_index).drop(columns=["_month_", "_wet_"])

            if test_df.empty:
                move_idx = inc_index[-1]
                move_row = target_valid.loc[[move_idx]]
                inc_target_df = inc_target_df.drop(index=move_idx)
                test_df = pd.concat(
                    [test_df, move_row.drop(columns=["_month_", "_wet_"])],
                    axis=0,
                )

        # Final train = pool (neighbors/others) + included slice of the target
        train_df = pd.concat([train_pool, inc_target_df], axis=0, copy=False)

        if train_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid,
                "n_rows": 0,
                "seconds": sec,
                "rows_train": 0,
                "rows_test": int(len(test_df)),
                "used_k_neighbors": used_k,
                "include_target_pct": float(pct),
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
                "MCM_MAE_d": np.nan,
                "MCM_RMSE_d": np.nan,
                "MCM_R2_d": np.nan,
                "MCM_KGE_d": np.nan,
                "MCM_MAE_m": np.nan,
                "MCM_RMSE_m": np.nan,
                "MCM_R2_m": np.nan,
                "MCM_KGE_m": np.nan,
                "MCM_MAE_y": np.nan,
                "MCM_RMSE_y": np.nan,
                "MCM_R2_y": np.nan,
                "MCM_KGE_y": np.nan,
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
                "model_kind": model_kind,
            }
            rows_report.append(row)
            pending_rows.append(row)
            if log_csv and len(pending_rows) >= flush_every:
                _append_rows_to_csv(
                    pending_rows, log_csv, header_written_flag=header_flag
                )
            if show_progress:
                tqdm.write(f"Station {sid}: empty train (skipped)")
            continue

        # --- Fit model --------------------------------------------------------
        model = make_model(model_kind=model_kind, model_params=model_params)

        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[feats].to_numpy(copy=False)
        y_test = test_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # Per-row predictions with canonical names
        pred_df = pd.DataFrame(
            {
                "station": sid,
                date_col: test_df[date_col].values,
                "latitude": test_df[lat_col].astype(float).values,
                "longitude": test_df[lon_col].astype(float).values,
                "altitude": test_df[alt_col].astype(float).values,
                "y_obs": y_test,
                "y_mod": y_hat,
            }
        )

        # ------------------- MODEL METRICS -----------------------------------
        daily = _safe_metrics_all(pred_df["y_obs"].values, pred_df["y_mod"].values)

        monthly, _ = _aggregate_and_score(
            pred_df,
            date_col=date_col,
            y_col="y_obs",
            yhat_col="y_mod",
            freq="M",
            agg=agg_for_metrics,
        )
        annual, _ = _aggregate_and_score(
            pred_df,
            date_col=date_col,
            y_col="y_obs",
            yhat_col="y_mod",
            freq="YS",
            agg=agg_for_metrics,
        )

        # ------------------- BASELINE MCM (DOY) ------------------------------
        # Compute doy-based climatology from *training* data
        doy_means = (
            train_df.groupby("doy")[target_col]
            .mean()
            .astype(float, copy=False)
        )
        global_mean = float(train_df[target_col].mean()) if not train_df.empty else np.nan

        test_doy = test_df["doy"].to_numpy()
        mcm_vals = np.empty_like(test_doy, dtype=float)
        for i, d in enumerate(test_doy):
            v = float(doy_means.get(int(d), np.nan))
            if np.isnan(v):
                v = global_mean
            mcm_vals[i] = v

        mcm_daily = _safe_metrics_all(y_test, mcm_vals)

        mcm_df = pd.DataFrame(
            {
                date_col: test_df[date_col].values,
                "y_obs": y_test,
                "y_hat": mcm_vals,
            }
        )
        mcm_monthly, _ = _aggregate_and_score(
            mcm_df,
            date_col=date_col,
            y_col="y_obs",
            yhat_col="y_hat",
            freq="M",
            agg=agg_for_metrics,
        )
        mcm_annual, _ = _aggregate_and_score(
            mcm_df,
            date_col=date_col,
            y_col="y_obs",
            yhat_col="y_hat",
            freq="YS",
            agg=agg_for_metrics,
        )

        # ------------------- Assemble report row -----------------------------
        sec = pd.Timestamp.utcnow().timestamp() - t0
        row = {
            "station": sid,
            "n_rows": int(len(pred_df)),
            "seconds": float(sec),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "used_k_neighbors": used_k,
            "include_target_pct": float(pct),
            # model metrics
            "MAE_d": float(daily["MAE"]),
            "RMSE_d": float(daily["RMSE"]),
            "R2_d": float(daily["R2"]),
            "KGE_d": float(daily["KGE"]),
            "MAE_m": float(monthly["MAE"]),
            "RMSE_m": float(monthly["RMSE"]),
            "R2_m": float(monthly["R2"]),
            "KGE_m": float(monthly["KGE"]),
            "MAE_y": float(annual["MAE"]),
            "RMSE_y": float(annual["RMSE"]),
            "R2_y": float(annual["R2"]),
            "KGE_y": float(annual["KGE"]),
            # baseline MCM metrics
            "MCM_MAE_d": float(mcm_daily["MAE"]),
            "MCM_RMSE_d": float(mcm_daily["RMSE"]),
            "MCM_R2_d": float(mcm_daily["R2"]),
            "MCM_KGE_d": float(mcm_daily["KGE"]),
            "MCM_MAE_m": float(mcm_monthly["MAE"]),
            "MCM_RMSE_m": float(mcm_monthly["RMSE"]),
            "MCM_R2_m": float(mcm_monthly["R2"]),
            "MCM_KGE_m": float(mcm_monthly["KGE"]),
            "MCM_MAE_y": float(mcm_annual["MAE"]),
            "MCM_RMSE_y": float(mcm_annual["RMSE"]),
            "MCM_R2_y": float(mcm_annual["R2"]),
            "MCM_KGE_y": float(mcm_annual["KGE"]),
            # coords & model meta
            "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
            "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
            "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
            "model_kind": model_kind,
        }

        rows_report.append(row)
        pending_rows.append(row)
        all_preds.append(pred_df)

        if log_csv and len(pending_rows) >= flush_every:
            _append_rows_to_csv(
                pending_rows, log_csv, header_written_flag=header_flag
            )

        if show_progress:
            tqdm.write(
                f"{sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={daily['MAE']:.3f} RMSE_d={daily['RMSE']:.3f} "
                f"R2_d={daily['R2']:.3f} KGE_d={daily['KGE']:.3f}"
            )

    # Flush pending progress lines
    if log_csv and pending_rows:
        _append_rows_to_csv(pending_rows, log_csv, header_written_flag=header_flag)

    # Final report & predictions
    report = pd.DataFrame(rows_report)
    if all_preds:
        preds = pd.concat(all_preds, axis=0, ignore_index=True)
    else:
        preds = pd.DataFrame(
            columns=[
                "station",
                date_col,
                "latitude",
                "longitude",
                "altitude",
                "y_obs",
                "y_mod",
            ]
        )

    # Optional save (report only)
    if save_table_path:
        ext = str(save_table_path).lower()
        if ext.endswith(".csv"):
            report.to_csv(save_table_path, index=False)
        elif ext.endswith(".parquet"):
            report.to_parquet(save_table_path, index=False, compression=parquet_compression)
        else:
            report.to_csv(save_table_path, index=False)

    # Sort by daily RMSE (ascending) for convenience
    if not report.empty and "RMSE_d" in report.columns:
        report = report.sort_values("RMSE_d", ascending=True).reset_index(drop=True)

    return report, preds


__all__ = [
    "evaluate_stations",
    "_require_columns",
    "_ensure_datetime_naive",
    "_add_time_features",
    "_build_neighbor_map_haversine",
]
