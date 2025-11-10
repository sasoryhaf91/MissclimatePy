# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise evaluation for climate-data imputation using ONLY spatial
coordinates (latitude, longitude, altitude) and calendar features
(year, month, day-of-year, optional cyclic sin/cos). One Random-Forest
model is trained per target station using either all other stations or
its K nearest neighbors (by haversine on lat/lon), with **controlled,
stratified inclusion** of a fraction of the target station's valid rows.

Key fixes vs. earlier drafts
----------------------------
- Leak-free split with **per-station deterministic stratified inclusion**.
- Inclusion is **by bin** (default: month), ensuring every bin keeps
  at least `min_test_per_bin` holdouts and the station keeps at least
  `min_test_rows` test samples overall.
- `min_station_rows` filters **observed valid rows** in the analysis window.

Outputs
-------
1) Station-level report (daily + monthly + yearly metrics).
2) Per-row predictions for the **held-out** target rows only:
   [station, date, latitude, longitude, altitude, y_obs, y_mod]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

try:
    from sklearn.neighbors import BallTree
    _HAS_SK_BALLTREE = True
except Exception:  # pragma: no cover
    _HAS_SK_BALLTREE = False

try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # fallback: transparent iterator


# --------------------------------------------------------------------------- #
# Hyperparameters container
# --------------------------------------------------------------------------- #

@dataclass
class RFParams:
    """
    Convenience container for RandomForestRegressor hyperparameters.
    """
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, int, float, None] = "sqrt"
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns include: {list(df.columns)[:12]}..."
        )

def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    # Avoid deprecated dtype helpers – check class directly
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_localize(None)
    return s

def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["year"]  = out[date_col].dt.year.astype("int32", copy=False)
    out["month"] = out[date_col].dt.month.astype("int16", copy=False)
    out["doy"]   = out[date_col].dt.dayofyear.astype("int16", copy=False)
    if add_cyclic:
        out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"].to_numpy() / 365.25)
        out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"].to_numpy() / 365.25)
    return out

def _rmse_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return np.nan
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))

def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse_manual(y_true, y_pred)
    if y_true.size < 2 or float(np.var(y_true)) == 0.0:
        r2 = np.nan
    else:
        r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

_FREQ_ALIAS = {"M": "MS", "A": "YS", "Y": "YS", "Q": "QS"}

def _freq_alias(freq: str) -> str:
    return _FREQ_ALIAS.get(freq, freq)

def _aggregate_and_score(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    yhat_col: str,
    freq: str = "M",
    agg: str = "sum",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    freq = _freq_alias(freq=freq)
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg_for_metrics must be in {'sum','mean','median'}.")

    df = df_pred[[date_col, y_col, yhat_col]].dropna(subset=[date_col]).copy()
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, df

    df = df.set_index(date_col).sort_index()
    agg_df = df.resample(freq).agg({y_col: agg, yhat_col: agg}).dropna()
    if agg_df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, agg_df

    m = _safe_metrics(agg_df[y_col].values, agg_df[yhat_col].values)
    return m, agg_df

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
    Neighbor map {station -> [neighbors...]} using Haversine KNN over
    per-station (median) coordinates.
    """
    if not _HAS_SK_BALLTREE:
        raise ImportError("scikit-learn BallTree is required for Haversine neighbors.")

    centroids = df.groupby(id_col)[[lat_col, lon_col]].median().reset_index()

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

def _station_random_state(seed: int, sid: Union[str, int]) -> np.random.RandomState:
    """
    Build a deterministic RandomState per station so runs are reproducible
    and independent across stations.
    """
    # Hash sid into an int32 and combine with seed
    sid_hash = abs(hash(str(sid))) % (2**31 - 1)
    return np.random.RandomState(seed ^ sid_hash)

def _stratified_take_indices(
    df: pd.DataFrame,
    *,
    bin_col: str,
    take_frac: float,
    min_test_per_bin: int,
    min_test_rows: int,
    rs: np.random.RandomState,
) -> Tuple[pd.Index, pd.Index, float]:
    """
    Choose indices to *include* (for training) in a stratified manner, keeping
    at least `min_test_per_bin` rows per bin and at least `min_test_rows` total
    in the holdout. Returns (include_idx, test_idx, actual_inclusion_pct).
    """
    # Group by bin (e.g., month)
    groups = df.groupby(bin_col)
    include_idx_parts: List[pd.Index] = []
    test_idx_parts: List[pd.Index] = []

    for _, grp in groups:
        n = len(grp)
        if n <= min_test_per_bin:
            # keep all in test for this bin
            test_idx_parts.append(grp.index)
            continue

        # nominal number to include from this bin
        n_take = int(np.floor(take_frac * n))
        # ensure at least min_test_per_bin left for test
        n_take = min(n_take, n - min_test_per_bin)
        if n_take <= 0:
            test_idx_parts.append(grp.index)
            continue

        take_idx = grp.sample(n=n_take, random_state=rs).index
        keep_test = grp.index.difference(take_idx)

        include_idx_parts.append(take_idx)
        test_idx_parts.append(keep_test)

    include_idx = pd.Index([]).append(include_idx_parts) if include_idx_parts else pd.Index([])
    test_idx    = pd.Index([]).append(test_idx_parts)    if test_idx_parts    else pd.Index([])

    # Guarantee overall minimum holdout size
    if len(test_idx) < min_test_rows and len(df) > 0:
        # Move some included rows back to test to meet the floor
        need = min_test_rows - len(test_idx)
        give_back = include_idx.to_series().sample(n=min(need, len(include_idx)), random_state=rs).index
        include_idx = include_idx.difference(give_back)
        test_idx = test_idx.union(give_back)

    actual_pct = 100.0 * (len(include_idx) / max(len(df), 1))
    return include_idx, test_idx, actual_pct


# --------------------------------------------------------------------------- #
# Public evaluator
# --------------------------------------------------------------------------- #

def evaluate_stations(
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
    # station selection (OR semantics)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # MDR filter (observed *valid* rows in window)
    min_station_rows: Optional[int] = None,
    # neighborhood & inclusion
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,     # 0..95 (desired)
    include_target_seed: int = 42,
    stratify_by: Optional[str] = "month",  # None | "month" | "doy" | "year"
    min_test_per_bin: int = 3,
    min_test_rows: int = 365,
    # model & metrics
    rf_params: Optional[Union[RFParams, Dict]] = None,
    agg_for_metrics: str = "sum",
    # UX / logging
    show_progress: bool = False,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    # outputs
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate one Random-Forest model per target station. The training pool is
    either the K haversine neighbors or *all other* stations. A user-selected
    percentage (0..95%) of the target station's **valid** rows is included in
    training via **stratified sampling** (default by month). Those included
    rows are *excluded* from the test set to avoid leakage.

    Returns
    -------
    (report_df, predictions_df)
      report_df columns:
        [station, n_rows, seconds, rows_train, rows_test,
         MAE_d, RMSE_d, R2_d, MAE_m, RMSE_m, R2_m, MAE_y, RMSE_y, R2_y,
         used_k_neighbors, include_target_pct,
         latitude, longitude, altitude]
      predictions_df columns:
        [station, date, latitude, longitude, altitude, y_obs, y_mod]
    """
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # --- preprocess once: window + features
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
    feats = list(dict.fromkeys(feats))  # ensure unique names

    keep = sorted(set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats))
    df = df[keep]

    # valid = rows with all features + target present
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # --- station selection
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
    # stable unique
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # --- MDR (observed valid rows in window)
    if min_station_rows is not None:
        valid_counts = df.loc[valid_mask_global, [id_col]].groupby(id_col).size().astype(int)
        before = len(stations)
        stations = [s for s in stations if int(valid_counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress and before != len(stations):
            tqdm.write(
                f"Filtered by min_station_rows(observed)={int(min_station_rows)}: "
                f"{before} -> {len(stations)} stations"
            )

    if not stations:
        return (
            pd.DataFrame(columns=[
                "station","n_rows","seconds","rows_train","rows_test",
                "MAE_d","RMSE_d","R2_d","MAE_m","RMSE_m","R2_m","MAE_y","RMSE_y","R2_y",
                "used_k_neighbors","include_target_pct","latitude","longitude","altitude"
            ]),
            pd.DataFrame(columns=[id_col, date_col, "latitude","longitude","altitude","y_obs","y_mod"])
        )

    # --- neighbor map
    used_k = None
    if neighbor_map is not None:
        nmap = neighbor_map
    elif k_neighbors is not None:
        nmap = _build_neighbor_map_haversine(
            df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k=int(k_neighbors), include_self=False
        )
        used_k = int(k_neighbors)
    else:
        nmap = None

    # --- RF params
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # --- medoids for reporting/preds
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median()
        .rename(columns={lat_col: "latitude", lon_col: "longitude", alt_col: "altitude"})
    )

    rows_report: List[Dict] = []
    all_preds: List[pd.DataFrame] = []
    #header_flag: Dict[str, bool] = {}
    #pending_rows: List[Dict] = []

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        is_target = (df[id_col] == sid)
        target_valid = df.loc[is_target & valid_mask_global].copy()
        if target_valid.empty:
            # nothing to test
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": 0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k, "include_target_pct": 0.0,
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
            }
            rows_report.append(row)
            continue

        # training pool: neighbors or all-others (always exclude target)
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            pool_mask = df[id_col].isin(neigh_ids) & (~is_target) & valid_mask_global
        else:
            pool_mask = (~is_target) & valid_mask_global
        train_pool = df.loc[pool_mask]

        # ---- stratified inclusion of target rows (no leakage)
        desired_pct = float(max(0.0, min(include_target_pct, 95.0)))
        rs = _station_random_state(int(include_target_seed), sid)

        if desired_pct <= 0.0:
            include_idx = pd.Index([])
            test_idx = target_valid.index
            actual_incl = 0.0
        else:
            if stratify_by in {"month", "doy", "year"}:
                bin_col = {"month": "month", "doy": "doy", "year": "year"}[stratify_by]
            else:
                bin_col = "month"  # default
            include_idx, test_idx, actual_incl = _stratified_take_indices(
                target_valid, bin_col=bin_col, take_frac=desired_pct/100.0,
                min_test_per_bin=int(min_test_per_bin), min_test_rows=int(min_test_rows), rs=rs
            )

        inc_target_df = target_valid.loc[include_idx]
        test_df       = target_valid.loc[test_idx].copy()

        # final train = pool + included slice from target
        train_df = pd.concat([train_pool, inc_target_df], axis=0, copy=False)

        if train_df.empty or test_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid,
                "n_rows": int(len(test_df)),
                "seconds": float(sec),
                "rows_train": int(len(train_df)),
                "rows_test": int(len(test_df)),
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(actual_incl),
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
            }
            rows_report.append(row)
            if show_progress:
                tqdm.write(f"{sid}: insufficient train/test after split (train={len(train_df)}, test={len(test_df)})")
            continue

        # ---- fit & predict
        model = RandomForestRegressor(**rf_kwargs)
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test  = test_df[feats].to_numpy(copy=False)
        y_test  = test_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # per-row predictions (held out only)
        pred_df = pd.DataFrame({
            "station": sid,
            date_col: test_df[date_col].values,
            "latitude": test_df[lat_col].astype(float).values,
            "longitude": test_df[lon_col].astype(float).values,
            "altitude": test_df[alt_col].astype(float).values,
            "y_obs": y_test,
            "y_mod": y_hat,
        })
        all_preds.append(pred_df)

        # metrics
        daily = _safe_metrics(pred_df["y_obs"].values, pred_df["y_mod"].values)
        monthly, _ = _aggregate_and_score(pred_df, date_col=date_col, y_col="y_obs", yhat_col="y_mod",
                                          freq="M", agg=agg_for_metrics)
        annual, _  = _aggregate_and_score(pred_df, date_col=date_col, y_col="y_obs", yhat_col="y_mod",
                                          freq="YS", agg=agg_for_metrics)

        sec = pd.Timestamp.utcnow().timestamp() - t0
        row = {
            "station": sid,
            "n_rows": int(len(pred_df)),
            "seconds": float(sec),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "MAE_d": float(daily["MAE"]),
            "RMSE_d": float(daily["RMSE"]),
            "R2_d": float(daily["R2"]) if daily["R2"] == daily["R2"] else np.nan,
            "MAE_m": float(monthly["MAE"]),
            "RMSE_m": float(monthly["RMSE"]),
            "R2_m": float(monthly["R2"]) if monthly["R2"] == monthly["R2"] else np.nan,
            "MAE_y": float(annual["MAE"]),
            "RMSE_y": float(annual["RMSE"]),
            "R2_y": float(annual["R2"]) if annual["R2"] == annual["R2"] else np.nan,
            "used_k_neighbors": used_k,
            "include_target_pct": float(actual_incl),
            "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
            "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
            "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
        }
        rows_report.append(row)

        if show_progress:
            holdout_pct = 100.0 * (len(test_df) / max(len(test_df) + len(inc_target_df), 1))
            tqdm.write(
                f"{sid}: {sec:.2f}s  (train={len(train_df):,}  test={len(test_df):,}  "
                f"incl={actual_incl:.1f}%  holdout={holdout_pct:.1f}%  k={used_k if used_k is not None else 'all'})  "
                f"MAE_d={daily['MAE']:.3f}  RMSE_d={daily['RMSE']:.3f}  R2_d={daily['R2'] if daily['R2']==daily['R2'] else float('nan'):.3f}"
            )

    report = pd.DataFrame(rows_report)
    preds = pd.concat(all_preds, axis=0, ignore_index=True) if all_preds else pd.DataFrame(
        columns=[id_col, date_col, "latitude", "longitude", "altitude", "y_obs", "y_mod"]
    )

    if not report.empty and "RMSE_d" in report.columns:
        report = report.sort_values("RMSE_d", ascending=True).reset_index(drop=True)

    # optional save (report only)
    if save_table_path:
        ext = str(save_table_path).lower()
        if ext.endswith(".csv"):
            report.to_csv(save_table_path, index=False)
        elif ext.endswith(".parquet"):
            report.to_parquet(save_table_path, index=False, compression=parquet_compression)
        else:
            report.to_csv(save_table_path, index=False)

    return report, preds


__all__ = ["RFParams", "evaluate_stations"]
