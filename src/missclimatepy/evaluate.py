# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Evaluation utilities for local Random-Forest imputation using only
spatio-temporal features (latitude, longitude, altitude, and calendar signals).

Public API
----------
- RFParams
- evaluate_stations(...)
- evaluate_all_stations_fast  (alias for backward compatibility)

Design
------
For each target station, a RandomForestRegressor is trained using:
  * all rows from K nearest neighbor stations (by centroid; Haversine),
  * plus an optional percentage of valid rows from the target itself
    (to study Minimum Data Requirement / LOSO-like behavior).

Predictions are produced ONLY for the target station's valid rows
(so metrics reflect how well neighbors (+optional inclusion %) explain the target).

Returned outputs:
  * A per-station report with daily, monthly, and annual metrics (MAE/RMSE/R²),
    training/testing row counts, wall-clock time, K used, inclusion %, and
    representative latitude/longitude/altitude for the evaluated station.
  * Optionally, a long table with row-level predictions:
    [station, date, latitude, longitude, altitude, y_obs, y_mod].

This file is intentionally dependency-light (numpy, pandas, scikit-learn, tqdm optional).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import BallTree

try:  # nice progress bars if available
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x


# --------------------------------------------------------------------------- #
# Types & hyperparameter container
# --------------------------------------------------------------------------- #

StationId = Union[str, int]


@dataclass
class RFParams:
    """Hyperparameters for the RandomForestRegressor.

    Notes
    -----
    - ``max_features`` default is None (sklearn≥1.3 recommends None/“sqrt”/“log2”).
    - Keep this minimal to preserve the "minimal, reproducible" philosophy.
    """
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, int, float, None] = None
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    # New, deprecation-free check:
    if isinstance(s.dtype, pd.DatetimeTZDtype):  # tz-aware datetimes
        s = s.dt.tz_localize(None)
    return s


def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Robust metric computation (handles empty inputs and zero-variance)."""
    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))  # avoid squared=
    if y_true.size < 2 or float(np.var(y_true)) == 0.0:
        r2 = np.nan
    else:
        r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


# resample aliases (human-friendly)
_FREQ_ALIAS = {"M": "ME", "A": "YE", "Y": "YE", "Q": "QE"}


def _aggregate_and_score(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    yhat_col: str,
    freq: str,
    agg: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Aggregate to a frequency and score there (sum/mean/median)."""
    freq = _FREQ_ALIAS.get(freq, freq)
    op = {"sum": "sum", "mean": "mean", "median": "median"}[agg]
    df = df_pred[[date_col, y_col, yhat_col]].dropna(subset=[date_col]).copy()
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, df
    df = df.set_index(date_col).sort_index()
    agg_df = df.resample(freq).agg({y_col: op, yhat_col: op}).dropna()
    if agg_df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, agg_df
    m = _safe_metrics(agg_df[y_col].values, agg_df[yhat_col].values)
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
    """Filter by period, add time features, and decide feature set."""
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

    keep = sorted(set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats))
    df = df[keep]
    return df, feats


def _build_neighbor_map(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k: int,
) -> Dict[StationId, List[StationId]]:
    """Return K nearest neighbors (by station centroid lat/lon) using Haversine BallTree.

    Self is excluded; exactly K neighbors are returned (if possible).
    """
    centroids = (
        df.groupby(id_col)[[lat_col, lon_col]].median().reset_index()
        .rename(columns={id_col: "station", lat_col: "lat", lon_col: "lon"})
    )
    lat_rad = np.deg2rad(centroids["lat"].to_numpy())
    lon_rad = np.deg2rad(centroids["lon"].to_numpy())
    coords_rad = np.c_[lat_rad, lon_rad]

    tree = BallTree(coords_rad, metric="haversine")
    dist, idx = tree.query(coords_rad, k=min(k + 1, len(centroids)))  # +1 for potential self
    stations = centroids["station"].tolist()

    nmap: Dict[StationId, List[StationId]] = {}
    for row_i, st in enumerate(stations):
        neighs = [stations[j] for j in idx[row_i].tolist()]
        # remove self if present, then trim to k
        neighs = [x for x in neighs if x != st][:k]
        nmap[st] = neighs
    return nmap


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
    # station selection
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[StationId]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[StationId], bool]] = None,
    min_station_rows: Optional[int] = None,
    # neighborhood & target inclusion
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[StationId, List[StationId]]] = None,
    include_target_pct: float = 0.0,       # 0 = exclude (pure LOSO), 1..95 = include %
    include_target_seed: int = 42,
    # model & metrics
    rf_params: Optional[Union[RFParams, Dict]] = None,
    agg_for_metrics: str = "sum",          # aggregation for monthly/annual metrics
    # UX
    show_progress: bool = False,
    # outputs
    return_predictions: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Evaluate local RF models station-by-station.

    Parameters
    ----------
    data : DataFrame
        Input table with columns: id, date, latitude, longitude, altitude, target.
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station ID, timestamp, coordinates, and target variable.
    start, end : str or None
        Optional period filter (inclusive).
    add_cyclic : bool
        If True, adds sin/cos of day-of-year as extra features.
    feature_cols : sequence of str or None
        Custom feature set. If None, defaults to [lat, lon, alt, year, month, doy, (opt) sin/cos].
    prefix, station_ids, regex, custom_filter : selectors
        Restrict stations to evaluate.
    min_station_rows : int or None
        Minimum number of **valid** rows (features+target not NaN) required for a station to be evaluated.
    k_neighbors : int or None
        Number of neighbor stations used for training. If None, use all stations except the target.
    neighbor_map : dict or None
        Precomputed neighbor list per station (overrides k_neighbors if provided).
    include_target_pct : float
        Percentage (0..95) of the target station's valid rows included in TRAIN (rest remain TEST).
    include_target_seed : int
        Random seed for sampling the target rows to include.
    rf_params : RFParams or dict or None
        RF hyperparameters. If None, defaults from RFParams().
    agg_for_metrics : {"sum","mean","median"}
        Aggregation used for monthly/annual metrics before scoring.
    show_progress : bool
        If True, prints textual progress (per station).
    return_predictions : bool
        If True, return (report, preds). Otherwise return report only.

    Returns
    -------
    report : DataFrame
        One row per evaluated station with:
        ['station','n_rows','seconds','rows_train','rows_test',
         'MAE_d','RMSE_d','R2_d','MAE_m','RMSE_m','R2_m','MAE_y','RMSE_y','R2_y',
         'used_k_neighbors','include_target_pct','latitude','longitude','altitude']
    (report, preds) : tuple
        If return_predictions=True, also return a long table with:
        ['station','date','latitude','longitude','altitude','y_obs','y_mod']
    """
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # 1) Preprocess once
    df, feats = _preprocess_once(
        data,
        id_col=id_col, date_col=date_col,
        lat_col=lat_col, lon_col=lon_col, alt_col=alt_col,
        target_col=target_col,
        start=start, end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
    )

    # 2) Global validity mask (features + target not NaN)
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # 3) Station universe and subset selection
    all_ids = df[id_col].dropna().unique().tolist()
    chosen: List[StationId] = []

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

    # unique while preserving order
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # 4) Optional filter by minimum valid rows
    if min_station_rows is not None:
        valid_counts = df.loc[valid_mask_global, [id_col]].groupby(id_col).size().astype(int)
        before = len(stations)
        stations = [s for s in stations if int(valid_counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress and before != len(stations):
            tqdm.write(f"Filtered by min_station_rows={min_station_rows}: {before} → {len(stations)} stations")

    # 5) Neighbor map
    used_k = None
    if neighbor_map is not None:
        nmap = neighbor_map
    elif k_neighbors is not None:
        nmap = _build_neighbor_map(df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k=int(k_neighbors))
        used_k = int(k_neighbors)
    else:
        nmap = None

    # 6) RF hyperparameters
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:  # dict-like
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # 7) Loop over stations
    rows: List[Dict] = []
    preds_rows: List[Dict] = []

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        is_target = (df[id_col] == sid)

        # TEST = all valid rows from the target station
        test_mask = is_target & valid_mask_global
        test_df = df.loc[test_mask]
        if test_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            rows.append({
                "station": sid, "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": 0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
                "latitude": np.nan, "longitude": np.nan, "altitude": np.nan,
            })
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid test rows (skipped)")
            continue

        # TRAIN = neighbors (or all but target) valid rows
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target)
        else:
            train_pool_mask = ~is_target
        train_pool = df.loc[train_pool_mask & valid_mask_global]

        # Controlled inclusion of target rows in TRAIN
        pct = max(0.0, min(float(include_target_pct), 95.0))
        if pct > 0.0:
            n_take = int(np.ceil(len(test_df) * (pct / 100.0)))
            sample_idx = test_df.sample(n=n_take, random_state=int(include_target_seed)).index
            train_df = pd.concat([train_pool, df.loc[sample_idx]], axis=0, copy=False)
        else:
            train_df = train_pool

        if train_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            rows.append({
                "station": sid, "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": int(len(test_df)),
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
                "latitude": float(test_df[lat_col].median()),
                "longitude": float(test_df[lon_col].median()),
                "altitude": float(test_df[alt_col].median()),
            })
            if show_progress:
                tqdm.write(f"Station {sid}: empty train (skipped)")
            continue

        # Fit RF
        model = RandomForestRegressor(**rf_kwargs)
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[feats].to_numpy(copy=False)
        y_test = test_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # Metrics: daily + aggregated (monthly, annual)
        pred_df = pd.DataFrame({
            date_col: test_df[date_col].values,
            "y_true": y_test,
            "y_pred": y_hat,
        })
        daily = _safe_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
        monthly, _ = _aggregate_and_score(
            pred_df, date_col=date_col, y_col="y_true", yhat_col="y_pred", freq="M", agg=agg_for_metrics
        )
        annual, _ = _aggregate_and_score(
            pred_df, date_col=date_col, y_col="y_true", yhat_col="y_pred", freq="YE", agg=agg_for_metrics
        )

        sec = pd.Timestamp.utcnow().timestamp() - t0

        # Representative coordinates (median over target test rows)
        rep_lat = float(test_df[lat_col].median())
        rep_lon = float(test_df[lon_col].median())
        rep_alt = float(test_df[alt_col].median())

        rows.append({
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
            "used_k_neighbors": k_neighbors if neighbor_map is None else None,
            "include_target_pct": float(pct),
            "latitude": rep_lat,
            "longitude": rep_lon,
            "altitude": rep_alt,
        })

        if return_predictions:
            preds_rows.append(pd.DataFrame({
                "station": sid,
                "date": test_df[date_col].values,
                "latitude": test_df[lat_col].values,
                "longitude": test_df[lon_col].values,
                "altitude": test_df[alt_col].values,
                "y_obs": y_test,
                "y_mod": y_hat,
            }))

        if show_progress:
            tqdm.write(
                f"Station {sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={daily['MAE']:.3f} RMSE_d={daily['RMSE']:.3f} R2_d={daily['R2']:.3f}"
            )

    # 8) Build outputs
    report = pd.DataFrame(rows)
    if not report.empty and "RMSE_d" in report.columns:
        report = report.sort_values("RMSE_d", ascending=True).reset_index(drop=True)

    if return_predictions:
        preds = (
            pd.concat(preds_rows, ignore_index=True)
            if preds_rows
            else pd.DataFrame(columns=["station", "date", "latitude", "longitude", "altitude", "y_obs", "y_mod"])
        )
        return report, preds

    return report


# Backward-compat alias (keeps existing notebooks/tests working)
evaluate_all_stations_fast = evaluate_stations

__all__ = [
    "RFParams",
    "evaluate_stations",
    "evaluate_all_stations_fast",
]
