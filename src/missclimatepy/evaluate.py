# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise evaluation with controlled inclusion of the target station and
optional K-neighborhood training pools.

This module provides:
- A small `RFParams` dataclass to declare RandomForest hyperparameters.
- A single-pass, memory-lean evaluator: `evaluate_all_stations_fast`.

Design goals
------------
- **One model per target station**. The model is trained either on all
  stations except the target or on its K nearest neighbors (if provided).
- **Controlled inclusion**. Optionally leak 1..95% of valid rows from the
  target station into training for better local adaptation. With `0.0`, the
  target is excluded (LOSO-like behavior).
- **Daily metrics prioritized**. We compute daily MAE/RMSE/R² on observed
  overlap; monthly/annual metrics are provided for completeness.
- **Generic schema**. Column names are *parameters* (id, date, coords, altitude,
  target), so the package does not enforce a specific naming convention.
- **Preprocessing once**. Datetime parsing, clipping and time features are
  computed once and reused.

Typical usage
-------------
>>> from missclimatepy.evaluate import evaluate_all_stations_fast, RFParams
>>> report = evaluate_all_stations_fast(
...     df,
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     start="1991-01-01", end="2020-12-31",
...     station_ids=["2038","2124","29007"],      # evaluate only these
...     k_neighbors=20,                           # train on K neighbors of each target
...     include_target_pct=30.0,                  # leak 30% target rows into training
...     min_station_rows=9125,                    # require this many valid rows to score
...     rf_params=RFParams(n_estimators=15, max_depth=30, n_jobs=-1, random_state=42),
...     show_progress=True,
... )
>>> report.head()

Notes
-----
- Monthly/annual metrics use aggregation `agg_for_metrics` ("sum" | "mean" | "median").
- If you pass `neighbor_map`, it overrides `k_neighbors`.
- Memory: features and masks are built once; per-station we subset views and fit small RFs.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from tqdm.auto import tqdm  # optional progress bars
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # fallback: transparent iterator

# Local neighbor utilities
from .neighbors import neighbor_distances

StationId = Union[str, int]


# --------------------------------------------------------------------------- #
# Public hyperparameters container
# --------------------------------------------------------------------------- #
@dataclass
class RFParams:
    """Typed RF hyperparameters for convenience (all optional)."""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    # scikit-learn >=1.2 deprecates "auto" for RandomForestRegressor
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
            f"Available: {list(df.columns)[:10]}..."
        )


def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(s):
        s = s.dt.tz_localize(None)
    return s


def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE/RMSE/R² robustly; R² = NaN for degenerate/short cases."""
    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if y_true.size < 2 or float(np.var(y_true)) == 0.0:
        r2 = np.nan
    else:
        r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


_FREQ_ALIAS = {"M": "ME", "A": "YE", "Y": "YE", "Q": "QE"}


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
    Aggregate to a given frequency and score on the overlap.
    Returns (metrics_dict, aggregated_dataframe).
    """
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
    """
    Single-pass preprocessing: datetime, optional clipping, time features,
    and feature list assembly. Returns (prepared_df, features_list).
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

    keep = sorted(set([id_col, date_col, lat_col, lon_col, alt_col, target_col] + feats))
    df = df[keep]
    return df, feats


def _build_neighbor_map_from_centroids(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k: int,
    include_self: bool = False,
) -> Dict[StationId, List[StationId]]:
    """
    Construct a neighbor map {station -> [neighbors...]} using Haversine KNN
    over per-station (median) centroids.
    """
    centroids = (
        df.groupby(id_col)[[lat_col, lon_col]]
        .median()
        .reset_index()
        .rename(columns={id_col: "station", lat_col: "latitude", lon_col: "longitude"})
    )
    tbl = neighbor_distances(
        stations=centroids,
        k_neighbors=k,
        include_self=include_self,
    )
    nmap: Dict[StationId, List[StationId]] = {}
    for st, sub in tbl.groupby("station"):
        nmap[st] = sub.sort_values("rank")["neighbor"].tolist()
    return nmap


def _append_rows_to_csv(rows: List[Dict], path: str, *, header_written_flag: Dict[str, bool]) -> None:
    if not rows or path is None:
        return
    df_tmp = pd.DataFrame(rows)
    first = (not header_written_flag.get(path, False))
    df_tmp.to_csv(path, mode="a", index=False, header=first)
    header_written_flag[path] = True


# --------------------------------------------------------------------------- #
# Public evaluator
# --------------------------------------------------------------------------- #
def evaluate_all_stations_fast(
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
    station_ids: Optional[Iterable[StationId]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[callable] = None,
    min_station_rows: Optional[int] = None,
    # neighborhood & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[StationId, List[StationId]]] = None,
    include_target_pct: float = 0.0,       # 0 = exclude; 1..95 = include %
    include_target_seed: int = 42,
    # model & metrics
    rf_params: Optional[Union[RFParams, Dict]] = None,
    agg_for_metrics: str = "sum",
    # logging / UX
    show_progress: bool = False,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    # output
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
) -> pd.DataFrame:
    """
    Evaluate one Random-Forest model per target station using either all other
    stations or only its K nearest neighbors as the training pool. Optionally
    include 1..95% of the target station's valid rows in training.

    Parameters
    ----------
    data : DataFrame
        Long-format table with at least (id_col, date_col, lat_col, lon_col,
        alt_col, target_col).
    start, end : str or None
        Optional inclusive window for analysis.
    add_cyclic : bool
        Add sin/cos seasonal features of day-of-year.
    feature_cols : list[str] or None
        Custom features to use. If None, defaults to coords + calendar (+cyclic).
    prefix, station_ids, regex, custom_filter :
        Subset of station ids to evaluate (OR semantics across filters). Training
        always uses the full pool (except target), or K-neighbors if configured.
    min_station_rows : int or None
        Require at least this number of valid rows per station to compute metrics.
    k_neighbors : int or None
        If provided and `neighbor_map` is None, builds a KNN map and *trains each
        target model only on its neighbors* (excluding itself), plus optionally the
        leaked fraction of target rows.
    neighbor_map : dict or None
        Precomputed `{station -> [neighbors...]}`. If provided, it overrides
        `k_neighbors`.
    include_target_pct : float
        Percentage (0..95) of target station valid rows to *include in training*.
        Use 0.0 to fully exclude target (LOSO-like).
    include_target_seed : int
        Random seed for sampling target rows when inclusion > 0.
    rf_params : RFParams | dict | None
        RF hyperparameters. Missing fields fall back to defaults.
    agg_for_metrics : {"sum","mean","median"}
        Aggregation for monthly/annual metrics.
    show_progress : bool
        Print per-station progress info.
    log_csv : str or None
        If provided, appends progress rows every `flush_every` stations.
    save_table_path : str or None
        If provided, saves the final report (.csv or .parquet depending on extension).

    Returns
    -------
    DataFrame
        One row per evaluated station with metrics and metadata:
        [station, n_rows, seconds, rows_train, rows_test,
         MAE_d, RMSE_d, R2_d, MAE_m, RMSE_m, R2_m, MAE_y, RMSE_y, R2_y,
         used_k_neighbors, include_target_pct, neighbors_ids (optional)]
    """
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # 1) Preprocess once (datetime, clip, time features, features list)
    df, feats = _preprocess_once(
        data,
        id_col=id_col, date_col=date_col,
        lat_col=lat_col, lon_col=lon_col, alt_col=alt_col,
        target_col=target_col,
        start=start, end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
    )

    # 2) Global valid mask for features+target (used to speed-up per-station slices)
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # 3) Determine stations to evaluate (OR over filters; default = all)
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

    # Keep unique & stable order
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # 4) Optional minimum rows filter (counted on valid rows)
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
        used_k = None  # user-provided list lengths may vary
    elif k_neighbors is not None:
        nmap = _build_neighbor_map_from_centroids(df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k=int(k_neighbors))
        used_k = int(k_neighbors)
    else:
        nmap = None
        used_k = None

    # 6) RF params (dict or dataclass)
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:
        # dict-like; merge with defaults
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # 7) Iterate over stations and fit/predict
    header_flag: Dict[str, bool] = {}
    pending: List[Dict] = []
    rows: List[Dict] = []

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        # Target & test subset (valid rows only)
        is_target = (df[id_col] == sid)
        test_mask = is_target & valid_mask_global
        test_df = df.loc[test_mask]
        if test_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": 0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
            }
            rows.append(row); pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
                pending = []
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid test rows (skipped)")
            continue

        # Training pool: all others or neighbors-only
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target)
        else:
            train_pool_mask = ~is_target

        train_pool = df.loc[train_pool_mask & valid_mask_global]

        # Optional controlled inclusion (sample % of target valid rows into training)
        pct = max(0.0, min(float(include_target_pct), 95.0))
        if pct > 0.0:
            n_take = int(np.ceil(len(test_df) * (pct / 100.0)))
            sample_idx = test_df.sample(n=n_take, random_state=int(include_target_seed)).index
            train_df = pd.concat([train_pool, df.loc[sample_idx]], axis=0, copy=False)
        else:
            train_df = train_pool

        if train_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": int(len(test_df)),
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
            }
            rows.append(row); pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
                pending = []
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

        # Metrics (daily)
        pred_df = pd.DataFrame({date_col: test_df[date_col].values, "y_true": y_test, "y_pred": y_hat})
        daily = _safe_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)

        # Optional aggregates (monthly/annual) for completeness
        monthly, _ = _aggregate_and_score(pred_df, date_col=date_col, y_col="y_true", yhat_col="y_pred", freq="M",  agg=agg_for_metrics)
        annual,  _ = _aggregate_and_score(pred_df, date_col=date_col, y_col="y_true", yhat_col="y_pred", freq="YE", agg=agg_for_metrics)

        sec = pd.Timestamp.utcnow().timestamp() - t0
        row = {
            "station": sid,
            "n_rows": int(len(pred_df)),
            "seconds": float(sec),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "MAE_d": float(daily["MAE"]),
            "RMSE_d": float(daily["RMSE"]),
            "R2_d": float(daily["R2"]) if daily["R2"] == daily["R2"] else np.nan,  # pass NaN
            "MAE_m": float(monthly["MAE"]),
            "RMSE_m": float(monthly["RMSE"]),
            "R2_m": float(monthly["R2"]) if monthly["R2"] == monthly["R2"] else np.nan,
            "MAE_y": float(annual["MAE"]),
            "RMSE_y": float(annual["RMSE"]),
            "R2_y": float(annual["R2"]) if annual["R2"] == annual["R2"] else np.nan,
            "used_k_neighbors": used_k,
            "include_target_pct": float(pct),
        }
        rows.append(row); pending.append(row)

        if log_csv and len(pending) >= flush_every:
            _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
            pending = []

        if show_progress:
            tqdm.write(
                f"Station {sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={daily['MAE']:.3f} RMSE_d={daily['RMSE']:.3f} R2_d={daily['R2']:.3f}"
            )

    # Flush pending progress lines
    if log_csv and pending:
        _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)

    # Final report
    report = pd.DataFrame(rows)

    # Optional save
    if save_table_path:
        ext = str(save_table_path).lower()
        if ext.endswith(".csv"):
            report.to_csv(save_table_path, index=False)
        elif ext.endswith(".parquet"):
            report.to_parquet(save_table_path, index=False, compression=parquet_compression)
        else:
            # default to CSV
            report.to_csv(save_table_path, index=False)

    # Sort by daily RMSE (ascending) for convenience
    if not report.empty and "RMSE_d" in report.columns:
        report = report.sort_values("RMSE_d", ascending=True).reset_index(drop=True)

    return report


__all__ = [
    "RFParams",
    "evaluate_all_stations_fast",
]
