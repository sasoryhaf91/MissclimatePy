# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise evaluation for climate-data imputation using ONLY spatial
coordinates (latitude, longitude, altitude) and calendar features
(year, month, day-of-year, optional cyclic sin/cos).

Design
------
* One RandomForestRegressor **per target station**.
* Training pool = either:
    - all other stations, or
    - K nearest neighbors by Haversine on (lat, lon).
* Optional, leakage-controlled inclusion of a fraction of the **target
  station's own valid rows** into TRAINING. Any included rows are removed
  from the TEST set to avoid information leakage.

What you get
------------
* report_df: one row per evaluated station, including sizes and metrics
  at daily / monthly / annual levels:
    ['station','n_rows','seconds','rows_train','rows_test',
     'MAE_d','RMSE_d','R2_d','MAE_m','RMSE_m','R2_m','MAE_y','RMSE_y','R2_y',
     'used_k_neighbors','include_target_pct','latitude','longitude','altitude']
* preds_df: per-row predictions for the test slice:
    ['station', <date_col>, 'latitude','longitude','altitude','y_obs','y_mod']

Key parameters
--------------
* `min_station_rows`: Minimum amount of observed data required to evaluate a
  station. By default we count **observed target rows** within [start, end].
  You can switch to "valid-features" counting if desired.
* `include_target_pct` (0..95): fraction of the target's valid rows (within
  the window) **included into TRAINING only**; those rows are removed from TEST.
  As `include_target_pct` increases, TRAIN grows and TEST shrinks (with at
  least one test row kept when possible).

Notes
-----
* Generic schema: you pass the column names.
* No external covariates required.
* Determinism via RF random_state + inclusion sampling seed.
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
# Public hyperparameters container
# --------------------------------------------------------------------------- #

@dataclass
class RFParams:
    """
    Convenience container for RandomForestRegressor hyperparameters.

    Parameters
    ----------
    n_estimators : int
    max_depth : int | None
    min_samples_split : int
    min_samples_leaf : int
    max_features : str | int | float | None
        For modern scikit-learn, preferred strings are {"sqrt","log2"} or None.
    bootstrap : bool
    n_jobs : int
    random_state : int
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
    """
    Ensure a naive (timezone-free) datetime64[ns] series.
    """
    s = pd.to_datetime(s, errors="coerce")
    # Avoid deprecated is_datetime64tz_dtype; check dtype class instead
    if isinstance(s.dtype, pd.DatetimeTZDtype):  # type: ignore[attr-defined]
        s = s.dt.tz_localize(None)
    return s


def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year.astype("int32", copy=False)
    out["month"] = out[date_col].dt.month.astype("int16", copy=False)
    out["doy"] = out[date_col].dt.dayofyear.astype("int16", copy=False)
    if add_cyclic:
        # 365.25 to keep leap-years roughly consistent
        out["doy_sin"] = np.sin(2.0 * np.pi * out["doy"].to_numpy() / 365.25)
        out["doy_cos"] = np.cos(2.0 * np.pi * out["doy"].to_numpy() / 365.25)
    return out


def _rmse_manual(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return np.nan
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def _safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE/RMSE/R² robustly; return NaN for degenerate/short cases.
    """
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
    """
    Normalize frequency short-hands to start-period codes:
    M -> MS; Y/A -> YS; Q -> QS.
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
    Aggregate to a given frequency and score on the overlap.

    Returns
    -------
    (metrics_dict, aggregated_dataframe)
    """
    freq = _freq_alias(freq=freq)
    op = _agg_op(agg)

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


def _append_rows_to_csv(rows: List[Dict], path: str, *, header_written_flag: Dict[str, bool]) -> None:
    if not rows or path is None:
        return
    tmp = pd.DataFrame(rows)
    first = (not header_written_flag.get(path, False))
    tmp.to_csv(path, mode="a", index=False, header=first)
    header_written_flag[path] = True


def _concat_index(parts: List[pd.Index]) -> pd.Index:
    """
    Concatenate a list of Index objects safely, ignoring empties.
    Avoids pandas' FutureWarning about empty entries in concatenation.
    """
    clean = [ix for ix in parts if len(ix) > 0]
    if not clean:
        return pd.Index([])
    if len(clean) == 1:
        return clean[0]
    return pd.Index(np.concatenate([ix.values for ix in clean]))


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
    # minimum data requirement
    min_station_rows: Optional[int] = None,
    min_station_rows_mode: str = "observed",   # {"observed", "valid_features"}
    # neighborhood & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,           # 0 = exclude; 1..95 = include %
    include_target_seed: int = 42,
    # model & metrics
    rf_params: Optional[Union[RFParams, Dict]] = None,
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
    Evaluate one Random-Forest model per target station using either all other
    stations or only its K nearest neighbors as the training pool. Optionally
    include 1..95% of the target station's valid rows in training. Any target
    rows included in training are REMOVED from testing to avoid leakage.

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
        Optional filters to choose which stations to *evaluate* (OR semantics).
    min_station_rows : int or None
        Minimum amount of data required to evaluate a station, counted according
        to `min_station_rows_mode`:
          - "observed": number of observed (non-NaN) target rows in [start,end]
          - "valid_features": number of rows having all features + target present
    min_station_rows_mode : {"observed","valid_features"}
        See above. Default "observed".
    k_neighbors : int or None
        If provided and `neighbor_map` is None, builds a KNN map (haversine on
        lat/lon) and trains each target model only on its neighbors (excluding
        itself), plus optional leakage.
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
        If provided, saves the final *report* (.csv or .parquet depending on extension).

    Returns
    -------
    (report_df, predictions_df)
        report_df : one row per evaluated station with metrics and metadata.
        predictions_df : per-row predictions for the held-out (test) rows.
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

    # Unique in stable order
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # 4) Optional minimum rows filter
    if min_station_rows is not None:
        mode = str(min_station_rows_mode).lower().strip()
        if mode not in {"observed", "valid_features"}:
            raise ValueError("min_station_rows_mode must be 'observed' or 'valid_features'.")

        if mode == "observed":
            # Count observed (non-NaN target) rows in the window
            obs_counts = df[~df[target_col].isna()].groupby(id_col, sort=False)[target_col].size().astype(int)
            def count_get(sid):
                return int(obs_counts.get(sid, 0))
            label = "observed"
        else:
            # Count rows that have all features + target present
            valid_counts = df.loc[valid_mask_global, [id_col]].groupby(id_col, sort=False).size().astype(int)
            def count_get(sid):
                return int(valid_counts.get(sid, 0))
            label = "valid_features"

        before = len(stations)
        stations = [s for s in stations if count_get(s) >= int(min_station_rows)]
        after = len(stations)
        if show_progress and before != after:
            tqdm.write(
                f"Filtered by min_station_rows({label})={int(min_station_rows)}: {before} → {after} stations"
            )

    # 5) Neighbor map
    used_k = None
    if neighbor_map is not None:
        nmap = neighbor_map
        used_k = None  # user-provided list lengths may vary
    elif k_neighbors is not None:
        nmap = _build_neighbor_map_haversine(
            df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k=int(k_neighbors), include_self=False
        )
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
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # 7) Iterate over stations and fit/predict
    header_flag: Dict[str, bool] = {}
    pending_rows: List[Dict] = []
    rows_report: List[Dict] = []
    all_preds: List[pd.DataFrame] = []

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    # Pre-compute station medoids to be included in the report
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median()
        .rename(columns={lat_col: "latitude", lon_col: "longitude", alt_col: "altitude"})
    )

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        # --- Target station valid slice -------------------------------------
        is_target = (df[id_col] == sid)
        target_valid = df.loc[is_target & valid_mask_global]

        if target_valid.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "n_rows": 0, "seconds": float(sec),
                "rows_train": 0, "rows_test": 0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
            }
            rows_report.append(row)
            pending_rows.append(row)
            if log_csv and len(pending_rows) >= flush_every:
                _append_rows_to_csv(pending_rows, log_csv, header_written_flag=header_flag)
                pending_rows = []
            if show_progress:
                tqdm.write(f"{sid}: 0 valid rows for test (skipped)")
            continue

        # --- Train pool (neighbors or all-others; EXCLUDE target) -----------
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target)
        else:
            train_pool_mask = ~is_target
        train_pool = df.loc[train_pool_mask & valid_mask_global]

        # --- Leakage-controlled inclusion -----------------------------------
        pct = max(0.0, min(float(include_target_pct), 95.0))
        n_all_valid = len(target_valid)
        if pct > 0.0 and n_all_valid > 0:
            n_take = int(np.floor(n_all_valid * (pct / 100.0)))
            # Keep at least one row for test if possible
            if n_all_valid > 1:
                n_take = min(n_take, n_all_valid - 1)
        else:
            n_take = 0

        if n_take > 0:
            inc_idx = target_valid.sample(n=n_take, random_state=int(include_target_seed)).index
            test_df = target_valid.drop(index=inc_idx)
            inc_target_df = target_valid.loc[inc_idx]
        else:
            test_df = target_valid
            inc_target_df = target_valid.iloc[0:0]

        # Final train = pool + included-target slice
        train_df = pd.concat([train_pool, inc_target_df], axis=0, copy=False)

        if train_df.empty or test_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "n_rows": int(len(test_df)), "seconds": float(sec),
                "rows_train": int(len(train_df)), "rows_test": int(len(test_df)),
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(pct),
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
            }
            rows_report.append(row)
            pending_rows.append(row)
            if log_csv and len(pending_rows) >= flush_every:
                _append_rows_to_csv(pending_rows, log_csv, header_written_flag=header_flag)
                pending_rows = []
            if show_progress:
                tqdm.write(f"{sid}: empty train or test (skipped)")
            continue

        # --- Fit & predict ---------------------------------------------------
        model = RandomForestRegressor(**rf_kwargs)
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[feats].to_numpy(copy=False)
        y_test = test_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # Per-row predictions with canonical names
        pred_df = pd.DataFrame({
            "station": sid,
            date_col: test_df[date_col].values,
            "latitude": test_df[lat_col].astype(float).values,
            "longitude": test_df[lon_col].astype(float).values,
            "altitude": test_df[alt_col].astype(float).values,
            "y_obs": y_test,
            "y_mod": y_hat,
        })

        # Metrics (daily)
        daily = _safe_metrics(pred_df["y_obs"].values, pred_df["y_mod"].values)

        # Monthly / Annual aggregates
        monthly, _ = _aggregate_and_score(
            pred_df, date_col=date_col, y_col="y_obs", yhat_col="y_mod", freq="M", agg=agg_for_metrics
        )
        annual, _ = _aggregate_and_score(
            pred_df, date_col=date_col, y_col="y_obs", yhat_col="y_mod", freq="YS", agg=agg_for_metrics
        )

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
            "include_target_pct": float(pct),
            "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan,
            "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan,
            "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan,
        }
        rows_report.append(row)
        pending_rows.append(row)
        all_preds.append(pred_df)

        if log_csv and len(pending_rows) >= flush_every:
            _append_rows_to_csv(pending_rows, log_csv, header_written_flag=header_flag)
            pending_rows = []

        if show_progress:
            tqdm.write(
                f"{sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={daily['MAE']:.3f} RMSE_d={daily['RMSE']:.3f} R2_d={daily['R2']:.3f}"
            )

    # Flush pending progress lines
    if log_csv and pending_rows:
        _append_rows_to_csv(pending_rows, log_csv, header_written_flag=header_flag)

    # Final report & predictions
    report = pd.DataFrame(rows_report)
    preds = pd.concat(all_preds, axis=0, ignore_index=True) if all_preds else pd.DataFrame(
        columns=[id_col, date_col, "latitude", "longitude", "altitude", "y_obs", "y_mod"]
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
    "RFParams",
    "evaluate_stations",
]
