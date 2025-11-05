# src/missclimatepy/evaluate.py
# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise evaluation with (a) K-neighborhood training pools and
(b) controlled inclusion (leakage) of the target station rows.

Adds optional return of per-observation predictions and includes x,y,z in metrics.

See docstring in `evaluate_all_stations_fast` for details.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import BallTree

try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

StationId = Union[str, int]


# --------------------------------------------------------------------------- #
# RF hyperparameters
# --------------------------------------------------------------------------- #
@dataclass
class RFParams:
    """RandomForestRegressor hyperparameters (sklearn-compatible defaults)."""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Union[str, int, float, None] = "sqrt"  # "auto" deprecated
    bootstrap: bool = True
    n_jobs: int = -1
    random_state: int = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _ensure_datetime_naive(s: pd.Series) -> pd.Series:
    """
    Ensure a Series is datetime64[ns] *without* timezone.

    - Coerces to datetime.
    - If tz-aware, removes the timezone (tz_localize(None)).

    This avoids the deprecated is_datetime64tz_dtype check.
    """
    s = pd.to_datetime(s, errors="coerce")
    # Robust, forward-compatible tz detection:
    if getattr(s.dtype, "tz", None) is not None:
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
    """MAE/RMSE/R2 using NumPy (no sklearn kwarg pitfalls)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if y_true.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    var = float(np.var(y_true))
    if y_true.size < 2 or var == 0.0:
        r2 = np.nan
    else:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


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
    """Aggregate to freq and score."""
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
    """Parse date, window, add time-features, assemble feature list."""
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


def _build_neighbor_map_balltree(
    df: pd.DataFrame,
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    k: int,
    include_self: bool = False,
) -> Dict[StationId, List[StationId]]:
    """
    {station -> [K nearest neighbors]} using per-station median (lat, lon)
    and BallTree(haversine). Altitude is **not** used for geodesic KNN.
    """
    cent = df.groupby(id_col)[[lat_col, lon_col]].median().reset_index()
    lat_rad = np.deg2rad(cent[lat_col].to_numpy().astype(float))
    lon_rad = np.deg2rad(cent[lon_col].to_numpy().astype(float))
    X = np.c_[lat_rad, lon_rad]
    tree = BallTree(X, metric="haversine")

    k_eff = min(int(k), len(cent)) + (1 if include_self else 0)
    dists, idxs = tree.query(X, k=k_eff)

    ids = cent[id_col].astype(object).to_numpy()
    nmap: Dict[StationId, List[StationId]] = {}
    for i, sid in enumerate(ids):
        neigh_ids = [ids[j] for j in idxs[i].tolist()]
        if not include_self:
            neigh_ids = [nid for nid in neigh_ids if nid != sid]
        nmap[sid] = neigh_ids[: min(int(k), len(neigh_ids))]
    return nmap


def _append_rows_to_csv(rows: List[Dict], path: str, *, header_written_flag: Dict[str, bool]) -> None:
    if not rows or path is None:
        return
    df_tmp = pd.DataFrame(rows)
    first = (not header_written_flag.get(path, False))
    df_tmp.to_csv(path, mode="a", index=False, header=first, encoding="utf-8")
    header_written_flag[path] = True


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def evaluate_all_stations_fast(
    data: pd.DataFrame,
    *,
    # schema
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # evaluation subset
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[StationId]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[StationId], bool]] = None,
    min_station_rows: Optional[int] = None,
    # neighborhood & inclusion
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[StationId, List[StationId]]] = None,
    include_target_pct: float = 0.0,     # 0..95
    include_target_seed: int = 42,
    # model & metrics
    rf_params: Optional[Union[RFParams, Dict]] = None,
    agg_for_metrics: str = "sum",
    # UX
    show_progress: bool = False,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    # output
    save_table_path: Optional[str] = None,
    parquet_compression: str = "snappy",
    # NEW: also return per-observation predictions
    return_predictions: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Train and evaluate one RF per target station.

    Returns
    -------
    DataFrame
        One row per station with daily/monthly/yearly metrics, plus:
        - `x`, `y`, `z`: median (lat, lon, alt) of the target station.
    Or (if return_predictions=True)
    -------
    (report_df, predictions_df)
        - report_df: as above
        - predictions_df columns:
            ['station','date','x','y','z','y_obs','y_mod']
    """
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # Preprocess
    df, feats = _preprocess_once(
        data,
        id_col=id_col, date_col=date_col,
        lat_col=lat_col, lon_col=lon_col, alt_col=alt_col,
        target_col=target_col,
        start=start, end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
    )
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # Stations to evaluate
    all_ids = df[id_col].dropna().unique().tolist()
    chosen: List[StationId] = []
    if prefix is not None:
        pfx = [prefix] if isinstance(prefix, str) else list(prefix)
        for p in pfx:
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
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # Optional min rows
    if min_station_rows is not None:
        valid_counts = df.loc[valid_mask_global, [id_col]].groupby(id_col).size().astype(int)
        before = len(stations)
        stations = [s for s in stations if int(valid_counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress and before != len(stations):
            tqdm.write(f"Filtered by min_station_rows={min_station_rows}: {before} → {len(stations)} stations")

    # Neighbor map
    if neighbor_map is not None:
        nmap = neighbor_map
        used_k: Optional[int] = None
    elif k_neighbors is not None:
        nmap = _build_neighbor_map_balltree(
            df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k=int(k_neighbors), include_self=False
        )
        used_k = int(k_neighbors)
    else:
        nmap = None
        used_k = None

    # RF params
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # Precompute per-station median coordinates for (x,y,z)
    med_coords = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]].median().rename(
            columns={lat_col: "x", lon_col: "y", alt_col: "z"}
        )
    )

    header_flag: Dict[str, bool] = {}
    pending: List[Dict] = []
    rows: List[Dict] = []
    preds_rows: List[pd.DataFrame] = []

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        is_target = (df[id_col] == sid)
        target_valid_mask = is_target & valid_mask_global
        target_df = df.loc[target_valid_mask]
        if target_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "x": np.nan, "y": np.nan, "z": np.nan,
                "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": 0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
            }
            rows.append(row)
            pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
                pending = []
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid rows (skipped)")
            continue

        # Coordinates (median) for row + predictions
        coord_row = med_coords.loc[sid] if sid in med_coords.index else pd.Series({"x": np.nan, "y": np.nan, "z": np.nan})
        x0, y0, z0 = float(coord_row["x"]), float(coord_row["y"]), float(coord_row["z"])

        # Training pool from neighbors
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target)
        else:
            train_pool_mask = (~is_target)
        train_pool = df.loc[train_pool_mask & valid_mask_global]

        # Controlled inclusion
        pct = max(0.0, min(float(include_target_pct), 95.0))
        if pct > 0.0 and not target_df.empty:
            n_take = int(np.ceil(len(target_df) * (pct / 100.0)))
            sampled = target_df.sample(n=n_take, random_state=int(include_target_seed))
            train_df = pd.concat([train_pool, sampled], axis=0, copy=False)
            test_df = target_df.drop(index=sampled.index, errors="ignore")
        else:
            train_df = train_pool
            test_df = target_df

        if train_df.empty or test_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "x": x0, "y": y0, "z": z0,
                "n_rows": int(len(test_df)),
                "seconds": float(sec),
                "rows_train": int(len(train_df)),
                "rows_test": int(len(test_df)),
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k,
                "include_target_pct": float(pct),
            }
            rows.append(row)
            pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
                pending = []
            if show_progress:
                tqdm.write(f"Station {sid}: empty train/test after split (skipped)")
            continue

        # Fit / predict
        model = RandomForestRegressor(**rf_kwargs)
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[feats].to_numpy(copy=False)
        y_test = test_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # Metrics
        pred_df = pd.DataFrame(
            {date_col: test_df[date_col].values, "y_true": y_test, "y_pred": y_hat}
        )
        daily = _safe_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
        monthly, _ = _aggregate_and_score(pred_df, date_col=date_col, y_col="y_true", yhat_col="y_pred", freq="M",  agg=agg_for_metrics)
        annual,  _ = _aggregate_and_score(pred_df, date_col=date_col, y_col="y_true", yhat_col="y_pred", freq="YE", agg=agg_for_metrics)

        sec = pd.Timestamp.utcnow().timestamp() - t0
        row = {
            "station": sid,
            "x": x0, "y": y0, "z": z0,
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
        }
        rows.append(row)
        pending.append(row)

        if return_predictions:
            # Build compact per-observation frame for this station
            tmp = pd.DataFrame({
                "station": sid,
                "date": pred_df[date_col].values,
                "x": x0, "y": y0, "z": z0,
                "y_obs": pred_df["y_true"].values,
                "y_mod": pred_df["y_pred"].values,
            })
            preds_rows.append(tmp)

        if log_csv and len(pending) >= flush_every:
            _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
            pending = []

        if show_progress:
            tqdm.write(
                f"Station {sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={daily['MAE']:.3f} RMSE_d={daily['RMSE']:.3f} R2_d={daily['R2'] if not np.isnan(daily['R2']) else np.nan}"
            )

    if log_csv and pending:
        _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)

    report = pd.DataFrame(rows)
    if not report.empty and "RMSE_d" in report.columns:
        report = report.sort_values("RMSE_d", ascending=True).reset_index(drop=True)

    if save_table_path:
        ext = str(save_table_path).lower()
        if ext.endswith(".csv"):
            report.to_csv(save_table_path, index=False)
        elif ext.endswith(".parquet"):
            report.to_parquet(save_table_path, index=False, compression=parquet_compression)
        else:
            report.to_csv(save_table_path, index=False)

    if return_predictions:
        preds = pd.concat(preds_rows, axis=0, ignore_index=True) if preds_rows else pd.DataFrame(
            columns=["station", "date", "x", "y", "z", "y_obs", "y_mod"]
        )
        preds = preds.sort_values(["station", "date"]).reset_index(drop=True)
        return report, preds

    return report


__all__ = [
    "RFParams",
    "evaluate_all_stations_fast",
]