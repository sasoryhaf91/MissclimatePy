from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .metrics import safe_metrics, aggregate_and_score
from .neighbors import neighbor_distances as _neighbor_distances

__all__ = ["RFParams", "evaluate_all_stations_fast"]

@dataclass(frozen=True)
class RFParams:
    """Small, explicit container for RF hyperparameters."""
    n_estimators: int = 200
    max_depth: Optional[int] = None
    random_state: int = 42
    n_jobs: int = -1
    max_samples: Optional[float] = None      # bootstrap sample ratio or None
    bootstrap: bool = True

def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool = False) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["doy"] = out[date_col].dt.dayofyear
    if add_cyclic:
        out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.25)
    return out

def _resolve_features(add_cyclic: bool, lat: str, lon: str, alt: str) -> List[str]:
    feats = [lat, lon, alt, "year", "month", "doy"]
    if add_cyclic:
        feats += ["doy_sin", "doy_cos"]
    return feats

def _valid_pct(p: float) -> float:
    return float(min(95.0, max(0.0, p)))

def evaluate_all_stations_fast(
    data: pd.DataFrame,
    *,
    # generic user-provided columns
    station_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # selection
    station_ids: Optional[Iterable[str]] = None,
    # neighbor logic
    k_neighbors: Optional[int] = 20,
    neighbor_table: Optional[pd.DataFrame] = None,
    # RF
    rf_params: RFParams = RFParams(),
    # time & features
    start: Optional[str] = None,
    end: Optional[str] = None,
    add_cyclic: bool = False,
    include_target_pct: float = 0.0,  # 0 => strict exclusion; otherwise (1..95] % leak
    min_station_rows: Optional[int] = None,
    # I/O
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Evaluate per-station Random Forests trained on K nearest neighbors
    (optionally including a percentage of the target station rows).

    Returns a tidy table of **daily metrics** per station:
    [station, n_rows, seconds, MAE_d, RMSE_d, R2_d, latitude, longitude, altitude, include_target_pct]
    """
    # ---- 1) validate & slice period
    df = data.copy()
    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' column.")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
        df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    required = {station_col, lat_col, lon_col, alt_col, target_col}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required column(s): {sorted(miss)}")

    # ---- 2) choose stations
    all_ids = df[station_col].dropna().astype(str).unique().tolist()
    stations = sorted(list(set(station_ids))) if station_ids else sorted(all_ids)

    # ---- 3) precompute features
    df = _add_time_features(df, date_col, add_cyclic=add_cyclic)
    feats = _resolve_features(add_cyclic, lat_col, lon_col, alt_col)
    valid_mask = ~df[feats + [target_col]].isna().any(axis=1)

    # ---- 4) neighbor map
    if k_neighbors is not None:
        if neighbor_table is None:
            base = df[[station_col, lat_col, lon_col]].drop_duplicates()
            nbd = _neighbor_distances(
                base, station_col=station_col, lat_col=lat_col, lon_col=lon_col,
                k_neighbors=max(1, int(k_neighbors)), include_self=False
            )
        else:
            # must contain [station_col, neighbor] at least
            cols = set(neighbor_table.columns)
            need = {station_col, "neighbor"}
            if not need.issubset(cols):
                raise ValueError("neighbor_table must contain [station_col, 'neighbor'] columns.")
            # keep best K per station if table contains rank or distance
            if "rank" in cols:
                nbd = (neighbor_table
                       .sort_values(["station", "rank"])
                       .groupby(station_col)
                       .head(int(k_neighbors))
                       .reset_index(drop=True))
            else:
                nbd = (neighbor_table
                       .groupby(station_col)
                       .head(int(k_neighbors))
                       .reset_index(drop=True))
        neighbor_map = {
            str(s): nbd.loc[nbd[station_col].astype(str) == str(s), "neighbor"]
            .astype(str).tolist()
            for s in stations
        }
    else:
        neighbor_map = {str(s): [str(x) for x in all_ids if str(x) != str(s)] for s in stations}

    # ---- 5) optional station filter by valid rows
    if min_station_rows is not None:
        counts = df.loc[valid_mask, [station_col]].groupby(station_col).size().astype(int)
        pre = len(stations)
        stations = [s for s in stations if int(counts.get(str(s), 0)) >= int(min_station_rows)]
        if show_progress:
            print(f"Filtered by min_station_rows={min_station_rows}: {pre} → {len(stations)} stations")

    # ---- 6) evaluation loop
    pct = _valid_pct(include_target_pct)
    results = []
    for sid in stations:
        t0 = time.time()
        is_target = (df[station_col].astype(str) == str(sid))
        test_df = df.loc[is_target & valid_mask].copy()
        if test_df.empty:
            results.append({
                "station": str(sid), "n_rows": 0, "seconds": time.time()-t0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "latitude": np.nan, "longitude": np.nan, "altitude": np.nan,
                "include_target_pct": pct / 100.0,
            })
            if show_progress:
                print(f"Station {sid}: 0 valid rows (skipped)")
            continue

        lat0 = float(test_df[lat_col].median())
        lon0 = float(test_df[lon_col].median())
        alt0 = float(test_df[alt_col].median())

        # Training pool from neighbors only (excludes target)
        neigh_ids = neighbor_map.get(str(sid), [])
        train_pool = df.loc[df[station_col].astype(str).isin(neigh_ids) & valid_mask]

        # Optional inclusion of target rows in training
        if pct > 0.0:
            n_take = int(np.ceil(len(test_df) * (pct / 100.0)))
            idx = test_df.sample(n=n_take, random_state=rf_params.random_state).index
            train_df = pd.concat([train_pool, df.loc[idx]], axis=0, copy=False)
        else:
            train_df = train_pool

        if train_df.empty:
            results.append({
                "station": str(sid), "n_rows": 0, "seconds": time.time()-t0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "latitude": lat0, "longitude": lon0, "altitude": alt0,
                "include_target_pct": pct / 100.0,
            })
            if show_progress:
                print(f"Station {sid}: empty train (skipped)")
            continue

        X_tr = train_df[feats].to_numpy(copy=False)
        y_tr = train_df[target_col].to_numpy(copy=False)
        X_te = test_df[feats].to_numpy(copy=False)
        y_te = test_df[target_col].to_numpy(copy=False)

        model = RandomForestRegressor(
            n_estimators=rf_params.n_estimators,
            max_depth=rf_params.max_depth,
            random_state=rf_params.random_state,
            n_jobs=rf_params.n_jobs,
            bootstrap=rf_params.bootstrap,
            max_samples=rf_params.max_samples,
        )
        model.fit(X_tr, y_tr)
        y_hat = model.predict(X_te)

        m_d = safe_metrics(y_te, y_hat)
        sec = time.time() - t0
        results.append({
            "station": str(sid),
            "n_rows": int(len(y_te)),
            "seconds": sec,
            "MAE_d": float(m_d["MAE"]),
            "RMSE_d": float(m_d["RMSE"]),
            "R2_d": float(m_d["R2"]),
            "latitude": lat0, "longitude": lon0, "altitude": alt0,
            "include_target_pct": pct / 100.0,
        })
        if show_progress:
            print(f"Station {sid}: {sec:.2f}s (train={len(train_df):,} | test={len(test_df):,} | incl={pct:.1f}%)")

    out = pd.DataFrame(results)
    return out
