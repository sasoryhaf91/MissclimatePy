# SPDX-License-Identifier: MIT
"""
Memory-lean fast evaluation (daily metrics only).

evaluate_all_stations_fast():
- Trains one RandomForest per target station (LOSO by default).
- Optional neighbor restriction (k_neighbors) via build_neighbor_map.
- Optional controlled leakage with include_target_pct (0, or 1..95).
- Computes DAILY metrics (MAE_d, RMSE_d, R2_d) only.
- Streams results to CSV (log_csv) to minimize RAM if desired.
"""

from __future__ import annotations
from typing import Iterable, Optional, Dict, List
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .neighbors import build_neighbor_map


# ---------- small helpers ----------

def _safe_daily_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    var = float(np.var(y_true))
    r2 = r2_score(y_true, y_pred) if (y_true.size >= 2 and var > 0.0) else np.nan
    return {"MAE_d": mae, "RMSE_d": rmse, "R2_d": r2}


def _resolve_cols(df: pd.DataFrame, names: Iterable[str]) -> List[str]:
    low = {c.lower(): c for c in df.columns}
    out = []
    for n in names:
        k = str(n).lower()
        if k in low:
            out.append(low[k])
    return out


def _append_rows_to_csv(rows: List[Dict], path: Optional[str], header_written: Dict[str, bool]):
    if not rows or path is None:
        return
    tmp = pd.DataFrame(rows)
    write_header = not header_written.get(path, False)
    tmp.to_csv(path, mode="a", index=False, header=write_header)
    header_written[path] = True


# ---------- public API ----------

def evaluate_all_stations_fast(
    data: pd.DataFrame,
    *,
    id_col: str = "station",
    date_col: str = "date",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    alt_col: str = "altitude",
    target_col: str = "tmin",
    station_ids: Optional[Iterable[int]] = None,
    prefix: Optional[Iterable[str]] = None,
    regex: Optional[str] = None,
    start: str = "1991-01-01",
    end: str = "2020-12-31",
    rf_params: Dict = dict(n_estimators=15, max_depth=30, random_state=42, n_jobs=-1),
    n_jobs: int = -1,
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[int, List[int]]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    min_station_rows: Optional[int] = None,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Fast LOSO-style evaluation with optional neighbor restriction and
    optional target leakage. Returns a DataFrame with daily metrics.

    Parameters
    ----------
    include_target_pct : float
        0.0 => strict LOSO; values in [1, 95] allowed (we internally clamp
        0<pct<1 to 1, and pct>95 to 95).
    """
    t0_all = time.time()

    cols_needed = _resolve_cols(
        data, [id_col, date_col, lat_col, lon_col, alt_col, target_col]
    )
    if len(cols_needed) < 6:
        missing = set([id_col, date_col, lat_col, lon_col, alt_col, target_col]) - set(cols_needed)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = data[cols_needed].copy()

    # dtypes (lean memory)
    df[id_col] = df[id_col].astype("int32", copy=False)
    df[lat_col] = df[lat_col].astype("float32", copy=False)
    df[lon_col] = df[lon_col].astype("float32", copy=False)
    df[alt_col] = df[alt_col].astype("float32", copy=False)
    df[target_col] = df[target_col].astype("float32", copy=False)

    dt = pd.to_datetime(df[date_col], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(dt):
        dt = dt.dt.tz_localize(None)
    df[date_col] = dt
    df = df.dropna(subset=[date_col])

    lo = pd.to_datetime(start)
    hi = pd.to_datetime(end)
    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # simple time features
    df["doy"] = df[date_col].dt.dayofyear.astype("int16")
    df["month"] = df[date_col].dt.month.astype("int8")
    df["year"] = df[date_col].dt.year.astype("int16")

    feat_cols = [lat_col, lon_col, alt_col, "year", "month", "doy"]
    valid_mask = ~df[feat_cols + [target_col]].isna().any(axis=1)

    all_ids = df[id_col].unique().tolist()
    if station_ids:
        eval_ids = [int(s) for s in station_ids]
    else:
        eval_ids = all_ids.copy()
        if prefix:
            pref = prefix if isinstance(prefix, (list, tuple)) else [prefix]
            eval_ids = [s for s in eval_ids if any(str(s).startswith(p) for p in pref)]
        if regex:
            import re as _re
            pat = _re.compile(regex)
            eval_ids = [s for s in eval_ids if pat.match(str(s))]
    eval_ids = sorted({int(s) for s in eval_ids})

    if min_station_rows is not None:
        counts = df.loc[valid_mask, [id_col]].groupby(id_col).size().astype(int)
        before = len(eval_ids)
        eval_ids = [s for s in eval_ids if int(counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress:
            print(f"[filter] min_station_rows={min_station_rows}: {before} → {len(eval_ids)} stations")

    if k_neighbors is not None and neighbor_map is None:
        neighbor_map = build_neighbor_map(
            df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k_neighbors=k_neighbors
        )

    header_flag: Dict[str, bool] = {}
    rows: List[Dict] = []
    pending: List[Dict] = []
    it = eval_ids if not show_progress else __import__("tqdm").auto.tqdm(eval_ids, desc="Stations", unit="st")

    rf_default = dict(
        n_estimators=rf_params.get("n_estimators", 15),
        max_depth=rf_params.get("max_depth", 30),
        random_state=rf_params.get("random_state", 42),
        n_jobs=rf_params.get("n_jobs", -1),
        bootstrap=rf_params.get("bootstrap", True),
        max_samples=rf_params.get("max_samples", None),
    )

    pct = float(include_target_pct or 0.0)
    if pct < 0:
        pct = 0.0
    if 0 < pct < 1:
        pct = 1.0
    if pct > 95:
        pct = 95.0

    for sid in it:
        t0 = time.time()
        is_target = (df[id_col] == sid)
        test_mask = is_target & valid_mask
        test_df = df.loc[test_mask, [date_col] + feat_cols + [target_col]]
        if test_df.empty:
            sec = time.time() - t0
            row = dict(station=int(sid), n_rows=0, seconds=sec,
                       MAE_d=np.nan, RMSE_d=np.nan, R2_d=np.nan,
                       include_target_pct=pct)
            rows.append(row)
            pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_flag)
                pending = []
            continue

        if k_neighbors is None:
            train_mask = (~is_target) & valid_mask
        else:
            neigh_ids = (neighbor_map or {}).get(int(sid), [])
            train_mask = (df[id_col].isin(neigh_ids) & valid_mask) if neigh_ids else ((~is_target) & valid_mask)

        train_df = df.loc[train_mask, feat_cols + [target_col]]

        if pct >= 1.0:
            n_take = int(np.ceil(len(test_df) * (pct / 100.0)))
            leak_idx = test_df.sample(n=n_take, random_state=include_target_seed).index
            leak_df = df.loc[leak_idx, feat_cols + [target_col]]
            train_df = pd.concat([train_df, leak_df], axis=0, copy=False)

        if train_df.empty:
            sec = time.time() - t0
            row = dict(station=int(sid), n_rows=0, seconds=sec,
                       MAE_d=np.nan, RMSE_d=np.nan, R2_d=np.nan,
                       include_target_pct=pct)
            rows.append(row)
            pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_flag)
                pending = []
            continue

        X_train = train_df[feat_cols].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test  = test_df[feat_cols].to_numpy(copy=False)
        y_test  = test_df[target_col].to_numpy(copy=False)

        model = RandomForestRegressor(**rf_default)
        model.fit(X_train, y_train)
        y_hat = model.predict(X_test).astype("float32", copy=False)

        m = _safe_daily_metrics(y_test, y_hat)
        sec = time.time() - t0

        st_coords = df.loc[df[id_col] == sid, [lat_col, lon_col, alt_col]].median()
        row = dict(
            station=int(sid), n_rows=int(len(y_test)), seconds=sec,
            MAE_d=float(m["MAE_d"]), RMSE_d=float(m["RMSE_d"]), R2_d=float(m["R2_d"]),
            include_target_pct=pct,
            **{lat_col: float(st_coords[lat_col]),
               lon_col: float(st_coords[lon_col]),
               alt_col: float(st_coords[alt_col])},
        )
        rows.append(row)
        pending.append(row)
        if log_csv and len(pending) >= flush_every:
            _append_rows_to_csv(pending, log_csv, header_flag)
            pending = []

    if log_csv and pending:
        _append_rows_to_csv(pending, log_csv, header_flag)

    out = pd.DataFrame(rows).sort_values("station").reset_index(drop=True)
    if show_progress:
        total_sec = time.time() - t0_all
        print(f"Done. {len(eval_ids)} stations in {total_sec:.1f}s "
              f"(avg {total_sec / max(1, len(eval_ids)):.2f}s/station).")
    return out
