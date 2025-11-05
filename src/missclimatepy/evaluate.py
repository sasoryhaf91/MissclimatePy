# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise evaluation with controlled inclusion of the target station and
optional K-neighborhood training pools.

This module provides:
- A small `RFParams` dataclass for RandomForest hyperparameters.
- A memory-lean evaluator: `evaluate_all_stations_fast`.

Design goals
------------
- **One model per target station**. Train either on all other stations or the
  K nearest neighbors.
- **Controlled inclusion**. Optionally leak 1..95% of the target station’s
  valid rows into training for local adaptation. With 0.0, the target is fully
  excluded (LOSO-like behavior). The **test set is always the complement**
  (100 − include_target_pct)% of the target’s valid rows → no leakage.
- **Daily metrics prioritized**. We compute daily MAE/RMSE/R²; monthly/annual
  metrics are provided for completeness.
- **Generic schema**. Column names (id/date/coords/alt/target) are parameters.
- **Preprocessing once**. Datetime parsing and time features are computed once
  and reused across stations.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as _mae
from sklearn.metrics import mean_squared_error as _mse
from sklearn.metrics import r2_score as _r2

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
    max_features: Union[str, int, float, None] = None  # None ~ sklearn default
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
    # Compatibilidad futura: evitar is_datetime64tz_dtype (deprecado)
    if getattr(s.dtype, "tz", None) is not None:  # dtype es DatetimeTZDtype
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
    mae = float(_mae(y_true, y_pred))
    # use squared=False for modern sklearn; revert to sqrt if needed
    try:
        rmse = float(_mse(y_true, y_pred, squared=False))
    except TypeError:  # older sklearn
        rmse = float(_mse(y_true, y_pred) ** 0.5)
    if y_true.size < 2 or float(np.var(y_true)) == 0.0:
        r2 = np.nan
    else:
        r2 = float(_r2(y_true, y_pred))
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
    """
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # 1) Preprocesado único
    df, feats = _preprocess_once(
        data,
        id_col=id_col, date_col=date_col,
        lat_col=lat_col, lon_col=lon_col, alt_col=alt_col,
        target_col=target_col,
        start=start, end=end,
        add_cyclic=add_cyclic,
        feature_cols=feature_cols,
    )

    # 2) Máscaras de validez
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)
    df_valid = df.loc[valid_mask_global]   # se usará para train/test

    # 3) Conjunto de estaciones a evaluar
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

    # únicos, orden estable
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # 4) Filtro por mínimo de filas válidas (contadas sobre df_valid)
    if min_station_rows is not None:
        valid_counts = df_valid[[id_col]].groupby(id_col).size().astype(int)
        before = len(stations)
        stations = [s for s in stations if int(valid_counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress and before != len(stations):
            tqdm.write(f"Filtered by min_station_rows={min_station_rows}: {before} → {len(stations)} stations")

    # 5) Neighbor map (si no nos lo pasan)
    if neighbor_map is not None:
        nmap = neighbor_map
        used_k = None
    elif k_neighbors is not None:
        # Centroides por estación calculados en df_valid (coordenadas medianas)
        centroids = (
            df_valid.groupby(id_col)[[lat_col, lon_col]]
            .median()
            .reset_index()
            .rename(columns={id_col: "station", lat_col: "latitude", lon_col: "longitude"})
        )
        tbl = neighbor_distances(
            stations=centroids,
            k_neighbors=int(k_neighbors),
            include_self=False,  # importante: excluirse a sí misma del vecindario
        )
        nmap: Dict[StationId, List[StationId]] = {}
        for st, sub in tbl.groupby("station"):
            nmap[st] = sub.sort_values("rank")["neighbor"].tolist()
        used_k = int(k_neighbors)
    else:
        nmap = None
        used_k = None

    # 6) RF params
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # 7) Iteración por estación objetivo
    header_flag: Dict[str, bool] = {}
    pending: List[Dict] = []
    rows: List[Dict] = []
    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    # porcentaje incluido acotado
    pct = max(0.0, min(float(include_target_pct), 95.0))
    rng = np.random.default_rng(int(include_target_seed))

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        # Test = TODAS las filas válidas de la estación objetivo (df_valid)
        test_df = df_valid.loc[df_valid[id_col] == sid]
        if test_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": 0,
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k, "include_target_pct": float(pct),
            }
            rows.append(row)
            pending.append(row)
            if log_csv and len(pending) >= flush_every:
                _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
                pending = []
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid test rows (skipped)")
            continue

        # Pool de vecinos: SOLO vecinas (si nmap) o todas las otras estaciones válidas
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            # Si no hay vecinos (caso borde), hacemos fallback a "todas excepto target"
            if neigh_ids:
                train_pool = df_valid.loc[df_valid[id_col].isin(neigh_ids)]
            else:
                train_pool = df_valid.loc[df_valid[id_col] != sid]
        else:
            train_pool = df_valid.loc[df_valid[id_col] != sid]

        # Inclusión controlada: % de filas de la propia estación objetivo
        if pct > 0:
            n_take = int(np.ceil(len(test_df) * (pct / 100.0)))
            # muestreo reproducible desde el índice del test_df
            sample_idx = rng.choice(test_df.index.to_numpy(), size=n_take, replace=False)
            train_df = pd.concat([train_pool, df_valid.loc[sample_idx]], axis=0, copy=False)
        else:
            train_df = train_pool

        # (opcional) quitar duplicados exactos por índice para evitar doble conteo
        train_df = train_df.loc[~train_df.index.duplicated(keep="first")]

        if train_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row = {
                "station": sid, "n_rows": 0, "seconds": sec,
                "rows_train": 0, "rows_test": int(len(test_df)),
                "MAE_d": np.nan, "RMSE_d": np.nan, "R2_d": np.nan,
                "MAE_m": np.nan, "RMSE_m": np.nan, "R2_m": np.nan,
                "MAE_y": np.nan, "RMSE_y": np.nan, "R2_y": np.nan,
                "used_k_neighbors": used_k, "include_target_pct": float(pct),
            }
            rows.append(row)
            pending.append(row)
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
        X_test  = test_df[feats].to_numpy(copy=False)
        y_test  = test_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)
        y_hat = model.predict(X_test)

        # Métricas
        pred_df = pd.DataFrame({date_col: test_df[date_col].values, "y_true": y_test, "y_pred": y_hat})
        daily = _safe_metrics(pred_df["y_true"].values, pred_df["y_pred"].values)
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

        if log_csv and len(pending) >= flush_every:
            _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)
            pending = []

        if show_progress:
            tqdm.write(
                f"Station {sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={daily['MAE']:.3f} RMSE_d={daily['RMSE']:.3f} R2_d={daily['R2']:.3f}"
            )

    # Flush y reporte final
    if log_csv and pending:
        _append_rows_to_csv(pending, log_csv, header_written_flag=header_flag)

    report = pd.DataFrame(rows)
    if save_table_path:
        ext = str(save_table_path).lower()
        if ext.endswith(".csv"):
            report.to_csv(save_table_path, index=False)
        elif ext.endswith(".parquet"):
            report.to_parquet(save_table_path, index=False, compression=parquet_compression)
        else:
            report.to_csv(save_table_path, index=False)

    if not report.empty and "RMSE_d" in report.columns:
        report = report.sort_values("RMSE_d", ascending=True).reset_index(drop=True)

    return report

__all__ = [
    "RFParams",
    "evaluate_all_stations_fast",
]

