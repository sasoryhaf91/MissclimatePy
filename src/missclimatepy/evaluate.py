# SPDX-License-Identifier: MIT
"""
Station-wise evaluation for climate-data imputation using XYZT-style models.

This module evaluates one regression model per station using only:
- spatial coordinates (latitude, longitude, altitude), and
- calendar features derived from the date (year, month, day-of-year,
  optional cyclic sin/cos).

Key ideas
---------
* Generic schema:
  callers provide column names, no enforced renaming.
* Local models:
  one model per station, trained on neighbors or all other stations.
* Controlled leakage:
  an optional fraction of the target station's own valid rows can be
  included in training via a precipitation-friendly month × dry/wet
  stratified sampler.
* Metrics:
  daily MAE/RMSE/R² plus monthly and yearly aggregates.
* Outputs:
  (1) a station-level report, and
  (2) a per-row prediction table with coordinates and observed/predicted
      values, ready for plotting or further analysis.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

try:  # pragma: no cover - tqdm is optional
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from .features import validate_required_columns, preprocess_for_model
from .neighbors import build_neighbor_map
from .models import make_model


# ---------------------------------------------------------------------------
# Internal metric helpers
# ---------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-square error with safe handling of empty inputs."""
    if y_true.size == 0:
        return float("nan")
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff * diff)))


def _safe_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE and R² with guards for edge cases.

    - If the series is empty, all metrics are NaN.
    - If var(y_true) == 0 or len < 2, R² is set to NaN.
    """
    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = _rmse(y_true, y_pred)

    if y_true.size < 2 or float(np.var(y_true)) == 0.0:
        r2 = float("nan")
    else:
        r2 = float(r2_score(y_true, y_pred))

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


_FREQ_ALIAS: Dict[str, str] = {"M": "MS", "A": "YS", "Y": "YS", "Q": "QS"}


def _normalize_freq(freq: str) -> str:
    """
    Normalize short frequency aliases to explicit pandas offsets.

    Examples
    --------
    "M" -> "MS"   (month start)
    "Y" -> "YS"   (year start)
    "A" -> "YS"
    "Q" -> "QS"
    """
    return _FREQ_ALIAS.get(freq, freq)


def _check_agg(agg: str) -> str:
    if agg not in {"sum", "mean", "median"}:
        raise ValueError("agg_for_metrics must be one of {'sum', 'mean', 'median'}.")
    return agg


def _aggregate_and_score(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_col: str,
    yhat_col: str,
    freq: str,
    agg: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Aggregate observed/predicted series to a given temporal frequency
    and compute MAE/RMSE/R² on the aggregated values.
    """
    agg = _check_agg(agg)
    freq = _normalize_freq(freq)

    if df_pred.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}, df_pred.copy()

    df = df_pred[[date_col, y_col, yhat_col]].dropna(subset=[date_col]).copy()
    if df.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}, df

    df = df.set_index(date_col).sort_index()
    agg_df = df.resample(freq).agg({y_col: agg, yhat_col: agg}).dropna()
    if agg_df.empty:
        return {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")}, agg_df

    metrics = _safe_regression_metrics(
        agg_df[y_col].to_numpy(),
        agg_df[yhat_col].to_numpy(),
    )
    return metrics, agg_df


# ---------------------------------------------------------------------------
# Station-selection and logging helpers
# ---------------------------------------------------------------------------


def _select_stations(
    df: pd.DataFrame,
    *,
    id_col: str,
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
) -> List[Union[str, int]]:
    """
    Select station ids using OR semantics across multiple filters.

    The base universe is all non-null station ids present in ``df``.
    """
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

        pattern = re.compile(regex)
        chosen.extend([s for s in all_ids if pattern.match(str(s))])

    if custom_filter is not None:
        chosen.extend([s for s in all_ids if custom_filter(s)])

    if not chosen:
        chosen = all_ids

    # Deduplicate while preserving order
    seen = set()
    result: List[Union[str, int]] = []
    for sid in chosen:
        if sid not in seen:
            seen.add(sid)
            result.append(sid)
    return result


def _append_rows_to_csv(
    rows: List[Dict[str, Any]],
    path: str,
    header_written: Dict[str, bool],
) -> None:
    """
    Append a batch of dict rows to CSV on disk, writing the header only once.
    """
    if not path or not rows:
        return
    tmp = pd.DataFrame(rows)
    first_time = not header_written.get(path, False)
    tmp.to_csv(path, mode="a", index=False, header=first_time)
    header_written[path] = True
    rows.clear()


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------


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
    # temporal window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # feature config
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection (OR semantics)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    min_station_rows: Optional[int] = None,
    # neighborhood & target leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,  # 0 = exclude; 1..95 = include %
    include_target_seed: int = 42,
    # model & metrics
    model_kind: str = "rf",
    model_params: Optional[Mapping[str, Any]] = None,
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
    Evaluate one XYZT-style model per target station.

    For each station:

    1. Build a training pool consisting of:
       - either all other stations, or
       - only its K nearest neighbors (if ``k_neighbors`` / ``neighbor_map`` are set), and
       - an optional (0–95%) leakage fraction of the station's own valid rows.

    2. Split the target station's valid rows into:
       - train-leakage (stratified by month × dry/wet when ``include_target_pct > 0``), and
       - a held-out test set.

    3. Fit the model on the training pool + leakage and compute metrics on the
       held-out test set at daily, monthly, and yearly scales.
    """
    # 1) Basic validation
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
    )

    # 2) Preprocess once for all stations (datetime, window, calendar features)
    df, features = preprocess_for_model(
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

    if df.empty:
        # Return empty scaffolds with expected columns
        report_cols = [
            "station",
            "n_rows",
            "seconds",
            "rows_train",
            "rows_test",
            "MAE_d",
            "RMSE_d",
            "R2_d",
            "MAE_m",
            "RMSE_m",
            "R2_m",
            "MAE_y",
            "RMSE_y",
            "R2_y",
            "used_k_neighbors",
            "include_target_pct",
            "latitude",
            "longitude",
            "altitude",
        ]
        preds_cols = [
            "station",
            date_col,
            "latitude",
            "longitude",
            "altitude",
            "y_obs",
            "y_mod",
        ]
        return (
            pd.DataFrame(columns=report_cols),
            pd.DataFrame(columns=preds_cols),
        )

    # Valid rows for features + target
    valid_mask = ~df[features + [target_col]].isna().any(axis=1)

    # 3) Station selection (OR semantics)
    stations = _select_stations(
        df,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )

    # 4) Optional minimum observed rows filter
    if min_station_rows is not None and min_station_rows > 0:
        obs_counts = (
            df.loc[~df[target_col].isna(), [id_col, target_col]]
            .groupby(id_col)[target_col]
            .size()
            .astype(int)
        )
        before = len(stations)
        stations = [s for s in stations if int(obs_counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress and before != len(stations):
            tqdm.write(
                f"Filtered by min_station_rows(observed)={int(min_station_rows)}: "
                f"{before} → {len(stations)} stations"
            )

    if not stations:
        report_cols = [
            "station",
            "n_rows",
            "seconds",
            "rows_train",
            "rows_test",
            "MAE_d",
            "RMSE_d",
            "R2_d",
            "MAE_m",
            "RMSE_m",
            "R2_m",
            "MAE_y",
            "RMSE_y",
            "R2_y",
            "used_k_neighbors",
            "include_target_pct",
            "latitude",
            "longitude",
            "altitude",
        ]
        preds_cols = [
            "station",
            date_col,
            "latitude",
            "longitude",
            "altitude",
            "y_obs",
            "y_mod",
        ]
        return (
            pd.DataFrame(columns=report_cols),
            pd.DataFrame(columns=preds_cols),
        )

    # 5) Neighbor map / K used
    if neighbor_map is not None:
        nmap = neighbor_map
        used_k: Optional[int] = None  # user-provided lists may vary in length
    elif k_neighbors is not None:
        nmap = build_neighbor_map(
            data=df,
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

    # 6) Station medoids for reporting
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median(numeric_only=True)
        .rename(columns={lat_col: "latitude", lon_col: "longitude", alt_col: "altitude"})
    )

    # 7) Evaluation loop
    header_written: Dict[str, bool] = {}
    pending_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []
    all_pred_blocks: List[pd.DataFrame] = []

    iterator = tqdm(stations, desc="Evaluating stations", unit="st") if show_progress else stations

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        is_target = df[id_col] == sid
        target_valid = df.loc[is_target & valid_mask].copy()

        if target_valid.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row_meta = {
                "station": sid,
                "n_rows": 0,
                "seconds": float(sec),
                "rows_train": 0,
                "rows_test": 0,
                "MAE_d": float("nan"),
                "RMSE_d": float("nan"),
                "R2_d": float("nan"),
                "MAE_m": float("nan"),
                "RMSE_m": float("nan"),
                "R2_m": float("nan"),
                "MAE_y": float("nan"),
                "RMSE_y": float("nan"),
                "R2_y": float("nan"),
                "used_k_neighbors": used_k,
                "include_target_pct": float(include_target_pct),
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else float("nan"),
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else float("nan"),
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else float("nan"),
            }
            report_rows.append(row_meta)
            pending_rows.append(row_meta)
            if log_csv and len(pending_rows) >= flush_every:
                _append_rows_to_csv(pending_rows, log_csv, header_written)
            if show_progress:
                tqdm.write(f"Station {sid}: 0 valid rows → skipped.")
            continue

        # Training pool: neighbors or all other stations
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            pool_mask = df[id_col].isin(neigh_ids) & (~is_target) & valid_mask
        else:
            pool_mask = (~is_target) & valid_mask

        train_pool = df.loc[pool_mask]

        # Stratified inclusion from target station (month × dry/wet)
        pct_raw = float(include_target_pct)
        pct = max(0.0, min(pct_raw, 95.0))  # cap at 95% to keep some test rows
        n_total = len(target_valid)

        if pct <= 0.0 or n_total <= 1:
            # Pure LOSO-like: no leakage
            inc_target_df = target_valid.iloc[0:0]  # empty with same schema
            test_df = target_valid
        else:
            n_take = int(np.ceil(n_total * (pct / 100.0)))
            # Ensure at least one test row remains
            n_take = min(n_take, n_total - 1)

            tv = target_valid.copy()
            tv["_month_"] = tv["month"].to_numpy()
            tv["_wet_"] = (tv[target_col].to_numpy() > 0.0)

            strata: Dict[Tuple[int, bool], np.ndarray] = {}
            for key, sub in tv.groupby(["_month_", "_wet_"]):
                strata[key] = sub.index.to_numpy()

            rng = np.random.RandomState(int(include_target_seed))
            inc_indices: List[int] = []

            for _, idx_arr in strata.items():
                if len(inc_indices) >= n_take:
                    break
                n_stratum = idx_arr.size
                n_stratum_take = int(round(n_take * (n_stratum / n_total)))
                n_stratum_take = max(0, min(n_stratum_take, n_stratum))
                if n_stratum_take > 0:
                    chosen = rng.choice(idx_arr, size=n_stratum_take, replace=False)
                    inc_indices.extend(chosen.tolist())

            if len(inc_indices) > n_take:
                inc_indices = rng.choice(np.array(inc_indices), size=n_take, replace=False).tolist()
            elif len(inc_indices) < n_take:
                remaining = n_take - len(inc_indices)
                all_idx = tv.index.to_numpy()
                mask_chosen = np.isin(all_idx, np.array(inc_indices))
                pool_left = all_idx[~mask_chosen]
                if pool_left.size > 0:
                    extra = rng.choice(pool_left, size=min(remaining, pool_left.size), replace=False)
                    inc_indices.extend(extra.tolist())

            inc_index = pd.Index(sorted(set(inc_indices)))
            inc_target_df = tv.loc[inc_index].drop(columns=["_month_", "_wet_"])
            test_df = tv.drop(index=inc_index).drop(columns=["_month_", "_wet_"])

            if test_df.empty:
                # Move one row back to test to avoid degenerate evaluation
                move_idx = inc_index[-1]
                move_row = tv.loc[[move_idx]].drop(columns=["_month_", "_wet_"])
                inc_target_df = inc_target_df.drop(index=move_idx)
                test_df = pd.concat([test_df, move_row], axis=0)

        # Final training set: neighbors + leakage slice
        train_df = pd.concat([train_pool, inc_target_df], axis=0, ignore_index=False)

        if train_df.empty:
            sec = pd.Timestamp.utcnow().timestamp() - t0
            row_meta = {
                "station": sid,
                "n_rows": 0,
                "seconds": float(sec),
                "rows_train": 0,
                "rows_test": int(len(test_df)),
                "MAE_d": float("nan"),
                "RMSE_d": float("nan"),
                "R2_d": float("nan"),
                "MAE_m": float("nan"),
                "RMSE_m": float("nan"),
                "R2_m": float("nan"),
                "MAE_y": float("nan"),
                "RMSE_y": float("nan"),
                "R2_y": float("nan"),
                "used_k_neighbors": used_k,
                "include_target_pct": float(pct),
                "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else float("nan"),
                "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else float("nan"),
                "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else float("nan"),
            }
            report_rows.append(row_meta)
            pending_rows.append(row_meta)
            if log_csv and len(pending_rows) >= flush_every:
                _append_rows_to_csv(pending_rows, log_csv, header_written)
            if show_progress:
                tqdm.write(f"Station {sid}: empty train pool → skipped.")
            continue

        # Fit model & predict
        model = make_model(model_kind=model_kind, model_params=model_params)
        X_train = train_df[features].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)
        X_test = test_df[features].to_numpy(copy=False)
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

        # Daily metrics
        daily = _safe_regression_metrics(
            pred_df["y_obs"].to_numpy(),
            pred_df["y_mod"].to_numpy(),
        )

        # Monthly / Annual aggregates
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

        sec = pd.Timestamp.utcnow().timestamp() - t0
        row_meta = {
            "station": sid,
            "n_rows": int(len(pred_df)),
            "seconds": float(sec),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "MAE_d": float(daily["MAE"]),
            "RMSE_d": float(daily["RMSE"]),
            "R2_d": float(daily["R2"]) if daily["R2"] == daily["R2"] else float("nan"),
            "MAE_m": float(monthly["MAE"]),
            "RMSE_m": float(monthly["RMSE"]),
            "R2_m": float(monthly["R2"]) if monthly["R2"] == monthly["R2"] else float("nan"),
            "MAE_y": float(annual["MAE"]),
            "RMSE_y": float(annual["RMSE"]),
            "R2_y": float(annual["R2"]) if annual["R2"] == annual["R2"] else float("nan"),
            "used_k_neighbors": used_k,
            "include_target_pct": float(pct),
            "latitude": float(medoids.loc[sid, "latitude"]) if sid in medoids.index else float("nan"),
            "longitude": float(medoids.loc[sid, "longitude"]) if sid in medoids.index else float("nan"),
            "altitude": float(medoids.loc[sid, "altitude"]) if sid in medoids.index else float("nan"),
        }
        report_rows.append(row_meta)
        pending_rows.append(row_meta)
        all_pred_blocks.append(pred_df)

        if log_csv and len(pending_rows) >= flush_every:
            _append_rows_to_csv(pending_rows, log_csv, header_written)

        if show_progress:
            tqdm.write(
                f"{sid}: {sec:.2f}s  "
                f"(train={len(train_df):,}  test={len(test_df):,}  incl={pct:.1f}%)  "
                f"MAE_d={daily['MAE']:.3f} RMSE_d={daily['RMSE']:.3f} R2_d={daily['R2']:.3f}"
            )

    # Flush pending progress lines
    if log_csv and pending_rows:
        _append_rows_to_csv(pending_rows, log_csv, header_written)

    # Final report & predictions
    report = pd.DataFrame(report_rows)
    if all_pred_blocks:
        preds = pd.concat(all_pred_blocks, axis=0, ignore_index=True)
    else:
        preds = pd.DataFrame(
            columns=["station", date_col, "latitude", "longitude", "altitude", "y_obs", "y_mod"]
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


__all__ = ["evaluate_stations", "_select_stations"]
