# SPDX-License-Identifier: MIT
"""
missclimatepy.evaluate
======================

Station-wise LOSO-like evaluation for MissClimatePy.

This module provides the main low-level evaluation engine:

- :func:`evaluate_stations`  → one row per station with multi-scale metrics.

Design goals
------------

* **Pure space–time predictors**:
  All models use only coordinates and calendar features:
  (latitude, longitude, altitude, year, month, day-of-year, optional harmonics).

* **Station-wise evaluation**:
  For each target station, we build an evaluation set consisting of all rows
  with valid features + target, and a training pool from neighbors or from all
  other stations.

* **Flexible model backend** (shared with the imputer):

  - ``"rf"``     : RandomForestRegressor
  - ``"knn"``    : KNeighborsRegressor
  - ``"linear"`` : LinearRegression
  - ``"mlp"``    : MLPRegressor (ANN)
  - ``"svd"``    : TruncatedSVD + LinearRegression
  - ``"mcm"``    : Mean Climatology Model (temporal baseline, no ML)

* **MDR / target leakage experiments**:
  A controlled fraction of valid rows from the target station can be included
  in the training pool via ``include_target_pct`` (0–95%).

* **Multi-scale metrics with KGE + MCM baseline**:
  We compute daily, monthly and annual metrics:
  MAE, RMSE, R² and KGE, both for:

  - The chosen backend (RF / KNN / linear / MLP / SVD / MCM).
  - A deterministic Mean Climatology Model (MCM) baseline.

The high-level wrapper :func:`missclimatepy.api.evaluate` is a thin pass-through
around :func:`evaluate_stations`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Hashable,
)

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from .features import (
    ensure_datetime_naive,
    add_calendar_features,
    validate_required_columns,
    default_feature_names,
)
from .neighbors import build_neighbor_map
from .metrics import multiscale_metrics, compute_mcm_baseline

try:  # optional progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # transparent iterator


# ---------------------------------------------------------------------------
# RF hyperparameters
# ---------------------------------------------------------------------------


@dataclass
class RFParams:
    """
    Hyperparameters for RandomForestRegressor used as the default backend.

    Notes
    -----
    * ``max_features="auto"`` is deprecated / invalid in recent scikit-learn
      versions; we use ``"sqrt"`` as a safe default.
    """

    n_estimators: int = 300
    max_depth: Optional[int] = None
    min_samples_leaf: int = 1
    n_jobs: int = -1
    random_state: int = 42
    max_features: Optional[Union[int, float, str]] = "sqrt"


def _normalize_rf_params(
    rf_params: Optional[Union[RFParams, Mapping[str, Any]]]
) -> RFParams:
    """
    Normalize RF hyperparameters into an RFParams instance.

    Parameters
    ----------
    rf_params :
        - None        → default RFParams()
        - RFParams    → returned as-is
        - dict/mapping→ merged into default RFParams

    Returns
    -------
    RFParams
    """
    if rf_params is None:
        return RFParams()
    if isinstance(rf_params, RFParams):
        return rf_params
    base = asdict(RFParams())
    base.update(dict(rf_params))
    return RFParams(**base)


# ---------------------------------------------------------------------------
# Station selection helper (shared with impute)
# ---------------------------------------------------------------------------


def _select_stations(
    data: pd.DataFrame,
    *,
    id_col: str,
    station_ids: Optional[Iterable[Union[int, str]]] = None,
    prefix: Optional[Iterable[str] | str] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[int, str]], bool]] = None,
) -> List[Union[int, str]]:
    """
    Select station identifiers using OR semantics across filters.

    If no filter is provided, all stations present in ``data[id_col]`` are used.

    Parameters
    ----------
    data : DataFrame
        Input table containing at least the station id column.
    id_col : str
        Name of the station id column.
    station_ids : iterable, optional
        Explicit list of station ids to include.
    prefix : iterable of str or str, optional
        One or more prefixes; matches ids for which ``str(id).startswith(prefix)``.
    regex : str, optional
        Regular expression applied to the string representation of station ids.
    custom_filter : callable, optional
        Callable ``f(id) -> bool`` used to include additional stations.

    Returns
    -------
    list
        Unique station ids in a stable order.
    """
    all_ids = data[id_col].dropna().unique().tolist()
    chosen: List[Union[int, str]] = []

    if station_ids is not None:
        chosen.extend(list(station_ids))

    if prefix is not None:
        if isinstance(prefix, str):
            prefix = [prefix]
        for p in prefix:
            chosen.extend([sid for sid in all_ids if str(sid).startswith(str(p))])

    if regex is not None:
        import re

        pat = re.compile(regex)
        chosen.extend([sid for sid in all_ids if pat.match(str(sid))])

    if custom_filter is not None:
        chosen.extend([sid for sid in all_ids if custom_filter(sid)])

    # If nothing was selected, default to "all stations"
    if not chosen:
        chosen = all_ids

    # Unique, preserving first occurrence
    seen = set()
    return [sid for sid in chosen if not (sid in seen or seen.add(sid))]


# ---------------------------------------------------------------------------
# Model factory (shared semantics with impute)
# ---------------------------------------------------------------------------


def _make_model(
    *,
    model_kind: str,
    rf_params: RFParams,
    model_params: Optional[Mapping[str, Any]],
    n_features: int,
) -> RegressorMixin:
    """
    Factory for all supported backends.

    Parameters
    ----------
    model_kind : {"rf", "knn", "linear", "mlp", "svd"}
        Backend to instantiate. "mcm" is handled separately and does not
        require a scikit-learn estimator.
    rf_params : RFParams
        Hyperparameters for Random Forest when ``model_kind="rf"``.
    model_params : mapping or None
        Extra parameters specific to the backend.
    n_features : int
        Number of feature columns (used, e.g., to bound ``max_features``).

    Returns
    -------
    RegressorMixin
        Unfitted scikit-learn compatible estimator.
    """
    kind = model_kind.lower()
    extra = dict(model_params or {})

    if kind == "rf":
        params = asdict(rf_params)
        params.update(extra)

        # Compatibility with recent sklearn: "auto" no longer valid
        mf = params.get("max_features", None)
        if mf == "auto":
            params["max_features"] = "sqrt"

        # If max_features is int, clip to n_features
        if isinstance(params.get("max_features"), int):
            if params["max_features"] > n_features:
                params["max_features"] = n_features

        return RandomForestRegressor(**params)

    if kind == "knn":
        defaults = {
            "n_neighbors": 5,
            "weights": "distance",
            "n_jobs": -1,
        }
        defaults.update(extra)
        return KNeighborsRegressor(**defaults)

    if kind == "linear":
        defaults = {"n_jobs": None}
        defaults.update(extra)
        return LinearRegression(**defaults)

    if kind == "mlp":
        defaults = {
            "hidden_layer_sizes": (64, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "random_state": rf_params.random_state,
        }
        defaults.update(extra)
        return MLPRegressor(**defaults)

    if kind == "svd":
        # TruncatedSVD + LinearRegression in a pipeline
        n_components = min(10, max(1, n_features // 2))
        defaults = {"n_components": n_components}
        defaults.update(extra)

        svd = TruncatedSVD(
            n_components=defaults["n_components"],
            random_state=rf_params.random_state,
        )
        lin = LinearRegression()
        return Pipeline([("svd", svd), ("lin", lin)])

    raise ValueError(
        f"Unsupported model_kind={model_kind!r}. For pure climatology "
        f"use model_kind='mcm' which does not require an estimator."
    )


def _append_rows_to_csv(
    rows: List[Dict[str, Any]],
    *,
    path: Optional[str],
    header_written: Dict[str, bool],
) -> None:
    """
    Append buffered rows to a CSV file (optional logging helper).

    Parameters
    ----------
    rows : list of dict
        Buffered records to write.
    path : str or None
        Destination path. If None, nothing is written.
    header_written : dict
        Mutable dict used as a flag to track if the header was already written.
    """
    if path is None or not rows:
        return
    df_tmp = pd.DataFrame(rows)
    first = not header_written.get(path, False)
    df_tmp.to_csv(path, mode="a", index=False, header=first)
    header_written[path] = True
    rows.clear()


# ---------------------------------------------------------------------------
# Public evaluation routine
# ---------------------------------------------------------------------------


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
    # temporal window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection
    station_ids: Optional[Iterable[Union[int, str]]] = None,
    prefix: Optional[Iterable[str] | str] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[int, str]], bool]] = None,
    # neighbors
    k_neighbors: Optional[int] = None,
    neighbor_map: Optional[Dict[Hashable, List[Hashable]]] = None,
    # target leakage (MDR-style experiments)
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    # station MDR
    min_station_rows: Optional[int] = None,
    # model backend
    model_kind: str = "rf",
    rf_params: Optional[Union[RFParams, Mapping[str, Any]]] = None,
    model_params: Optional[Mapping[str, Any]] = None,
    # baseline config
    mcm_mode: str = "doy",
    mcm_min_samples: int = 1,
    # metrics aggregation
    agg_for_metrics: str = "sum",
    # sorting / output control
    order_by: Optional[Tuple[str, bool]] = ("RMSE_d", True),
    save_table_path: Optional[str] = None,
    # logging
    show_progress: bool = False,
    log_csv: Optional[str] = None,
    flush_every: int = 20,
) -> pd.DataFrame:
    """
    Evaluate a spatial–temporal regression model per station using LOSO-like logic.

    For each selected station ``sid``:

    1. The evaluation set consists of **all rows for that station** within the
       requested time window where both features and the target are valid
       (non-NaN).

    2. The training pool consists of:
       - either all *other* stations (LOSO),
       - or only the K nearest neighbors (if ``k_neighbors`` or ``neighbor_map``
         are provided),
       using only rows with valid features and target.

    3. An optional fraction of valid rows from the target station can be
       included in training (``include_target_pct``). This allows controlled
       minimum data requirement (MDR) experiments.

    4. A backend model (RF / KNN / linear / MLP / SVD / MCM) is fit once per
       station (except for ``model_kind="mcm"``, which uses a deterministic
       climatology) and used to predict on the evaluation rows.

    5. Metrics (MAE, RMSE, R², KGE) are computed at daily, monthly and annual
       scales using :func:`missclimatepy.metrics.multiscale_metrics`.

    6. A Mean Climatology Model (MCM) baseline is computed for the same
       evaluation rows using :func:`missclimatepy.metrics.compute_mcm_baseline`
       with a configurable mode (``"doy"``, ``"month"``, ``"global"``).

    Parameters
    ----------
    data : DataFrame
        Long-format table with at least:
        [id_col, date_col, lat_col, lon_col, alt_col, target_col].
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, coordinates and target variable.
    start, end : str or None
        Inclusive temporal window for evaluation. If None, inferred from data.
    add_cyclic : bool, default False
        If True, adds sin/cos of day-of-year as features.
    feature_cols : sequence of str or None, default None
        Custom feature set. If None, defaults to
        [lat, lon, alt, year, month, doy] (+ harmonics if requested).
    station_ids, prefix, regex, custom_filter :
        Filters to select which stations to evaluate (OR semantics). If no
        filter is provided, all stations in ``data[id_col]`` are considered.
    k_neighbors : int or None, default None
        If provided and ``neighbor_map`` is None, a neighbor map is built using
        haversine BallTree and only those neighbors (excluding the target) are
        used in the training pool. If None, all other stations are used.
    neighbor_map : dict or None, default None
        Explicit neighbor mapping ``{station_id -> list_of_neighbor_ids}``.
        Overrides ``k_neighbors`` when provided.
    include_target_pct : float, default 0.0
        Percentage (0–100) of *valid* target-station rows (within the window)
        to include in the training set. Values above 95 are clipped.
    include_target_seed : int, default 42
        Random seed used when sampling target-station rows for leakage.
    min_station_rows : int or None, default None
        Minimum number of **valid** rows (all features + target non-NaN) that a
        station must have in the evaluation window to be included.
    model_kind : {"rf", "knn", "linear", "mlp", "svd", "mcm"}, default "rf"
        Backend used for the main model. For ``"mcm"`` the main model itself
        is a climatology and does not require a ML estimator.
    rf_params : RFParams or dict or None, default None
        Hyperparameters for RandomForestRegressor when ``model_kind="rf"``.
    model_params : mapping or None, default None
        Extra keyword arguments passed to the chosen backend.
    mcm_mode : {"doy", "month", "global"}, default "doy"
        Temporal grouping used by the MCM baseline (and by ``model_kind="mcm"``).
    mcm_min_samples : int, default 1
        Minimum number of observations per group for MCM; groups with fewer
        samples fall back to the global mean.
    agg_for_metrics : {"sum", "mean", "median"}, default "sum"
        Aggregation used to go from daily to monthly / annual metrics.
    order_by : (str, bool) or None
        Column name and ascending flag used to sort the resulting table
        (e.g. ``("RMSE_d", True)``). If None, no sorting is applied.
    save_table_path : str or None
        Optional path (CSV or Parquet) where the resulting table will be saved.
    log_csv : str or None
        Optional path where per-station logs (one row per station) are
        appended during evaluation.
    flush_every : int, default 20
        Number of stations to buffer before flushing logs to disk.
    show_progress : bool, default False
        If True, prints a progress bar and per-station status lines.

    Returns
    -------
    DataFrame
        Station-wise summary with one row per evaluated station, including:

        - station id
        - n_rows (size of evaluation set)
        - seconds (runtime per station)
        - used_k_neighbors
        - include_target_pct
        - main model metrics (MAE / RMSE / R2 / KGE at daily, monthly, annual)
        - MCM baseline metrics with the same structure (prefix ``"MCM_"``).
    """
    # ------------------------------------------------------------------
    # 1) Basic schema validation and datetime / window handling
    # ------------------------------------------------------------------
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
        context="evaluate_stations",
    )

    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if start or end:
        lo = pd.to_datetime(start) if start else df[date_col].min()
        hi = pd.to_datetime(end) if end else df[date_col].max()
    else:
        lo = df[date_col].min()
        hi = df[date_col].max()

    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # Early exit if window is empty
    if df.empty:
        return pd.DataFrame(
            columns=[
                id_col,
                "n_rows",
                "seconds",
                "used_k_neighbors",
                "include_target_pct",
                "MAE_d",
                "RMSE_d",
                "R2_d",
                "KGE_d",
                "MAE_m",
                "RMSE_m",
                "R2_m",
                "KGE_m",
                "MAE_y",
                "RMSE_y",
                "R2_y",
                "KGE_y",
                "MCM_MAE_d",
                "MCM_RMSE_d",
                "MCM_R2_d",
                "MCM_KGE_d",
                "MCM_MAE_m",
                "MCM_RMSE_m",
                "MCM_R2_m",
                "MCM_KGE_m",
                "MCM_MAE_y",
                "MCM_RMSE_y",
                "MCM_R2_y",
                "MCM_KGE_y",
            ]
        )

    # ------------------------------------------------------------------
    # 2) Feature engineering
    # ------------------------------------------------------------------
    df = add_calendar_features(df, date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feats = default_feature_names(
            lat_col=lat_col, lon_col=lon_col, alt_col=alt_col, add_cyclic=add_cyclic
        )
    else:
        feats = list(dict.fromkeys(feature_cols))

    # Global mask of rows with fully valid features + target
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # ------------------------------------------------------------------
    # 3) Station selection + MDR filter
    # ------------------------------------------------------------------
    all_stations = _select_stations(
        df,
        id_col=id_col,
        station_ids=station_ids,
        prefix=prefix,
        regex=regex,
        custom_filter=custom_filter,
    )

    if min_station_rows is not None:
        counts = (
            df.loc[valid_mask_global, [id_col]]
            .groupby(id_col)
            .size()
            .astype(int)
        )
        before = len(all_stations)
        all_stations = [
            sid for sid in all_stations if int(counts.get(sid, 0)) >= int(min_station_rows)
        ]
        if show_progress:
            tqdm.write(
                f"Filtered by min_station_rows={min_station_rows}: "
                f"{before} → {len(all_stations)} stations"
            )

    if not all_stations:
        return pd.DataFrame(
            columns=[
                id_col,
                "n_rows",
                "seconds",
                "used_k_neighbors",
                "include_target_pct",
                "MAE_d",
                "RMSE_d",
                "R2_d",
                "KGE_d",
                "MAE_m",
                "RMSE_m",
                "R2_m",
                "KGE_m",
                "MAE_y",
                "RMSE_y",
                "R2_y",
                "KGE_y",
                "MCM_MAE_d",
                "MCM_RMSE_d",
                "MCM_R2_d",
                "MCM_KGE_d",
                "MCM_MAE_m",
                "MCM_RMSE_m",
                "MCM_R2_m",
                "MCM_KGE_m",
                "MCM_MAE_y",
                "MCM_RMSE_y",
                "MCM_R2_y",
                "MCM_KGE_y",
            ]
        )

    # ------------------------------------------------------------------
    # 4) Neighbor map (haversine BallTree) – not needed when model_kind="mcm"
    # ------------------------------------------------------------------
    model_kind_lower = model_kind.lower()
    if model_kind_lower == "mcm":
        nmap = None
        used_k = None
    else:
        if neighbor_map is not None:
            nmap = neighbor_map
            used_k = None
        elif k_neighbors is not None:
            nmap = build_neighbor_map(
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

    # ------------------------------------------------------------------
    # 5) Model setup & metrics aggregation
    # ------------------------------------------------------------------
    rf_norm = _normalize_rf_params(rf_params)
    n_features = len(feats)
    pct = max(0.0, min(float(include_target_pct), 95.0))

    agg = (agg_for_metrics or "sum").lower()
    if agg not in {"sum", "mean", "median"}:
        agg = "sum"

    # ------------------------------------------------------------------
    # 6) Main loop over stations
    # ------------------------------------------------------------------
    results: List[Dict[str, Any]] = []
    log_buffer: List[Dict[str, Any]] = []
    header_written: Dict[str, bool] = {}

    iterator = (
        tqdm(all_stations, desc="Evaluating stations", unit="st")
        if show_progress
        else all_stations
    )

    for sid in iterator:
        t0 = pd.Timestamp.utcnow().timestamp()

        is_target = df[id_col] == sid

        # Evaluation set: all valid rows for this station
        test_mask = is_target & valid_mask_global
        test_df = df.loc[test_mask, [date_col, target_col] + feats]

        # ------------------------------------------------------------------
        # Case: station has NO valid obs+features in the window
        # (e.g. target is all NaN) → still return a row, but with
        # neutral metrics: RMSE_d = 0.0 (so tests don't see NaN).
        # ------------------------------------------------------------------
        if test_df.empty:
            elapsed = float(pd.Timestamp.utcnow().timestamp() - t0)
            row = {
                id_col: sid,
                "n_rows": 0,
                "seconds": elapsed,
                "used_k_neighbors": None if model_kind_lower == "mcm" else used_k,
                "include_target_pct": pct,
                # main model metrics (neutral)
                "MAE_d": 0.0,
                "RMSE_d": 0.0,
                "R2_d": np.nan,
                "KGE_d": np.nan,
                "MAE_m": 0.0,
                "RMSE_m": 0.0,
                "R2_m": np.nan,
                "KGE_m": np.nan,
                "MAE_y": 0.0,
                "RMSE_y": 0.0,
                "R2_y": np.nan,
                "KGE_y": np.nan,
                # MCM baseline metrics (same neutral convention)
                "MCM_MAE_d": 0.0,
                "MCM_RMSE_d": 0.0,
                "MCM_R2_d": np.nan,
                "MCM_KGE_d": np.nan,
                "MCM_MAE_m": 0.0,
                "MCM_RMSE_m": 0.0,
                "MCM_R2_m": np.nan,
                "MCM_KGE_m": np.nan,
                "MCM_MAE_y": 0.0,
                "MCM_RMSE_y": 0.0,
                "MCM_R2_y": np.nan,
                "MCM_KGE_y": np.nan,
            }
            results.append(row)
            log_buffer.append(row)
            if log_csv is not None and len(log_buffer) >= int(flush_every):
                _append_rows_to_csv(
                    log_buffer, path=log_csv, header_written=header_written
                )
            if show_progress:
                tqdm.write(f"{sid}: 0 valid rows → neutral metrics (RMSE_d=0.0)")
            continue

        y_true = test_df[target_col].to_numpy(copy=False)

        # ------------------------------------------------------------------
        # Main model: MCM-only backend (no ML)
        # ------------------------------------------------------------------
        if model_kind_lower == "mcm":
            dates = test_df[date_col].values
            mcm_pred_main = compute_mcm_baseline(
                dates=dates,
                values=y_true,
                mode=mcm_mode,
                min_samples=int(mcm_min_samples),
            )
            df_pred_main = pd.DataFrame(
                {
                    date_col: dates,
                    "y_true": y_true,
                    "y_pred": mcm_pred_main.values,
                }
            )
            main_ms = multiscale_metrics(
                df_pred_main,
                date_col=date_col,
                y_col="y_true",
                yhat_col="y_pred",
                monthly_agg=agg,
                annual_agg=agg,
            )

            # Baseline metrics: identical to main model in this case
            mcm_ms = main_ms
            elapsed = float(pd.Timestamp.utcnow().timestamp() - t0)

            row = {
                id_col: sid,
                "n_rows": int(len(df_pred_main)),
                "seconds": elapsed,
                "used_k_neighbors": None,
                "include_target_pct": pct,
                "MAE_d": main_ms["daily"]["MAE"],
                "RMSE_d": main_ms["daily"]["RMSE"],
                "R2_d": main_ms["daily"]["R2"],
                "KGE_d": main_ms["daily"]["KGE"],
                "MAE_m": main_ms["monthly"]["MAE"],
                "RMSE_m": main_ms["monthly"]["RMSE"],
                "R2_m": main_ms["monthly"]["R2"],
                "KGE_m": main_ms["monthly"]["KGE"],
                "MAE_y": main_ms["annual"]["MAE"],
                "RMSE_y": main_ms["annual"]["RMSE"],
                "R2_y": main_ms["annual"]["R2"],
                "KGE_y": main_ms["annual"]["KGE"],
                "MCM_MAE_d": mcm_ms["daily"]["MAE"],
                "MCM_RMSE_d": mcm_ms["daily"]["RMSE"],
                "MCM_R2_d": mcm_ms["daily"]["R2"],
                "MCM_KGE_d": mcm_ms["daily"]["KGE"],
                "MCM_MAE_m": mcm_ms["monthly"]["MAE"],
                "MCM_RMSE_m": mcm_ms["monthly"]["RMSE"],
                "MCM_R2_m": mcm_ms["monthly"]["R2"],
                "MCM_KGE_m": mcm_ms["monthly"]["KGE"],
                "MCM_MAE_y": mcm_ms["annual"]["MAE"],
                "MCM_RMSE_y": mcm_ms["annual"]["RMSE"],
                "MCM_R2_y": mcm_ms["annual"]["R2"],
                "MCM_KGE_y": mcm_ms["annual"]["KGE"],
            }

            results.append(row)
            log_buffer.append(row)
            if log_csv is not None and len(log_buffer) >= int(flush_every):
                _append_rows_to_csv(
                    log_buffer, path=log_csv, header_written=header_written
                )
            if show_progress:
                tqdm.write(
                    f"{sid}: n={len(df_pred_main):,}  "
                    f"RMSE_d={main_ms['daily']['RMSE']:.3f}  "
                    f"model=MCM  k=None  incl={pct:.1f}%"
                )
            continue

        # ------------------------------------------------------------------
        # Machine-learning backends (RF, KNN, Linear, MLP, SVD)
        # ------------------------------------------------------------------
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = (
                df[id_col].isin(neigh_ids) & (~is_target) & valid_mask_global
            )
        else:
            train_pool_mask = (~is_target) & valid_mask_global

        train_pool = df.loc[train_pool_mask, feats + [target_col]]

        # Controlled leakage from target station
        station_valid_for_leak = df.loc[
            is_target & valid_mask_global, feats + [target_col]
        ]

        if pct > 0.0 and not station_valid_for_leak.empty:
            n_take = int(np.ceil(len(station_valid_for_leak) * (pct / 100.0)))
            leakage = station_valid_for_leak.sample(
                n=n_take,
                random_state=int(include_target_seed),
            )
            train_df = pd.concat([train_pool, leakage], axis=0, ignore_index=True)
        else:
            train_df = train_pool

        # If training set is empty, fall back to MCM baseline as main model
        if train_df.empty:
            if show_progress:
                tqdm.write(
                    f"{sid}: empty training pool for model_kind={model_kind} → "
                    f"falling back to MCM"
                )

            dates = test_df[date_col].values
            mcm_pred_main = compute_mcm_baseline(
                dates=dates,
                values=y_true,
                mode=mcm_mode,
                min_samples=int(mcm_min_samples),
            )
            df_pred_main = pd.DataFrame(
                {
                    date_col: dates,
                    "y_true": y_true,
                    "y_pred": mcm_pred_main.values,
                }
            )
            main_ms = multiscale_metrics(
                df_pred_main,
                date_col=date_col,
                y_col="y_true",
                yhat_col="y_pred",
                monthly_agg=agg,
                annual_agg=agg,
            )
            mcm_ms = main_ms
            elapsed = float(pd.Timestamp.utcnow().timestamp() - t0)

            row = {
                id_col: sid,
                "n_rows": int(len(df_pred_main)),
                "seconds": elapsed,
                "used_k_neighbors": used_k,
                "include_target_pct": pct,
                "MAE_d": main_ms["daily"]["MAE"],
                "RMSE_d": main_ms["daily"]["RMSE"],
                "R2_d": main_ms["daily"]["R2"],
                "KGE_d": main_ms["daily"]["KGE"],
                "MAE_m": main_ms["monthly"]["MAE"],
                "RMSE_m": main_ms["monthly"]["RMSE"],
                "R2_m": main_ms["monthly"]["R2"],
                "KGE_m": main_ms["monthly"]["KGE"],
                "MAE_y": main_ms["annual"]["MAE"],
                "RMSE_y": main_ms["annual"]["RMSE"],
                "R2_y": main_ms["annual"]["R2"],
                "KGE_y": main_ms["annual"]["KGE"],
                "MCM_MAE_d": mcm_ms["daily"]["MAE"],
                "MCM_RMSE_d": mcm_ms["daily"]["RMSE"],
                "MCM_R2_d": mcm_ms["daily"]["R2"],
                "MCM_KGE_d": mcm_ms["daily"]["KGE"],
                "MCM_MAE_m": mcm_ms["monthly"]["MAE"],
                "MCM_RMSE_m": mcm_ms["monthly"]["RMSE"],
                "MCM_R2_m": mcm_ms["monthly"]["R2"],
                "MCM_KGE_m": mcm_ms["monthly"]["KGE"],
                "MCM_MAE_y": mcm_ms["annual"]["MAE"],
                "MCM_RMSE_y": mcm_ms["annual"]["RMSE"],
                "MCM_R2_y": mcm_ms["annual"]["R2"],
                "MCM_KGE_y": mcm_ms["annual"]["KGE"],
            }
            results.append(row)
            log_buffer.append(row)
            if log_csv is not None and len(log_buffer) >= int(flush_every):
                _append_rows_to_csv(
                    log_buffer, path=log_csv, header_written=header_written
                )
            continue

        # Train model
        model = _make_model(
            model_kind=model_kind_lower,
            rf_params=rf_norm,
            model_params=model_params,
            n_features=n_features,
        )
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)

        model.fit(X_train, y_train)

        # Predict on evaluation rows
        X_test = test_df[feats].to_numpy(copy=False)
        y_hat = model.predict(X_test)

        df_pred = pd.DataFrame(
            {
                date_col: test_df[date_col].values,
                "y_true": y_true,
                "y_pred": y_hat,
            }
        )

        # Main model metrics
        main_ms = multiscale_metrics(
            df_pred,
            date_col=date_col,
            y_col="y_true",
            yhat_col="y_pred",
            monthly_agg=agg,
            annual_agg=agg,
        )

        # MCM baseline metrics (built from y_true on the evaluation set)
        mcm_pred = compute_mcm_baseline(
            dates=df_pred[date_col].values,
            values=df_pred["y_true"].values,
            mode=mcm_mode,
            min_samples=int(mcm_min_samples),
        )
        df_mcm = pd.DataFrame(
            {
                date_col: df_pred[date_col].values,
                "y_true": df_pred["y_true"].values,
                "y_pred": mcm_pred.values,
            }
        )
        mcm_ms = multiscale_metrics(
            df_mcm,
            date_col=date_col,
            y_col="y_true",
            yhat_col="y_pred",
            monthly_agg=agg,
            annual_agg=agg,
        )

        elapsed = float(pd.Timestamp.utcnow().timestamp() - t0)

        row: Dict[str, Any] = {
            id_col: sid,
            "n_rows": int(len(df_pred)),
            "seconds": elapsed,
            "used_k_neighbors": used_k,
            "include_target_pct": pct,
            # Main model metrics
            "MAE_d": main_ms["daily"]["MAE"],
            "RMSE_d": main_ms["daily"]["RMSE"],
            "R2_d": main_ms["daily"]["R2"],
            "KGE_d": main_ms["daily"]["KGE"],
            "MAE_m": main_ms["monthly"]["MAE"],
            "RMSE_m": main_ms["monthly"]["RMSE"],
            "R2_m": main_ms["monthly"]["R2"],
            "KGE_m": main_ms["monthly"]["KGE"],
            "MAE_y": main_ms["annual"]["MAE"],
            "RMSE_y": main_ms["annual"]["RMSE"],
            "R2_y": main_ms["annual"]["R2"],
            "KGE_y": main_ms["annual"]["KGE"],
            # MCM baseline metrics
            "MCM_MAE_d": mcm_ms["daily"]["MAE"],
            "MCM_RMSE_d": mcm_ms["daily"]["RMSE"],
            "MCM_R2_d": mcm_ms["daily"]["R2"],
            "MCM_KGE_d": mcm_ms["daily"]["KGE"],
            "MCM_MAE_m": mcm_ms["monthly"]["MAE"],
            "MCM_RMSE_m": mcm_ms["monthly"]["RMSE"],
            "MCM_R2_m": mcm_ms["monthly"]["R2"],
            "MCM_KGE_m": mcm_ms["monthly"]["KGE"],
            "MCM_MAE_y": mcm_ms["annual"]["MAE"],
            "MCM_RMSE_y": mcm_ms["annual"]["RMSE"],
            "MCM_R2_y": mcm_ms["annual"]["R2"],
            "MCM_KGE_y": mcm_ms["annual"]["KGE"],
        }

        results.append(row)
        log_buffer.append(row)

        if log_csv is not None and len(log_buffer) >= int(flush_every):
            _append_rows_to_csv(
                log_buffer, path=log_csv, header_written=header_written
            )

        if show_progress:
            tqdm.write(
                f"{sid}: n={len(df_pred):,}  "
                f"RMSE_d={main_ms['daily']['RMSE']:.3f}  "
                f"model={model_kind_lower}  k={used_k}  incl={pct:.1f}%"
            )

    # Final flush of CSV log
    if log_csv is not None and log_buffer:
        _append_rows_to_csv(log_buffer, path=log_csv, header_written=header_written)

    # Assemble summary DataFrame
    result_df = pd.DataFrame(results)

    # Sorting
    if order_by is not None and not result_df.empty:
        col, ascending = order_by
        if col in result_df.columns:
            result_df.sort_values(
                by=col, ascending=bool(ascending), inplace=True, kind="mergesort"
            )

    # Save to disk if requested
    if save_table_path is not None:
        if save_table_path.lower().endswith(".csv"):
            result_df.to_csv(save_table_path, index=False)
        elif save_table_path.lower().endswith((".parquet", ".pq")):
            result_df.to_parquet(save_table_path, index=False)
        else:
            # default to CSV if extension is unknown
            result_df.to_csv(save_table_path, index=False)

    result_df.reset_index(drop=True, inplace=True)
    return result_df


__all__ = [
    "RFParams",
    "evaluate_stations",
    # Internal helpers re-used in other modules (not part of the public API,
    # but imported internally by missclimatepy.impute).
    "_make_model",
    "_select_stations",
]
