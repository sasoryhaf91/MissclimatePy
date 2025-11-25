# SPDX-License-Identifier: MIT
"""
missclimatepy.impute
====================

Local XYZT-style imputation for a single target variable.

This module implements a *minimal* yet expressive imputation engine for
long-format daily climate records. The core philosophy is:

- Use **only** space–time coordinates as predictors:
  - Spatial: latitude, longitude, altitude  → (x, y, z)
  - Temporal: calendar features derived from the date  → (t)
- Train a **local model per station**, using:
  - Neighboring stations in space (K-nearest or a user-provided map), and
  - An optional fraction of the target station’s own observed history.

The goal is to reconstruct complete daily series for each selected station
over a given time window, while remaining independent of internal covariates
that are often also missing in practice.

Key ideas
---------

- **Schema-agnostic**: You provide column names for id, date, coordinates and
  target; the function works with any naming convention.

- **Local models**: For each station, a model is trained from:
  - either its K nearest neighbors (based on haversine distance in (lat, lon)),
  - or *all other* stations when ``k_neighbors=None``,
  - plus an optional fraction of the station’s own observed rows.

- **XYZT features only**:
  - Coordinates: ``lat_col``, ``lon_col``, ``alt_col``.
  - Calendar: ``year``, ``month``, ``doy`` (and optionally ``doy_sin``,
    ``doy_cos`` when ``add_cyclic=True``).

- **Model-agnostic**: Any regressor supported by :mod:`missclimatepy.models`
  can be used via ``model_kind`` and ``model_params`` (e.g., ``"rf"` for
  Random Forest, ``"etr"`` for Extra Trees, ``"linreg"`` for linear
  regression, etc.), as long as it accepts a 2D array of XYZT features.

- **Minimal tidy output**: The returned table always has exactly:

    ``[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]``

  where ``source ∈ {"observed", "imputed"}``.

Station eligibility (min_station_rows)
--------------------------------------

A station is imputed only if the number of **observed (non-NaN) target rows**
within the analysis window [start, end] is at least ``min_station_rows``
(when this parameter is provided and > 0). If ``min_station_rows`` is
``None`` or 0, *no minimum* is enforced and all selected stations are
eligible.

Information visibility (include_target_pct)
-------------------------------------------

For each eligible station, the fraction of its **own observed rows** that is
used in the training set is controlled by ``include_target_pct``:

- ``include_target_pct is None or >= 100``:
    All observed rows of the station are allowed into the training set
    (full inclusion, original behavior).

- ``0 < include_target_pct < 100``:
    Only that percentage (floored) of the station's observed rows is
    randomly selected (without replacement) and added to the training set.
    The selection is done with a reproducible RNG controlled by
    ``include_target_seed``. The remaining observed rows are kept in the
    output as ``source="observed"``, but are *not* seen by the model.

- ``include_target_pct == 0``:
    No local rows are used in training (extreme LOSO-like scenario);
    imputation is driven solely by neighboring stations (when available).

Note that this is **not** a train/test split for evaluation – it only
controls how much the local history is exposed to the model during
imputation. For proper validation, use :func:`missclimatepy.evaluate.evaluate_stations`.

Persistence
-----------

Results can be saved automatically using:

- ``save_path``: destination path (CSV or Parquet).
- ``save_format``: ``"csv"``, ``"parquet"`` or ``"auto"`` (infer from suffix).
- ``save_index``: whether to write the index.
- ``save_partitions``: if True, write **one file per station**.

Examples
--------

>>> from missclimatepy.impute import impute_dataset
>>> out = impute_dataset(
...     data=df,
...     id_col="station",
...     date_col="date",
...     lat_col="lat",
...     lon_col="lon",
...     alt_col="alt",
...     target_col="tmin",
...     start="1981-01-01",
...     end="2023-12-31",
...     k_neighbors=20,
...     model_kind="rf",
...     model_params={"n_estimators": 300, "max_depth": 30, "n_jobs": -1, "random_state": 42},
...     min_station_rows=365,
...     include_target_pct=50.0,
...     include_target_seed=42,
...     show_progress=True,
...     save_path="outputs/tmin_imputed.parquet",
...     save_format="auto",
...     save_partitions=False,
... )
>>> out.columns.tolist()
['station', 'date', 'latitude', 'longitude', 'altitude', 'tmin', 'source']
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union, Literal

import numpy as np
import pandas as pd

from .features import (
    validate_required_columns,
    ensure_datetime_naive,
    add_calendar_features,
    default_feature_names,
)
from .neighbors import build_neighbor_map
from .models import make_model

try:  # optional progress bar
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover

    def tqdm(x, **kwargs):
        return x  # transparent iterator


# ---------------------------------------------------------------------------
# Internal: persistence helpers
# ---------------------------------------------------------------------------


def _infer_format_from_path(path: str) -> Literal["csv", "parquet"]:
    """
    Infer file format from the output path extension.

    - ``*.parquet``           → ``"parquet"``
    - ``*.csv[.gz|.bz2|...]`` → ``"csv"``
    - otherwise               → ``"csv"`` (conservative default)
    """
    p = path.lower()
    if p.endswith(".parquet"):
        return "parquet"
    if (
        p.endswith(".csv")
        or p.endswith(".csv.gz")
        or p.endswith(".csv.bz2")
        or p.endswith(".csv.xz")
        or p.endswith(".csv.zip")
    ):
        return "csv"
    return "csv"


def _ensure_parent_dir(path: Path) -> None:
    """
    Create parent directories for ``path`` if they do not exist.
    """
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def _write_csv(df: pd.DataFrame, dest: Path, *, index: bool) -> None:
    """
    Write a DataFrame to CSV (compression inferred from suffix).
    """
    _ensure_parent_dir(dest)
    df.to_csv(dest, index=index)


def _write_parquet(df: pd.DataFrame, dest: Path, *, index: bool) -> None:
    """
    Write a DataFrame to Parquet.
    """
    _ensure_parent_dir(dest)
    df.to_parquet(dest, index=index)


def _write_partitions(
    df: pd.DataFrame,
    *,
    station_col: str,
    base_path: Path,
    fmt: Literal["csv", "parquet"],
    index: bool,
) -> None:
    """
    Write one file (CSV/Parquet) per station.

    - Parquet: ``base_path / f"station=<sid>/part.parquet"`` (Hive-like layout).
    - CSV:     ``base_name + "_station=<sid>.csv[.gz]"`` per station.
    """
    if fmt == "parquet":
        # Partitioned folders: base/station=<sid>/part.parquet
        for sid, sdf in df.groupby(station_col, sort=False):
            part_dir = base_path / f"station={sid}"
            _ensure_parent_dir(part_dir / "part.parquet")
            sdf.to_parquet(part_dir / "part.parquet", index=index)
    else:
        # CSV per station with suffix
        base = str(base_path)
        for sid, sdf in df.groupby(station_col, sort=False):
            lower = base.lower()
            if lower.endswith(".csv"):
                out_name = base[:-4] + f"_station={sid}.csv"
            elif any(lower.endswith(ext) for ext in (".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip")):
                # split at ".csv" and keep the rest (e.g., ".gz")
                pos = lower.rfind(".csv")
                out_name = base[:pos] + f"_station={sid}" + base[pos:]
            else:
                # no extension -> add .csv
                out_name = base + f"_station={sid}.csv"
            dest = Path(out_name)
            _ensure_parent_dir(dest)
            sdf.to_csv(dest, index=index)


def _save_result_df(
    df: pd.DataFrame,
    *,
    path: Optional[str],
    fmt: Literal["csv", "parquet", "auto"],
    index: bool,
    partition: bool,
    station_col: str,
) -> None:
    """
    Save the imputed DataFrame if ``path`` is provided. No-op if ``path`` is None.

    This helper:

    - infers format if ``fmt="auto"``,
    - validates a minimal schema,
    - writes a single file or one file per station.
    """
    if path is None:
        return

    base = Path(path)
    if fmt == "auto":
        fmt_resolved = _infer_format_from_path(str(base))
    else:
        fmt_resolved = fmt

    if fmt_resolved not in ("csv", "parquet"):
        raise ValueError(
            f"Unsupported save_format '{fmt}'. Use 'csv', 'parquet', or 'auto'."
        )

    # Minimal schema: station id, date, coordinates and source flag
    required = {station_col, "date", "latitude", "longitude", "altitude", "source"}
    if not required.issubset(df.columns):
        raise ValueError(
            "Result schema missing required columns for saving. "
            f"Expected to include: {sorted(required)}; got: {list(df.columns)[:10]}..."
        )

    if partition:
        _write_partitions(
            df,
            station_col=station_col,
            base_path=base,
            fmt=fmt_resolved,
            index=index,
        )
        return

    if fmt_resolved == "parquet":
        _write_parquet(df, base, index=index)
    else:
        _write_csv(df, base, index=index)


# ---------------------------------------------------------------------------
# Public imputation API
# ---------------------------------------------------------------------------


def impute_dataset(
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
    # features (calendar)
    add_cyclic: bool = False,
    # station selection (optional, OR semantics)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # neighbors
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    # minimum observed rows within the window
    min_station_rows: Optional[int] = None,
    # local history visibility
    include_target_pct: Optional[float] = None,
    include_target_seed: int = 42,
    # model (generic XYZT regressor)
    model_kind: str = "rf",
    model_params: Optional[Mapping[str, Any]] = None,
    # logging
    show_progress: bool = False,
    # persistence
    save_path: Optional[str] = None,
    save_format: Literal["csv", "parquet", "auto"] = "auto",
    save_index: bool = False,
    save_partitions: bool = False,
) -> pd.DataFrame:
    """
    Impute a single target variable over a daily window using local XYZT models.

    This function trains one model *per station* and returns a minimal long-format
    DataFrame containing both observed and imputed values over the requested
    [start, end] interval. Only space–time coordinates (x, y, z, t) are used as
    predictors.

    Parameters
    ----------
    data : DataFrame
        Long-format input with at least:

        ``[id_col, date_col, lat_col, lon_col, alt_col, target_col]``.

        The target column may contain NaNs (gaps) to be imputed.
    id_col, date_col, lat_col, lon_col, alt_col, target_col : str
        Column names for station id, timestamp, spatial coordinates and target.
    start, end : str or None
        Inclusive analysis window. If None, inferred from the data span.
    add_cyclic : bool, optional
        If True, adds harmonic day-of-year terms (``doy_sin``, ``doy_cos``) to
        the feature set.
    prefix, station_ids, regex, custom_filter :
        Optional filters to select which stations to process (OR semantics):

        - ``prefix``: string or iterable of strings; keep stations whose id
          starts with any of these prefixes.
        - ``station_ids``: explicit list of station ids to include.
        - ``regex``: regular expression that station ids must match.
        - ``custom_filter``: callable ``f(id) -> bool``.
          If no filters are provided, *all* stations in ``data`` are considered.
    k_neighbors : int or None, default 20
        If provided and ``neighbor_map`` is None, builds a KNN haversine
        neighbor map using station coordinates and trains each station’s model
        with those neighbors (excluding the target). If None, all other
        stations are used as the spatial pool.
    neighbor_map : dict or None
        Overrides ``k_neighbors``. Mapping
        ``{station_id -> list_of_neighbor_ids}`` used directly.
    min_station_rows : int or None
        Minimum number of **observed** target rows within [start, end] required
        for a station to be imputed. If None or 0, no minimum is enforced.

    include_target_pct : float in [0, 100] or None
        Fraction of each station's **observed** target rows allowed into the
        training set. Values ≥ 100 or None correspond to full inclusion.
        When 0, the model never sees the target station’s history (LOSO-like).
    include_target_seed : int
        Seed for the random sampling of local rows when
        ``0 < include_target_pct < 100``.

    model_kind : str, default "rf"
        Name of the model family to use. The available options and default
        hyperparameters are defined in :mod:`missclimatepy.models`.
        Examples include ``"rf"`` (Random Forest), ``"etr"`` (Extra Trees),
        ``"gbrt"``, ``"hgbt"``, ``"knn"``, ``"mlp"``, ``"svr"``,
        ``"linreg"``, ``"elasticnet"``, and others.
    model_params : mapping or None
        Dictionary of hyperparameters for the chosen model. Any missing keys
        fall back to the model-specific defaults.

    show_progress : bool
        If True, prints per-station progress lines using ``tqdm`` when
        available.
    save_path : str or None
        If provided, the resulting DataFrame is written to this path:

        - If ``save_format="auto"``, the format is inferred from the suffix:
          ``*.parquet`` → Parquet; ``*.csv[.gz|...]`` → CSV.
        - Parent directories are created as needed.
    save_format : {"csv","parquet","auto"}
        File format for saving. Use "auto" to infer from the extension.
    save_index : bool
        Whether to write the DataFrame index to disk.
    save_partitions : bool
        If True, write **one file per station**:

        - Parquet: ``base_dir / "station=<ID>/part.parquet"``
        - CSV: ``base_name + "_station=<ID>.csv[.gz]"``

    Returns
    -------
    DataFrame
        Minimal long-format table with **exactly** the columns:

        ``[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]``

        where:

        - ``source = "observed"`` for original non-missing values, and
        - ``source = "imputed"`` for values filled by the model.
    """
    # ------------------------------------------------------------------
    # 1) Basic validation & time window
    # ------------------------------------------------------------------
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
    )

    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    if df.empty:
        # Empty scaffold with canonical output columns
        empty = pd.DataFrame(
            columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]
        )
        _save_result_df(
            empty,
            path=save_path,
            fmt=save_format,
            index=save_index,
            partition=save_partitions,
            station_col=id_col,
        )
        return empty

    lo = pd.to_datetime(start) if start is not None else df[date_col].min()
    hi = pd.to_datetime(end) if end is not None else df[date_col].max()
    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    if df.empty:
        empty = pd.DataFrame(
            columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]
        )
        _save_result_df(
            empty,
            path=save_path,
            fmt=save_format,
            index=save_index,
            partition=save_partitions,
            station_col=id_col,
        )
        return empty

    # ------------------------------------------------------------------
    # 2) Feature engineering (calendar + XYZT)
    # ------------------------------------------------------------------
    df = add_calendar_features(df, date_col=date_col, add_cyclic=add_cyclic)
    feats = default_feature_names(
        lat_col=lat_col,
        lon_col=lon_col,
        alt_col=alt_col,
        add_cyclic=add_cyclic,
        extra=None,
    )

    # ------------------------------------------------------------------
    # 3) Station selection (OR semantics)
    # ------------------------------------------------------------------
    all_ids = df[id_col].dropna().unique().tolist()
    chosen: List[Union[str, int]] = []

    if prefix is not None:
        if isinstance(prefix, str):
            prefix_iter = [prefix]
        else:
            prefix_iter = list(prefix)
        for p in prefix_iter:
            chosen.extend([s for s in all_ids if str(s).startswith(str(p))])

    if station_ids is not None:
        chosen.extend(list(station_ids))

    if regex is not None:
        import re

        pat = re.compile(regex)
        chosen.extend([s for s in all_ids if pat.search(str(s))])

    if custom_filter is not None:
        chosen.extend([s for s in all_ids if custom_filter(s)])

    if not chosen:
        chosen = all_ids

    # Stable order, unique
    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # ------------------------------------------------------------------
    # 4) MDR filter (by observed target rows in window)
    # ------------------------------------------------------------------
    if min_station_rows is not None and min_station_rows > 0:
        observed_mask = ~df[target_col].isna()
        obs_counts = (
            df.loc[observed_mask, [id_col, target_col]]
            .groupby(id_col, sort=False)[target_col]
            .size()
            .astype(int)
        )
        eligible = [s for s in stations if int(obs_counts.get(s, 0)) >= int(min_station_rows)]
        if show_progress and len(eligible) != len(stations):
            skipped = [s for s in stations if s not in eligible]
            tqdm.write(
                f"min_station_rows(observed≥{min_station_rows}): "
                f"{len(stations)} → {len(eligible)} stations "
                f"(skipped {len(skipped)})"
            )
        stations = eligible

    if not stations:
        empty = pd.DataFrame(
            columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]
        )
        _save_result_df(
            empty,
            path=save_path,
            fmt=save_format,
            index=save_index,
            partition=save_partitions,
            station_col=id_col,
        )
        return empty

    # ------------------------------------------------------------------
    # 5) Neighbor map / training pool
    # ------------------------------------------------------------------
    if neighbor_map is not None:
        nmap = neighbor_map
    elif k_neighbors is not None:
        nmap = build_neighbor_map(
            data=df,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=int(k_neighbors),
            include_self=False,
        )
    else:
        nmap = None

    # Canonical date grid & station medoids for output
    full_dates = pd.date_range(lo, hi, freq="D")
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median(numeric_only=True)
        .rename(columns={lat_col: "latitude", lon_col: "longitude", alt_col: "altitude"})
    )

    # Global mask of rows with complete XYZT + target for training
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    # Prepare model factory
    # (Validation of model_kind happens inside make_model)
    # ------------------------------------------------------------------
    iterator = tqdm(stations, desc="Imputing stations", unit="st") if show_progress else stations
    out_blocks: List[pd.DataFrame] = []

    for sid in iterator:
        # Medoid coordinates for reporting / grid
        lat0 = float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan
        lon0 = float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan
        alt0 = float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan

        # Base date grid for this station
        grid = pd.DataFrame({date_col: full_dates})
        grid[id_col] = sid
        # Coordinates in original schema (for XYZT features)
        grid[lat_col] = lat0
        grid[lon_col] = lon0
        grid[alt_col] = alt0

        # Attach observed target values (restricted to window)
        st_obs = df.loc[df[id_col] == sid, [date_col, target_col]]
        merged = grid.merge(st_obs, on=date_col, how="left")

        # Recompute calendar features on the full grid
        merged = add_calendar_features(merged, date_col=date_col, add_cyclic=add_cyclic)

        # Training pool: neighbors or all other stations (excluding target)
        is_target_mask = df[id_col] == sid
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            pool_mask = df[id_col].isin(neigh_ids) & (~is_target_mask) & valid_mask_global
        else:
            pool_mask = (~is_target_mask) & valid_mask_global

        train_pool = df.loc[pool_mask, feats + [target_col]]

        # Fallback: if neighbor pool is empty, use all other stations with
        # observed target within the window.
        if train_pool.empty:
            alt_pool_mask = (~is_target_mask) & (~df[target_col].isna()) & valid_mask_global
            train_pool = df.loc[alt_pool_mask, feats + [target_col]]

        # Local valid rows for this station (features + target present)
        st_valid = df.loc[
            is_target_mask & valid_mask_global,
            feats + [target_col],
        ]

        # --------------------------------------------------------------
        # 6) include_target_pct: random, reproducible sampling of local rows
        # --------------------------------------------------------------
        if include_target_pct is not None:
            if include_target_pct < 0:
                raise ValueError("include_target_pct must be >= 0 or None.")
            if include_target_pct >= 100.0:
                st_valid_for_train = st_valid  # full inclusion
            else:
                n_local = len(st_valid)
                if n_local > 0:
                    k_sel = int(np.floor(n_local * (include_target_pct / 100.0)))
                    if k_sel > 0:
                        rng = np.random.RandomState(int(include_target_seed))
                        idx = rng.choice(st_valid.index.to_numpy(), size=k_sel, replace=False)
                        st_valid_for_train = st_valid.loc[idx]
                    else:
                        # no local rows used in training
                        st_valid_for_train = st_valid.iloc[0:0, :]
                else:
                    st_valid_for_train = st_valid
        else:
            # None → same as full inclusion
            st_valid_for_train = st_valid

        # Concatenate training pool + selected local rows
        train_df = pd.concat([train_pool, st_valid_for_train], axis=0, ignore_index=True)

        if train_df.empty:
            # No training information at all → return observed-only rows
            y_obs = merged[target_col].to_numpy()
            mask_nan = np.isnan(y_obs)

            source = np.empty(len(y_obs), dtype="object")
            source[mask_nan] = np.nan
            source[~mask_nan] = "observed"

            out = pd.DataFrame(
                {
                    id_col: sid,
                    "date": merged[date_col].to_numpy(),
                    "latitude": np.full(len(merged), lat0, dtype=float),
                    "longitude": np.full(len(merged), lon0, dtype=float),
                    "altitude": np.full(len(merged), alt0, dtype=float),
                    target_col: y_obs,
                    "source": source,
                }
            )
            out_blocks.append(out)
            if show_progress:
                tqdm.write(f"Station {sid}: empty training set → observed-only.")
            continue

        # ------------------------------------------------------------------
        # 7) Fit model & predict full grid
        # ------------------------------------------------------------------
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)

        model = make_model(model_kind=model_kind, model_params=model_params)
        model.fit(X_train, y_train)

        X_full = merged[feats].to_numpy(copy=False)
        y_hat = model.predict(X_full)

        # Observed values win; only fill gaps
        y_obs = merged[target_col].to_numpy()
        mask_nan = np.isnan(y_obs)
        filled = y_obs.copy()
        filled[mask_nan] = y_hat[mask_nan]

        # Build 'source'
        if merged[target_col].isna().all():
            # Pure LOSO (no observations at all): all rows imputed
            source = np.full(len(merged), "imputed", dtype="object")
        else:
            source = np.empty(len(y_obs), dtype="object")
            source[mask_nan] = "imputed"
            source[~mask_nan] = "observed"

        out = pd.DataFrame(
            {
                id_col: sid,
                "date": merged[date_col].to_numpy(),
                "latitude": np.full(len(merged), lat0, dtype=float),
                "longitude": np.full(len(merged), lon0, dtype=float),
                "altitude": np.full(len(merged), alt0, dtype=float),
                target_col: filled,
                "source": source,
            }
        )
        out_blocks.append(out)

        if show_progress:
            n_obs = int((out["source"] == "observed").sum())
            n_imp = int((out["source"] == "imputed").sum())
            k_text: Union[int, str]
            if nmap is None:
                k_text = "all"
            else:
                k_text = len(nmap.get(sid, []))
            pct_text = (
                100 if include_target_pct is None else include_target_pct
            )
            tqdm.write(
                f"Station {sid}: window={len(out):,}  observed={n_obs:,}  "
                f"imputed={n_imp:,}  k={k_text}, include_target_pct={pct_text}"
            )

    # ----------------------------------------------------------------------
    # 8) Stack, sort, save, return
    # ----------------------------------------------------------------------
    result = pd.concat(out_blocks, axis=0, ignore_index=True)
    result = result.sort_values([id_col, "date"], kind="mergesort").reset_index(drop=True)

    # Ensure exact column order
    result = result[[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]]

    _save_result_df(
        result,
        path=save_path,
        fmt=save_format,
        index=save_index,
        partition=save_partitions,
        station_col=id_col,
    )

    return result


__all__ = [
    "impute_dataset",
]
