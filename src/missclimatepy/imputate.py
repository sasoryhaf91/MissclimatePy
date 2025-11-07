# SPDX-License-Identifier: MIT
"""
missclimatepy.imputate
======================

Final imputation of complete time series per station using ONLY spatial
coordinates (latitude, longitude, altitude) and calendar features (year, month,
day-of-year, optionally cyclic sin/cos). One Random-Forest model is trained per
target station using: (a) all other stations or the K nearest neighbors, and
(b) an optional controlled inclusion of a percentage of the target's own valid
rows to adapt locally.

Design notes
------------
- Schema-agnostic: caller passes column names.
- MDR enforcement (per station): a station is processed only if the **count of
  original rows** within [start, end] is >= max(1826, min_station_rows if given).
  This uses the number of rows present (independent of target missingness).
- Outputs: full time series for each station and day in the window, with:
  [station, date, latitude, longitude, altitude, <target_col>, source]
  where `source` ∈ {"observed", "imputed"}.

Typical usage
-------------
>>> from missclimatepy.imputate import impute_dataset
>>> from missclimatepy.evaluate import RFParams
>>> full = impute_dataset(
...     df,
...     id_col="station", date_col="date",
...     lat_col="latitude", lon_col="longitude", alt_col="altitude",
...     target_col="tmin",
...     start="1981-01-01", end="2023-12-31",
...     k_neighbors=20, include_target_pct=80.0,
...     rf_params=RFParams(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42),
...     show_progress=True,
... )

"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluate import (
    RFParams,
    _require_columns,
    _ensure_datetime_naive,
    _add_time_features,
    _build_neighbor_map_haversine,
)

try:
    from tqdm.auto import tqdm  # optional progress bar
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x  # fallback: transparent iterator


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
    # window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # features
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection (optional)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Union[str, int]]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Callable[[Union[str, int]], bool]] = None,
    # neighbors & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Dict[Union[str, int], List[Union[str, int]]]] = None,
    include_target_pct: float = 0.0,     # 0..95
    include_target_seed: int = 42,
    # station MDR
    min_station_rows: Optional[int] = None,
    # model
    rf_params: Optional[Union[RFParams, Dict]] = None,
    # logging
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Impute complete series for each eligible station within [start, end].
    A station is eligible if its **original row count** in the window is
    >= max(1826, min_station_rows if provided), regardless of missingness
    in the target.

    Parameters
    ----------
    min_station_rows : int or None
        Optional caller hint. The function will enforce:
        MDR_MIN = max(1826, min_station_rows or 0).

    Returns
    -------
    DataFrame
        Full imputed series for all stations meeting MDR, with:
        [station, date, latitude, longitude, altitude, <target_col>, source]
        where `source` ∈ {"observed", "imputed"}.
    """
    _require_columns(data, [id_col, date_col, lat_col, lon_col, alt_col, target_col])

    # Normalize datetime and clip window
    df = data.copy()
    df[date_col] = _ensure_datetime_naive(df[date_col])
    df = df.dropna(subset=[date_col])

    # Compute window bounds
    lo = pd.to_datetime(start) if start else df[date_col].min()
    hi = pd.to_datetime(end) if end else df[date_col].max()
    df = df[(df[date_col] >= lo) & (df[date_col] <= hi)]

    # Add calendar features
    df = _add_time_features(df, date_col, add_cyclic=add_cyclic)

    # Assemble default features if none provided
    if feature_cols is None:
        feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
        if add_cyclic:
            feats += ["doy_sin", "doy_cos"]
    else:
        feats = list(feature_cols)

    # Enforce unique feature names to avoid duplicate columns downstream
    feats = list(dict.fromkeys(feats))

    # Determine stations to process (OR filters; default all)
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

    seen = set()
    stations = [s for s in chosen if not (s in seen or seen.add(s))]

    # --- MDR PER-STATION (window-length based on original rows) ---
    # Enforce at least 1826, even if user passes a smaller min_station_rows
    base_mdr = 1826
    user_mdr = int(min_station_rows) if (min_station_rows is not None) else 0
    MDR_MIN = max(base_mdr, user_mdr)

    # Count original rows per station *within the clipped window*
    rows_per_station = df.groupby(id_col, sort=False)[date_col].size().astype(int)
    eligible_stations = [s for s in stations if int(rows_per_station.get(s, 0)) >= MDR_MIN]

    if show_progress and len(eligible_stations) != len(stations):
        diff = len(stations) - len(eligible_stations)
        tqdm.write(f"MDR filter (>= {MDR_MIN} rows in window): {len(stations)} → {len(eligible_stations)} stations (filtered {diff})")

    stations = eligible_stations
    if not stations:
        return pd.DataFrame(columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"])

    # Precompute a neighbor map if requested
    if neighbor_map is not None:
        nmap = neighbor_map
    elif k_neighbors is not None:
        nmap = _build_neighbor_map_haversine(
            df, id_col=id_col, lat_col=lat_col, lon_col=lon_col, k=int(k_neighbors), include_self=False
        )
    else:
        nmap = None

    # RF params
    if rf_params is None:
        rf_kwargs = asdict(RFParams())
    elif isinstance(rf_params, RFParams):
        rf_kwargs = asdict(rf_params)
    else:
        base = asdict(RFParams())
        base.update(dict(rf_params))
        rf_kwargs = base

    # Canonical output date grid (inclusive)
    full_dates = pd.date_range(lo, hi, freq="D")

    # Station medoids (to keep coords stable in output)
    medoids = (
        df.groupby(id_col)[[lat_col, lon_col, alt_col]]
        .median()
        .rename(columns={lat_col: "latitude", lon_col: "longitude", alt_col: "altitude"})
    )

    out_blocks: List[pd.DataFrame] = []
    iterator = tqdm(stations, desc="Imputing stations", unit="st") if show_progress else stations

    # Global valid rows mask for training (neighbors/others)
    valid_mask_global = ~df[feats + [target_col]].isna().any(axis=1)

    for sid in iterator:
        # Slice current station (we only need date + target for merge; features are re-derived)
        st_cols = list(dict.fromkeys([date_col, target_col]))  # ensure unique
        st = df.loc[df[id_col] == sid, st_cols].copy()

        # Build full grid
        base = pd.DataFrame({date_col: full_dates})

        # Medoid coordinates for the station (fallback to NaN if missing)
        lat0 = float(medoids.loc[sid, "latitude"]) if sid in medoids.index else np.nan
        lon0 = float(medoids.loc[sid, "longitude"]) if sid in medoids.index else np.nan
        alt0 = float(medoids.loc[sid, "altitude"]) if sid in medoids.index else np.nan

        # Merge the existing rows (may be sparse) into the full date grid
        merged = base.merge(st, on=date_col, how="left")
        # Attach constant station meta
        merged[id_col] = sid
        merged["latitude"] = lat0
        merged["longitude"] = lon0
        merged["altitude"] = alt0

        # Recompute calendar features on the full grid (fast)
        merged = _add_time_features(merged, date_col, add_cyclic=add_cyclic)

        # Build training pool: neighbors or all-other stations, valid rows only
        is_target_mask = (df[id_col] == sid)
        if nmap is not None:
            neigh_ids = nmap.get(sid, [])
            train_pool_mask = df[id_col].isin(neigh_ids) & (~is_target_mask) & valid_mask_global
        else:
            train_pool_mask = (~is_target_mask) & valid_mask_global

        train_pool = df.loc[train_pool_mask, feats + [target_col]]

        # Controlled inclusion (leakage) from target valid rows
        st_valid = df.loc[is_target_mask & valid_mask_global, feats + [target_col]]
        pct = max(0.0, min(float(include_target_pct), 95.0))
        if pct > 0.0 and not st_valid.empty:
            n_take = int(np.ceil(len(st_valid) * (pct / 100.0)))
            leakage = st_valid.sample(n=n_take, random_state=int(include_target_seed))
            train_df = pd.concat([train_pool, leakage], axis=0, copy=False, ignore_index=True)
        else:
            train_df = train_pool

        if train_df.empty:
            if show_progress:
                tqdm.write(f"Station {sid}: empty training set (skipped)")
            continue

        # Train RF on features
        X_train = train_df[feats].to_numpy(copy=False)
        y_train = train_df[target_col].to_numpy(copy=False)

        model = RandomForestRegressor(**rf_kwargs)
        model.fit(X_train, y_train)

        # Predict for the full grid of the station
        X_full = merged[feats].to_numpy(copy=False)
        y_hat = model.predict(X_full)

        # Compose final series: keep observed where present, fill with predictions otherwise
        y_obs = merged[target_col].to_numpy()
        filled = y_obs.copy()
        mask_nan = np.isnan(filled)
        filled[mask_nan] = y_hat[mask_nan]

        source = np.where(mask_nan, "imputed", "observed")

        out = pd.DataFrame({
            id_col: sid,
            "date": merged[date_col].to_numpy(),
            "latitude": merged["latitude"].astype(float).to_numpy(),
            "longitude": merged["longitude"].astype(float).to_numpy(),
            "altitude": merged["altitude"].astype(float).to_numpy(),
            target_col: filled,
            "source": source,
        })

        out_blocks.append(out)

        if show_progress:
            n_obs = int((source == "observed").sum())
            n_imp = int((source == "imputed").sum())
            k_text = len(nmap.get(sid, [])) if nmap is not None else "all"
            tqdm.write(
                f"Station {sid}: done (window={len(merged)}  observed={n_obs}  imputed={n_imp}  "
                f"k={k_text}  incl={pct:.1f}%)"
            )

    if not out_blocks:
        # Return an empty but well-formed DataFrame if nothing was imputed
        return pd.DataFrame(
            columns=[id_col, "date", "latitude", "longitude", "altitude", target_col, "source"]
        )

    result = pd.concat(out_blocks, axis=0, ignore_index=True)
    result = result.sort_values([id_col, "date"]).reset_index(drop=True)
    return result
