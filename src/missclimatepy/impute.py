# SPDX-License-Identifier: MIT
"""
missclimatepy.impute
====================

XYZT-based imputation of daily climate station records.

Public API
----------

- :func:`impute_dataset`: fill missing values (and missing *dates*) in a
  long-format daily table, station by station.

Design
------

* Features: latitude, longitude, altitude + calendar fields (year, month, doy)
  and optional cyclic transforms of day-of-year.
* Model: any regressor created by :func:`missclimatepy.models.make_model`
  (Random Forest by default).
* Training: for each **target station** we train one model using:
    - all *other* stations, or
    - only its K nearest neighbors (if a neighbor map is provided or
      ``k_neighbors`` is given),
    - plus an optional fraction of the target station's own observed rows
      (``include_target_pct``).
* Imputation: predictions are generated no sólo para filas con el target NaN,
  sino también para **fechas que no existían** en el dataframe original:
  primero se construye una malla diaria completa en el intervalo de análisis.
* Output: tabla en formato largo que contiene **todos los días** en el periodo
  solicitado, con:

    - observaciones originales intactas,
    - huecos llenados cuando es posible,
    - una columna ``source`` con valores ``"observed"`` o ``"imputed"``.

Las primeras columnas del resultado son siempre:

    [id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .features import (
    add_calendar_features,
    ensure_datetime_naive,
    select_station_ids,
    validate_required_columns,
)
from .models import make_model
from .neighbors import build_neighbor_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_features(
    *,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    add_cyclic: bool,
) -> List[str]:
    """Default list of feature columns for XYZT regression."""
    feats = [lat_col, lon_col, alt_col, "year", "month", "doy"]
    if add_cyclic:
        feats += ["doy_sin", "doy_cos"]
    return feats


def _empty_output(
    *,
    id_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Empty result with canonical columns."""
    cols = [id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Public imputer
# ---------------------------------------------------------------------------


def impute_dataset(
    data: pd.DataFrame,
    *,
    # core schema
    id_col: str,
    date_col: str,
    lat_col: str,
    lon_col: str,
    alt_col: str,
    target_col: str,
    # time window
    start: Optional[str] = None,
    end: Optional[str] = None,
    # feature configuration
    add_cyclic: bool = False,
    feature_cols: Optional[Sequence[str]] = None,
    # station selection (OR semantics; default = all stations in `data`)
    prefix: Optional[Iterable[str]] = None,
    station_ids: Optional[Iterable[Hashable]] = None,
    regex: Optional[str] = None,
    custom_filter: Optional[Any] = None,
    # minimum information per target station (observed rows)
    min_station_rows: Optional[int] = None,
    # neighborhood & leakage
    k_neighbors: Optional[int] = 20,
    neighbor_map: Optional[Mapping[Hashable, Sequence[Hashable]]] = None,
    include_target_pct: float = 0.0,
    include_target_seed: int = 42,
    # model
    model_kind: str = "rf",
    model_params: Optional[Dict[str, Any]] = None,
    # UX
    show_progress: bool = False,
) -> pd.DataFrame:
    """
    Impute missing values (and missing **dates**) in a long-format daily table.

    Para cada estación seleccionada:

    1. Construye una serie diaria completa en el intervalo de análisis
       (ya sea [start, end] o el rango mínimo–máximo en `data`).
    2. Asigna sus coordenadas (lat, lon, alt) como la mediana por estación.
    3. Entrena un modelo XYZT usando vecinos / todas las demás estaciones y
       una fracción opcional de filas observadas de la estación objetivo.
    4. Predice el target en todas las filas donde `target_col` está ausente.
    5. Devuelve la serie diaria completa con la columna ``source`` indicando
       si el valor es ``"observed"`` o ``"imputed"``.

    Las estaciones que no alcanzan `min_station_rows` se **devuelven tal cual**:
    sus huecos siguen como NaN y `source="missing"`.
    """
    # ------------------------------------------------------------------
    # 1) Validación básica y ventana temporal
    # ------------------------------------------------------------------
    validate_required_columns(
        data,
        [id_col, date_col, lat_col, lon_col, alt_col, target_col],
        context="impute_dataset",
    )

    df = data.copy()
    df[date_col] = ensure_datetime_naive(df[date_col])

    if df.empty:
        return _empty_output(
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            alt_col=alt_col,
            date_col=date_col,
            target_col=target_col,
        )

    global_lo = df[date_col].min()
    global_hi = df[date_col].max()
    start_dt = pd.to_datetime(start) if start is not None else global_lo
    end_dt = pd.to_datetime(end) if end is not None else global_hi

    df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
    if df.empty:
        return _empty_output(
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            alt_col=alt_col,
            date_col=date_col,
            target_col=target_col,
        )

    # ------------------------------------------------------------------
    # 2) Estaciones objetivo (para COBERTURA) y estaciones a imputar
    # ------------------------------------------------------------------
    # Estaciones presentes en el dataframe
    all_station_ids = df[id_col].dropna().unique().tolist()

    # Estaciones que el usuario quiere considerar (para cobertura + salida)
    stations_eval = select_station_ids(
        df,
        id_col=id_col,
        prefix=prefix,
        station_ids=station_ids,
        regex=regex,
        custom_filter=custom_filter,
    )
    # Asegurar que están en el dataset
    station_set = set(all_station_ids)
    stations_eval = [s for s in stations_eval if s in station_set]

    if not stations_eval:
        return _empty_output(
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            alt_col=alt_col,
            date_col=date_col,
            target_col=target_col,
        )

    # Subconjunto que SÍ se intentará imputar (aplica min_station_rows)
    if min_station_rows is not None:
        obs_counts = (
            df.loc[~df[target_col].isna(), [id_col, target_col]]
            .groupby(id_col)[target_col]
            .size()
            .astype(int)
        )
        stations_impute = [
            s for s in stations_eval if int(obs_counts.get(s, 0)) >= int(min_station_rows)
        ]
    else:
        stations_impute = list(stations_eval)

    # ------------------------------------------------------------------
    # 3) Expandir cada estación evaluada a una malla diaria completa
    # ------------------------------------------------------------------
    daily_index = pd.date_range(start_dt, end_dt, freq="D")

    expanded_targets: List[pd.DataFrame] = []
    for sid in stations_eval:
        sub = df[df[id_col] == sid].copy()
        if sub.empty:
            # Aun si no hay registros, creamos la grilla diaria con coords NaN
            base = pd.DataFrame(
                {
                    id_col: sid,
                    date_col: daily_index,
                    lat_col: np.nan,
                    lon_col: np.nan,
                    alt_col: np.nan,
                }
            )
            expanded_targets.append(base)
            continue

        lat_med = float(sub[lat_col].median()) if sub[lat_col].notna().any() else np.nan
        lon_med = float(sub[lon_col].median()) if sub[lon_col].notna().any() else np.nan
        alt_med = float(sub[alt_col].median()) if sub[alt_col].notna().any() else np.nan

        base = pd.DataFrame(
            {
                id_col: sid,
                date_col: daily_index,
                lat_col: lat_med,
                lon_col: lon_med,
                alt_col: alt_med,
            }
        )

        other_cols = [
            c
            for c in sub.columns
            if c not in (id_col, date_col, lat_col, lon_col, alt_col)
        ]
        if other_cols:
            base = base.merge(
                sub[[date_col] + other_cols],
                on=date_col,
                how="left",
            )

        expanded_targets.append(base)

    df_targets = pd.concat(expanded_targets, ignore_index=True)

    # Otras estaciones (no seleccionadas para cobertura) pueden seguir
    # con su span nativo y sirven como pool de entrenamiento.
    df_others = df[~df[id_col].isin(stations_eval)].copy()

    if df_others.empty:
        df_all = df_targets.copy()
    else:
        df_all = pd.concat([df_targets, df_others], ignore_index=True)

    # ------------------------------------------------------------------
    # 4) Añadir features temporales y decidir lista de features
    # ------------------------------------------------------------------
    df_all = add_calendar_features(df_all, date_col=date_col, add_cyclic=add_cyclic)

    if feature_cols is None:
        feats = _default_features(
            lat_col=lat_col, lon_col=lon_col, alt_col=alt_col, add_cyclic=add_cyclic
        )
    else:
        feats = list(feature_cols)

    # ------------------------------------------------------------------
    # 5) Mapa de vecinos
    # ------------------------------------------------------------------
    if neighbor_map is not None:
        # Copia defensiva
        nmap: Dict[Hashable, List[Hashable]] = {
            sid: list(nei) for sid, nei in neighbor_map.items()
        }
    elif k_neighbors is not None:
        nmap = build_neighbor_map(
            df_all,
            id_col=id_col,
            lat_col=lat_col,
            lon_col=lon_col,
            k=int(k_neighbors),
            include_self=False,
        )
    else:
        nmap = {}

    # ------------------------------------------------------------------
    # 6) Máscaras globales (sobre df_all)
    # ------------------------------------------------------------------
    feat_ok = ~df_all[feats].isna().any(axis=1)
    target = df_all[target_col]
    target_na = target.isna()

    train_base_mask = feat_ok & (~target_na)  # filas observadas
    missing_base_mask = feat_ok & target_na   # filas a imputar

    idx_arr = df_all.index.to_numpy()
    station_arr = df_all[id_col].to_numpy()

    # Salida: sólo estaciones evaluadas, con source inicial
    df_out = df_all[df_all[id_col].isin(stations_eval)].copy()
    df_out["source"] = np.where(df_out[target_col].isna(), "missing", "observed")

    rng_global = np.random.RandomState(int(include_target_seed))

    # ------------------------------------------------------------------
    # 7) Bucle de imputación por estación
    #     Sólo para estaciones que pasan `min_station_rows`
    # ------------------------------------------------------------------
    for sid in stations_impute:
        is_station = (station_arr == sid)

        miss_mask_sid = is_station & missing_base_mask
        miss_idx = idx_arr[miss_mask_sid]
        if miss_idx.size == 0:
            if show_progress:
                print(f"[impute] Station {sid}: no rows to impute.")
            continue

        # Pool base: filas observadas de *otras* estaciones
        train_mask_sid = train_base_mask & (~is_station)

        # Restringir por neighbor_map si existe
        if nmap:
            neigh_ids = nmap.get(sid, None)
            if neigh_ids is None:
                # Sin entrada en el mapa -> usamos todas las demás estaciones
                train_mask_sid = train_base_mask & (~is_station)
            else:
                neigh_ids = list(neigh_ids)
                if len(neigh_ids) == 0:
                    # Lista vacía -> SIN pool de entrenamiento (caso de test)
                    train_mask_sid = np.zeros_like(train_mask_sid, dtype=bool)
                else:
                    train_mask_sid &= np.isin(station_arr, np.array(neigh_ids))

        # Inclusión opcional de filas observadas de la propia estación
        if include_target_pct > 0.0:
            pct = float(max(0.0, min(include_target_pct, 100.0)))
            obs_mask_sid = is_station & train_base_mask
            obs_idx = idx_arr[obs_mask_sid]
            if obs_idx.size > 0 and pct > 0.0:
                n_take = int(np.ceil(obs_idx.size * (pct / 100.0)))
                n_take = max(1, min(n_take, obs_idx.size))
                chosen_idx = rng_global.choice(obs_idx, size=n_take, replace=False)
                train_mask_sid |= np.isin(idx_arr, chosen_idx)

        train_idx = idx_arr[train_mask_sid]
        if train_idx.size == 0:
            if show_progress:
                print(f"[impute] Station {sid}: empty training pool, skip.")
            continue

        # Fit & predict
        model = make_model(model_kind=model_kind, model_params=model_params or {})

        X_train = df_all.loc[train_idx, feats].to_numpy()
        y_train = df_all.loc[train_idx, target_col].to_numpy()
        X_miss = df_all.loc[miss_idx, feats].to_numpy()

        model.fit(X_train, y_train)
        y_hat = model.predict(X_miss)

        # Mapear predicciones en df_out para esta estación
        mask_out_sid = (df_out[id_col] == sid) & df_out[target_col].isna()

        # Alinear longitudes por seguridad
        if mask_out_sid.sum() != len(y_hat):
            idx_out_sid = df_out.index[mask_out_sid]
            common_idx = np.intersect1d(idx_out_sid.to_numpy(), miss_idx)
            if common_idx.size == 0:
                continue
            X_miss = df_all.loc[common_idx, feats].to_numpy()
            y_hat = model.predict(X_miss)
            mask_out_sid = df_out.index.isin(common_idx)

        df_out.loc[mask_out_sid, target_col] = y_hat
        df_out.loc[mask_out_sid, "source"] = "imputed"

        if show_progress:
            n_imp = int(mask_out_sid.sum())
            print(
                f"[impute] Station {sid}: "
                f"train={train_idx.size:,}  imputed_rows={n_imp:,}"
            )

    # ------------------------------------------------------------------
    # 8) Reordenar columnas y ordenar por estación/fecha
    # ------------------------------------------------------------------
    leading_cols = [id_col, lat_col, lon_col, alt_col, date_col, target_col, "source"]
    leading_cols = [c for c in leading_cols if c in df_out.columns]
    other_cols = [c for c in df_out.columns if c not in leading_cols]

    df_out = df_out[leading_cols + other_cols]
    df_out = df_out.sort_values([id_col, date_col]).reset_index(drop=True)

    return df_out


__all__ = ["impute_dataset"]
