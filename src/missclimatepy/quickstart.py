# src/missclimatepy/quickstart.py
# SPDX-License-Identifier: MIT
"""
Quickstart runner
=================

Pequeño *front-end* para evaluar el imputador con una configuración mínima
y columnas genéricas definidas por el usuario.

Puntos clave:
- Columnas genéricas: el usuario indica station/date/lat/lon/alt/target.
- Vecinos: se construyen con Haversine mediante `neighbor_distances`.
- Porcentaje de inclusión de la estación objetivo (0–95% recomendado).
- Llama a `evaluate_all_stations_fast` y devuelve un DataFrame de métricas
  diarias/mensuales/anuales por estación (nos interesan sobre todo las diarias).

Ejemplo rápido
--------------
from missclimatepy.quickstart import run_quickstart, QuickstartConfig
from missclimatepy.evaluate import RFParams

report = run_quickstart(
    data_path="/kaggle/input/data.csv",
    station_col="station", date_col="date",
    lat_col="latitude", lon_col="longitude", alt_col="altitude",
    target="tmin",
    period=("1991-01-01", "2020-12-31"),
    stations=["2038", "2124", "29007"],      # o None para todas
    k_neighbors=20,
    include_target_pct=30.0,
    min_station_rows=9125,
    rf_params=RFParams(n_estimators=15, max_depth=30, random_state=42, n_jobs=-1),
    show_progress=True,
)
print(report.head())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, List

import pandas as pd

from .evaluate import evaluate_all_stations_fast, RFParams
from .neighbors import neighbor_distances
from .io import read_csv


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class QuickstartConfig:
    # IO
    data_path: str

    # Columnas del usuario
    station_col: str = "station"
    date_col: str = "date"
    lat_col: str = "latitude"
    lon_col: str = "longitude"
    alt_col: str = "altitude"

    # Variable objetivo
    target: str = "prec"

    # Periodo a analizar
    period: Tuple[str, str] = ("1991-01-01", "2020-12-31")

    # Subconjunto de estaciones (opcional)
    stations: Optional[Iterable[Union[str, int]]] = None

    # Vecinos e inclusión de objetivo
    k_neighbors: Optional[int] = 20
    include_target_pct: float = 0.0  # 0 = LOSO estricto; 1–95 recomendado

    # Filtro mínimo de filas válidas por estación (opcional)
    min_station_rows: Optional[int] = None

    # Parámetros del Random Forest
    rf_params: Union[RFParams, dict, None] = None

    # Verbosidad
    show_progress: bool = True


# ---------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------
def run_quickstart(**kwargs) -> pd.DataFrame:
    """
    Ejecuta el quickstart construyendo internamente la configuración.

    Parameters
    ----------
    **kwargs : QuickstartConfig fields
        Cualquier campo válido de `QuickstartConfig`.

    Returns
    -------
    pd.DataFrame
        Tabla de métricas por estación.
    """
    cfg = QuickstartConfig(**kwargs)  # type: ignore[arg-type]
    return _run(cfg)


# ---------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------
def _coerce_station_list(stations: Optional[Iterable[Union[str, int]]]) -> Optional[List[str]]:
    if stations is None:
        return None
    out: List[str] = []
    for v in stations:
        out.append(str(v))
    return out


def _clamp_inclusion(p: float) -> float:
    # Forzamos a [0, 95] por seguridad numérica
    p = float(p)
    if p < 0:
        return 0.0
    if p > 95.0:
        return 95.0
    return p


def _run(cfg: QuickstartConfig) -> pd.DataFrame:
    # 1) carga
    df = read_csv(cfg.data_path)

    # 2) casting de fecha
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")
    df = df.dropna(subset=[cfg.date_col])

    # 3) periodo
    lo = pd.to_datetime(cfg.period[0])
    hi = pd.to_datetime(cfg.period[1])
    df = df[(df[cfg.date_col] >= lo) & (df[cfg.date_col] <= hi)]

    # 4) subconjunto de estaciones (opcional)
    station_ids = _coerce_station_list(cfg.stations)
    if station_ids is not None:
        df = df[df[cfg.station_col].astype(str).isin(station_ids)]

    # 5) preparar vecinos (tabla Haversine)
    neighbor_table = None
    if cfg.k_neighbors is not None and cfg.k_neighbors > 0:
        # NOTA: sólo usamos station/lat/lon únicos para el catálogo
        stations_cat = (
            df[[cfg.station_col, cfg.lat_col, cfg.lon_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        neighbor_table = neighbor_distances(
            stations=stations_cat,
            station_col=cfg.station_col,
            lat_col=cfg.lat_col,
            lon_col=cfg.lon_col,
            k_neighbors=int(cfg.k_neighbors),
            include_self=False,
        )

    # 6) RF params
    if cfg.rf_params is None:
        rf_params = RFParams(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    elif isinstance(cfg.rf_params, dict):
        rf_params = RFParams(**cfg.rf_params)  # type: ignore[arg-type]
    else:
        rf_params = cfg.rf_params

    # 7) inclusión objetivo clamp
    include_pct = _clamp_inclusion(cfg.include_target_pct)

    # 8) evaluación (lean memory)
    res = evaluate_all_stations_fast(
        df,
        station_col=cfg.station_col,
        date_col=cfg.date_col,
        lat_col=cfg.lat_col,
        lon_col=cfg.lon_col,
        alt_col=cfg.alt_col,
        target_col=cfg.target,
        station_ids=station_ids,
        start=cfg.period[0],
        end=cfg.period[1],
        # vecinos: pasamos ambos argumentos (k y la tabla)
        k_neighbors=cfg.k_neighbors,
        neighbor_table=neighbor_table,
        # inclusión
        include_target_pct=include_pct,
        # filtros / RF
        min_station_rows=cfg.min_station_rows,
        rf_params=rf_params,
        # logging
        show_progress=cfg.show_progress,
    )
    return res
