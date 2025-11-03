# src/missclimatepy/quickstart.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any, Union
import pandas as pd
import numpy as np
import os

from .prepare import (
    enforce_schema, filter_period, missing_summary, select_stations as _select_stations
)
from .evaluate import evaluate_per_station
from .neighbors import neighbor_distances  # usado para construir mapa de vecinos
from .api import MissClimateImputer  # sólo para tipos/compat

# ------------------------- Config -------------------------

@dataclass
class QuickstartConfig:
    data_path: str
    target: str = "tmin"
    period: Tuple[str, str] = ("1991-01-01", "2020-12-31")

    # IMPORTANTE: ahora `stations` **sólo** controla qué estaciones evaluar.
    # El entrenamiento usa todo el universo filtrado por periodo / min_obs.
    stations: Optional[Iterable[Union[str, int]]] = None

    min_obs: int = 60

    # Modelo / features
    engine: str = "rf"
    k_neighbors: Optional[int] = None  # vecinos para imputación local (eval)
    n_estimators: int = 200
    n_jobs: int = -1
    random_state: int = 42
    rf_params: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None

    # Salidas
    outputs_dir: str = "./outputs"
    do_metrics: bool = True
    train_frac: float = 0.3   # 0.0 = LOSO estricto; 0.8 = enmascarado “duro”
    do_mdr: bool = False
    verbose: bool = True

    # Plots (muestra)
    plot_sample_stations: int = 3
    plots_dir: str = "plots"
    title_tag: str = ""

def _load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    elif ext == ".feather":
        return pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# ------------------------- Run -------------------------

def _run(cfg: QuickstartConfig) -> Dict[str, Any]:
    # 1) Load & schema
    if cfg.verbose:
        print(f"Loading: {cfg.data_path}")
    df = _load_dataframe(cfg.data_path)
    df = enforce_schema(df, target=cfg.target)

    # 2) Period
    df = filter_period(df, *cfg.period)
    if cfg.verbose:
        print(f"Rows after period filter: {len(df):,} | stations: {df['station'].nunique():,}")

    # 3) Resumen de faltantes (solo informativo)
    os.makedirs(cfg.outputs_dir, exist_ok=True)
    summary_path = os.path.join(cfg.outputs_dir, f"missing_summary_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv")
    missing_summary(df, target=cfg.target).to_csv(summary_path, index=False)
    if cfg.verbose:
        print(f"Missing summary -> {summary_path}")

    # 4) Universo de entrenamiento = TODO el df filtrado
    #    Estaciones a evaluar = lista explícita o todas.
    all_stations: List[str] = sorted(df["station"].astype(str).unique().tolist())
    if cfg.stations is None:
        eval_stations = all_stations
    else:
        eval_stations = [str(s) for s in cfg.stations]
    if cfg.verbose:
        print(f"Training universe: rows={len(df):,} | stations={len(all_stations):,}")
        if cfg.stations is not None:
            print(f"Evaluating only {len(eval_stations)} station(s).")

    # 5) Mapa de vecinos (si aplica)
    neighbor_map: Optional[Dict[str, List[str]]] = None
    if cfg.k_neighbors is not None and cfg.k_neighbors > 0:
        # neighbor_distances devuelve distancias; aquí nos quedamos con los ids
        neigh = neighbor_distances(
            df[["station", "latitude", "longitude"]].drop_duplicates("station"),
            k=cfg.k_neighbors + 1  # incluye a sí misma en posición 0
        )
        neighbor_map = {}
        for st, rows in neigh.groupby("station"):
            ids = rows.sort_values("distance_km")["neighbor"].astype(str).tolist()
            neighbor_map[str(st)] = [i for i in ids if i != str(st)][:cfg.k_neighbors]

    # 6) Evaluación por estación (enmascarado controlado)
    metrics_df = None
    if cfg.do_metrics:
        if cfg.verbose:
            print(f"Imputing with {cfg.engine} (global training, local neighbors) ...")
        metrics_df = evaluate_per_station(
            df_src=df,
            target=cfg.target,
            train_frac=cfg.train_frac,
            min_obs=cfg.min_obs,
            engine=cfg.engine,
            k_neighbors=cfg.k_neighbors,
            model_params=(cfg.model_params or dict(rf_params=cfg.rf_params or dict(n_estimators=cfg.n_estimators))),
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            eval_stations=eval_stations,           # <-- sólo evaluamos estas
            neighbor_map=neighbor_map              # <-- reutilizamos el mapa
        )
        out_path = os.path.join(cfg.outputs_dir, f"metrics_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv")
        metrics_df.to_csv(out_path, index=False)
        if cfg.verbose:
            print(f"Metrics -> {out_path}")

    # 7) Muestra de gráficas (opcional, no intrusivo)
    # (se deja para un módulo de viz; aquí sólo devolvemos el reporte)

    report = {
        "n_rows": len(df),
        "n_stations": len(all_stations),
        "n_eval_stations": len(eval_stations),
        "outputs_dir": cfg.outputs_dir,
        "target": cfg.target,
        "period": cfg.period,
        "engine": cfg.engine,
        "k_neighbors": cfg.k_neighbors,
        "metrics_head": None if metrics_df is None else metrics_df.head().to_dict(orient="list"),
    }
    return report


def run_quickstart(**kwargs) -> Dict[str, Any]:
    """
    Build `QuickstartConfig` from kwargs and execute.
    Key behavior: `stations` limits *evaluation only*; training uses the full
    filtered universe.
    """
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)
