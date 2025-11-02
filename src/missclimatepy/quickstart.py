# src/missclimatepy/quickstart.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict, Any

import os
import pandas as pd

from .prepare import (
    enforce_schema,
    filter_period,
    missing_summary,
    select_stations,
)
from .api import MissClimateImputer
from .evaluate import evaluate_per_station
from .viz import plot_station_series, plot_metrics_distribution


# ------------------------------- Utilities --------------------------------- #
def _load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a dataframe from CSV/Parquet. This function is intentionally small
    because schema enforcement is handled later by `enforce_schema`.

    Notes
    -----
    * CSV is read with `low_memory=False` to avoid dtype guessing issues.
    * Parquet is preferred for speed/memory when available.
    """
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    # default: CSV (including .csv.gz)
    return pd.read_csv(path, low_memory=False)


# ------------------------------ Configuration ------------------------------ #
@dataclass
class QuickstartConfig:
    """
    High-level configuration for a one-call run.

    Parameters
    ----------
    data_path : str
        Path to the input dataset (CSV or Parquet).
    target : str
        Target variable (e.g., 'tmin', 'tmax', 'prec', 'evap').
    period : (str, str)
        Inclusive date range in 'YYYY-MM-DD' format.
    stations : Iterable[str], optional
        Subset of stations to *evaluate/plot*. The model is trained/imputed
        using the whole filtered network; this field only restricts evaluation.
        If None, all eligible stations are evaluated.
    min_obs : int
        Minimum valid observations of `target` per station to be included in
        the training universe (and to be eligible for evaluation).
    k_neighbors : int
        Number of spatial neighbors provided to the local model.
    n_estimators : int
        Shortcut for Random Forest when `engine='rf'`.
    model_params : dict, optional
        Extra hyperparameters forwarded to the underlying model builder.
        For RF you can pass {'n_estimators': 200, 'max_depth': 25,
        'min_samples_leaf': 2}, etc. When provided, it overrides the shortcut.
    n_jobs : int
        Parallelism used by the learner (if supported).
    random_state : int
        Random seed for masking and stochastic learners.
    outputs_dir : str
        Directory where artifacts (parquet/csv/plots) will be written.
    do_metrics : bool
        If True, run per-station evaluation.
    train_frac : float
        Fraction of observed rows in the *evaluated* station kept for training
        (0.0 = strict LOSO, 1.0 = no simulated missing). Other stations are
        never masked—so the model always trains on the full network.
    do_mdr : bool
        Reserved flag for future multi-density-ratio sweeps (not used here).
    mdr_grid : Iterable[float], optional
        Reserved grid for MDR sweeps (not used here).
    verbose : bool
        Print progress information.
    plot_sample_stations : int
        If >0, plot up to N evaluated stations.
    plots_dir : str
        Subdirectory (inside outputs_dir) for plot images.
    title_tag : str
        Optional suffix appended to plot titles.
    """

    data_path: str
    target: str
    period: Tuple[str, str]

    stations: Optional[Iterable[str]] = None  # evaluate/plot subset
    min_obs: int = 60
    k_neighbors: int = 20

    # RF shortcut (common case)
    n_estimators: int = 200

    model_params: Optional[Dict[str, Any]] = None
    n_jobs: int = -1
    random_state: int = 42

    # outputs
    outputs_dir: str = "./outputs"
    do_metrics: bool = True
    train_frac: float = 0.7
    do_mdr: bool = False
    mdr_grid: Optional[Iterable[float]] = None
    verbose: bool = True
    plot_sample_stations: int = 0
    plots_dir: str = "plots"
    title_tag: str = ""


# ------------------------------- Orchestrator ------------------------------- #
def run_quickstart(**kwargs) -> dict:
    """
    Convenience wrapper: build `QuickstartConfig` from kwargs and execute.
    """
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)


def _run(cfg: QuickstartConfig) -> dict:
    # 1) Load & enforce schema
    if cfg.verbose:
        print(f"Loading: {cfg.data_path}")
    df = _load_dataframe(cfg.data_path)
    df = enforce_schema(df, target=cfg.target)

    # 2) Period filter
    df = filter_period(df, cfg.period)
    if cfg.verbose:
        print(f"Rows after period filter: {len(df):,} | stations: {df['station'].nunique():,}")

    # 3) Outputs directory
    os.makedirs(cfg.outputs_dir, exist_ok=True)

    # 4) Missing summary (diagnostics)
    miss_path = os.path.join(
        cfg.outputs_dir,
        f"missing_summary_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv",
    )
    missing_summary(df, cfg.target).to_csv(miss_path, index=False)
    if cfg.verbose:
        print(f"Missing summary -> {miss_path}")

    # 5) Select stations with at least `min_obs` target observations
    df_sel = select_stations(df, target=cfg.target, min_obs=cfg.min_obs)
    if cfg.verbose:
        print(
            f"Training universe: rows={len(df_sel):,} | "
            f"stations={df_sel['station'].nunique():,}"
        )

    # 6) Fit + impute using the entire network (global training, local neighbors)
    rf_params = cfg.model_params or {"n_estimators": cfg.n_estimators}
    imp = MissClimateImputer(
        engine="rf",
        target=cfg.target,
        k_neighbors=cfg.k_neighbors,
        min_obs_per_station=0,  # allow strict LOSO for evaluated station
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        model_params=rf_params,
    )
    if cfg.verbose:
        print("Imputing with RF (global training, local neighbors) ...")
    df_imp = imp.fit_transform(df_sel)

    # 7) Save imputed dataset
    imp_path = os.path.join(cfg.outputs_dir, "imputed.parquet")
    df_imp.to_parquet(imp_path, index=False)

    report = {
        "target": cfg.target,
        "rows": int(len(df_imp)),
        "stations": int(df_imp["station"].nunique()),
        "missing_after": int(df_imp[cfg.target].isna().sum()),
        "missing_rate_after": float(df_imp[cfg.target].isna().mean()),
    }
    if cfg.verbose:
        print(f"Imputed parquet -> {imp_path}")
        print(f"Report: {report}")

    # 8) Per-station evaluation ONLY for requested stations (or all)
    if cfg.do_metrics:
        eval_ids = cfg.stations  # None => evaluate all eligible stations
        if cfg.verbose:
            print(f"Evaluating per-station with train_frac={cfg.train_frac} ...")

        metrics_df = evaluate_per_station(
            df_full=df_sel,
            target=cfg.target,
            station_ids=eval_ids,
            train_frac=cfg.train_frac,
            min_obs=cfg.min_obs,
            engine="rf",
            k_neighbors=cfg.k_neighbors,
            model_params=rf_params,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        metrics_path = os.path.join(cfg.outputs_dir, "metrics_per_station.csv")
        metrics_df.to_csv(metrics_path, index=False)
        if cfg.verbose:
            print(f"Metrics per-station -> {metrics_path}")

        # Optional plots
        if cfg.plot_sample_stations and not metrics_df.empty:
            plot_dir = os.path.join(cfg.outputs_dir, cfg.plots_dir)
            sample = metrics_df["station"].head(cfg.plot_sample_stations).tolist()

            plot_station_series(
                df_raw=df_sel,
                df_imp=df_imp,
                target=cfg.target,
                station_ids=sample,
                out_dir=plot_dir,
                title_suffix=cfg.title_tag,
            )
            plot_metrics_distribution(
                metrics_df,
                out_dir=plot_dir,
                title_suffix=cfg.title_tag,
            )

    return report

    if cfg.verbose:
        print(f"Loading: {cfg.data_path}")
    df = _load_dataframe(cfg.data_path)
    df = enforce_schema(df, target=cfg.target)

    # 1) Period filter
    df = filter_period(df, cfg.period)
    if cfg.verbose:
        print(f"Rows after period filter: {len(df):,} | stations: {df['station'].nunique():,}")

    os.makedirs(cfg.outputs_dir, exist_ok=True)

    # 2) Missing summary (for diagnostics)
    miss_path = os.path.join(
        cfg.outputs_dir, f"missing_summary_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv"
    )
    missing_summary(df, cfg.target).to_csv(miss_path, index=False)
    if cfg.verbose:
        print(f"Missing summary -> {miss_path}")

    # 3) Select stations that at least meet min_obs for the target
    df_sel = select_stations(df, target=cfg.target, min_obs=cfg.min_obs)
    if cfg.verbose:
        print(f"Training universe: rows={len(df_sel):,} | stations={df_sel['station'].nunique():,}")

    # 4) Fit+impute with the entire network
    imp = MissClimateImputer(
        engine="rf",
        target=cfg.target,
        k_neighbors=cfg.k_neighbors,
        min_obs_per_station=0,
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        model_params=cfg.model_params or {"n_estimators": cfg.n_estimators},
    )
    if cfg.verbose:
        print("Imputing with RF (global training, local neighbors) ...")
    df_imp = imp.fit_transform(df_sel)

    # 5) Save imputed dataframe (parquet to be memory-friendly)
    imp_path = os.path.join(cfg.outputs_dir, "imputed.parquet")
    df_imp.to_parquet(imp_path, index=False)
    report = {
        "target": cfg.target,
        "rows": int(len(df_imp)),
        "stations": int(df_imp["station"].nunique()),
        "missing_after": int(df_imp[cfg.target].isna().sum()),
        "missing_rate_after": float(df_imp[cfg.target].isna().mean()),
    }
    if cfg.verbose:
        print(f"Imputed parquet -> {imp_path}")
        print(f"Report: {report}")

    # 6) Per-station evaluation ONLY for the user-selected stations (or all)
    if cfg.do_metrics:
        eval_ids = cfg.stations  # None means evaluate all
        if cfg.verbose:
            print(f"Evaluating per-station with train_frac={cfg.train_frac} ...")
        metrics_df = evaluate_per_station(
            df_full=df_sel,
            target=cfg.target,
            station_ids=eval_ids,
            train_frac=cfg.train_frac,
            min_obs=cfg.min_obs,
            engine="rf",
            k_neighbors=cfg.k_neighbors,
            model_params=cfg.model_params or {"n_estimators": cfg.n_estimators},
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        metrics_path = os.path.join(cfg.outputs_dir, "metrics_per_station.csv")
        metrics_df.to_csv(metrics_path, index=False)
        if cfg.verbose:
            print(f"Metrics per-station -> {metrics_path}")

        # Optional plots
        if cfg.plot_sample_stations and not metrics_df.empty:
            sample = metrics_df["station"].head(cfg.plot_sample_stations).tolist()
            plot_station_series(
                df_raw=df_sel, df_imp=df_imp, target=cfg.target,
                station_ids=sample,
                out_dir=os.path.join(cfg.outputs_dir, cfg.plots_dir),
                title_suffix=cfg.title_tag,
            )
            plot_metrics_distribution(
                metrics_df, out_dir=os.path.join(cfg.outputs_dir, cfg.plots_dir),
                title_suffix=cfg.title_tag,
            )

    return report