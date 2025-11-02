# src/missclimatepy/quickstart.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict, Any
import os
import pandas as pd

from .prepare import enforce_schema, filter_period, missing_summary, select_stations
from .api import MissClimateImputer
from .evaluate import evaluate_per_station

# ---- Lazy/optional viz imports (so package works without viz.py) ----
_HAS_VIZ = True
try:
    from .viz import plot_station_series, plot_metrics_distribution
except Exception:  # viz not available in this build; define no-ops
    _HAS_VIZ = False

    def plot_station_series(*args, **kwargs):
        return None

    def plot_metrics_distribution(*args, **kwargs):
        return None


# ------------------------------- Utilities --------------------------------- #
def _load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a dataframe from CSV/Parquet. Schema enforcement is handled by
    `enforce_schema` afterwards.
    """
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


# ------------------------------ Configuration ------------------------------ #
@dataclass
class QuickstartConfig:
    """
    High-level configuration for a single-call run.

    The model is **trained/imputed using the whole filtered network**.
    The `stations` argument only restricts **which stations are evaluated/plot**.

    Parameters
    ----------
    data_path : str
    target : str
    period : (str, str)
    stations : Iterable[str], optional
        Only these stations are evaluated/plot. If None, evaluate all eligible.
    min_obs : int
        Minimum valid `target` obs per station to be in the training universe.
    k_neighbors : int
        Number of spatial neighbors for the local model.
    n_estimators : int
        RF shortcut (overridden by `model_params` if provided).
    model_params : dict, optional
        Extra hyperparameters for the underlying model (e.g. RF).
    n_jobs : int
    random_state : int
    outputs_dir : str
    do_metrics : bool
    train_frac : float
        For evaluated stations only: fraction kept for training. 0.0 = LOSO.
    do_mdr, mdr_grid : reserved
    verbose : bool
    plot_sample_stations : int
    plots_dir : str
    title_tag : str
    """

    data_path: str
    target: str
    period: Tuple[str, str]

    stations: Optional[Iterable[str]] = None
    min_obs: int = 60
    k_neighbors: int = 20

    n_estimators: int = 200  # RF shortcut
    model_params: Optional[Dict[str, Any]] = None
    n_jobs: int = -1
    random_state: int = 42

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
    """Build `QuickstartConfig` from kwargs and execute."""
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)


def _run(cfg: QuickstartConfig) -> dict:
    # 1) Load & schema
    if cfg.verbose:
        print(f"Loading: {cfg.data_path}")
    df = _load_dataframe(cfg.data_path)
    df = enforce_schema(df, target=cfg.target)

    # 2) Period filter
    df = filter_period(df, cfg.period)
    if cfg.verbose:
        print(f"Rows after period filter: {len(df):,} | stations: {df['station'].nunique():,}")

    # 3) Outputs dir
    os.makedirs(cfg.outputs_dir, exist_ok=True)

    # 4) Missing summary (diagnostics)
    miss_path = os.path.join(
        cfg.outputs_dir, f"missing_summary_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv"
    )
    missing_summary(df, cfg.target).to_csv(miss_path, index=False)
    if cfg.verbose:
        print(f"Missing summary -> {miss_path}")

    # 5) Select training/evaluation universe
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
        min_obs_per_station=0,  # allow strict LOSO on evaluated station
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        model_params=rf_params,
    )
    if cfg.verbose:
        print("Imputing with RF (global training, local neighbors) ...")
    df_imp = imp.fit_transform(df_sel)

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

    # 7) Per-station evaluation (restricted to cfg.stations if provided)
    if cfg.do_metrics:
        if cfg.verbose:
            print(f"Evaluating per-station with train_frac={cfg.train_frac} ...")

        metrics_df = evaluate_per_station(
            df_full=df_sel,                # full network to train
            target=cfg.target,
            station_ids=cfg.stations,      # only these are evaluated/masked
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

        # 8) Optional plots (only if viz is present)
        if _HAS_VIZ and cfg.plot_sample_stations and not metrics_df.empty:
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
                metrics_df=metrics_df,
                out_dir=plot_dir,
                title_suffix=cfg.title_tag,
            )
        elif cfg.verbose and not _HAS_VIZ:
            print("Plots skipped (viz module not available in this build).")

    return report