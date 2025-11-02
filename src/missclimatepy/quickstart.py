# src/missclimatepy/quickstart.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

import os
import pandas as pd

from .prepare import (
    enforce_schema,
    filter_period,
    missing_summary,
    select_stations,
)
from .neighbors import neighbor_distances
from .evaluate import evaluate_per_station
from .api import MissClimateImputer

# Optional plotting imports (kept lazy to avoid CI/runtime hard deps)
try:
    from .viz import plot_station_series, plot_metrics_distribution, plot_inclusion_aggregate  # type: ignore
except Exception:  # pragma: no cover
    plot_station_series = plot_metrics_distribution = plot_inclusion_aggregate = None  # type: ignore


@dataclass
class QuickstartConfig:
    # Required
    data_path: str
    target: str

    # Typical inputs
    period: Tuple[str, str] = ("1991-01-01", "2020-12-31")
    stations: Optional[List[str]] = None
    min_obs: int = 60

    # <<< Important for spatial context >>>
    k_neighbors: int = 20  # <--- ADDED: number of spatial neighbors used

    # RF defaults (you can also pass a custom rf_params dict to MissClimateImputer)
    n_estimators: int = 100
    n_jobs: int = -1
    random_state: int = 42

    # Outputs & evaluation
    outputs_dir: str = "."
    do_metrics: bool = True
    train_frac: float = 0.3  # 0.0 = strict LOSO (no local training data)
    do_mdr: bool = False
    mdr_grid: Optional[List[float]] = None  # e.g., [0.04, 0.1, 0.2, 0.4, 0.6, 0.8]
    verbose: bool = True

    # Visualization
    plot_sample_stations: int = 0  # how many stations to plot (0 = none)
    plots_dir: str = "plots"
    title_tag: str = ""


def run_quickstart(**kwargs) -> Dict[str, Any]:
    """
    High-level, batteries-included entry-point for quick experiments on Kaggle/Colab.

    Parameters (kwargs):
        data_path: CSV or Parquet file with the schema expected by `enforce_schema`.
        target: column name to impute/predict (e.g., "tmin", "tmax", "prec", "evap").
        period: (start, end) inclusive strings "YYYY-MM-DD".
        stations: list of station ids to filter (or None for all).
        min_obs: minimum valid rows per-station for inclusion.
        k_neighbors: number of spatial neighbors used by the local predictor.  <-- IMPORTANT
        n_estimators: RF trees (if using the default RF engine).
        n_jobs, random_state: standard parallel/seed controls.
        outputs_dir: directory for all artifacts (parquets, csvs, plots).
        do_metrics: compute per-station metrics via masked validation.
        train_frac: fraction of valid rows kept for training (the rest are masked for validation).
                    0.0 is strict LOSO (no station self-data included).
        do_mdr/mdr_grid: optional MDR sweep.
        plot_sample_stations/plots_dir/title_tag: plotting controls.

    Returns:
        dict summary with core counts; side effects: writes artifacts to outputs_dir.
    """
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)


def _load_dataframe(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pd.read_parquet(path)
    # default to CSV
    return pd.read_csv(path)


def _run(cfg: QuickstartConfig) -> Dict[str, Any]:
    os.makedirs(cfg.outputs_dir, exist_ok=True)

    # 1) Load & normalize schema
    if cfg.verbose:
        print(f"Loading: {cfg.data_path}")
    df = _load_dataframe(cfg.data_path)
    df = enforce_schema(df, target=cfg.target)

    # 2) Period filter
    df = filter_period(df, start=cfg.period[0], end=cfg.period[1])
    if cfg.verbose:
        n_st = df["station"].nunique()
        print(f"Rows after period filter: {len(df):,} | stations: {n_st:,}")

    # 3) Missing summary (always helpful)
    miss_path = os.path.join(
        cfg.outputs_dir,
        f"missing_summary_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv",
    )
    missing_summary(df, target=cfg.target).to_csv(miss_path, index=False)
    if cfg.verbose:
        print(f"Missing summary -> {miss_path}")

    # 4) Station selection
    df_sel = select_stations(
        df, target=cfg.target, stations=cfg.stations, min_obs=cfg.min_obs
    )
    if cfg.verbose:
        print(
            f"Selected stations: {df_sel['station'].nunique():,} | rows: {len(df_sel):,}"
        )

    # 5) Fit + impute entire selection with the RF engine (spatio-temporal local)
    if cfg.verbose:
        print("Imputing with RF ...")
    imp = MissClimateImputer(
        engine="rf",
        target=cfg.target,
        k_neighbors=cfg.k_neighbors,        # <--- propagate k
        min_obs_per_station=max(0, int(cfg.min_obs * cfg.train_frac)),
        n_jobs=cfg.n_jobs if 'cfg' in globals() else cfg.n_jobs,  # safety if someone copy-pastes
        random_state=cfg.random_state,
        rf_params={"n_estimators": cfg.n_estimators},
    )
    df_imp = imp.fit_transform(df_sel)

    # Save imputed parquet
    imp_path = os.path.join(cfg.outputs_dir, "imputed.parquet")
    df_imp.to_parquet(imp_path, index=False)
    if cfg.verbose:
        print(f"Imputed parquet -> {imp_path}")

    # Basic report
    report = imp.report(df_imp)
    if cfg.verbose:
        print(f"Report: {report}")

    # 6) Neighbors distances snapshot (useful for diagnosing spatial context)
    neigh_path = os.path.join(
        cfg.outputs_dir, f"neighbor_distances_k{cfg.k_neighbors}.csv"
    )
    neighbor_distances(
        df_sel[["station", "latitude", "longitude", "elevation"]].drop_duplicates(),
        k_neighbors=cfg.k_neighbors,  # <--- propagate k
    ).to_csv(neigh_path, index=False)
    if cfg.verbose:
        print(f"Neighbor distances -> {neigh_path}")

    # 7) Optional evaluation (masked)
    if cfg.do_metrics:
        if cfg.verbose:
            print(f"Evaluating per-station with train_frac={cfg.train_frac} ...")
        metrics_df = evaluate_per_station(
            df_src=df_sel,
            target=cfg.target,
            train_frac=cfg.train_frac,
            min_obs=cfg.min_obs,
            engine="rf",
            model_params={
                "k_neighbors": cfg.k_neighbors,       # <--- propagate k
                "n_estimators": cfg.n_estimators,
                "n_jobs": cfg.n_jobs,
                "random_state": cfg.random_state,
            },
        )
        met_path = os.path.join(cfg.outputs_dir, "metrics_per_station.csv")
        metrics_df.to_csv(met_path, index=False)
        if cfg.verbose:
            print(f"Per-station metrics -> {met_path}")

        # optional plots if viz is available
        if plot_metrics_distribution is not None and cfg.plot_sample_stations >= 0:
            try:
                os.makedirs(os.path.join(cfg.outputs_dir, cfg.plots_dir), exist_ok=True)
                plot_metrics_distribution(
                    metrics_df,
                    out_dir=os.path.join(cfg.outputs_dir, cfg.plots_dir),
                    title_suffix=cfg.title_tag,
                )
                if cfg.plot_sample_stations > 0 and plot_station_series is not None:
                    # plot a few random stations
                    for st in metrics_df["station"].head(cfg.plot_sample_stations):
                        plot_station_series(
                            original=df_sel[df_sel["station"] == st],
                            imputed=df_imp[df_imp["station"] == st],
                            target=cfg.target,
                            out_dir=os.path.join(cfg.outputs_dir, cfg.plots_dir),
                            title_suffix=f"{cfg.title_tag} [st={st}]",
                        )
            except Exception:
                pass  # plotting is best-effort

    # 8) Optional MDR sweep (kept minimal: users can call mdr_grid_search directly)
    if cfg.do_mdr and cfg.mdr_grid:
        try:
            from .mdr import mdr_grid_search  # lazy import
            mdr_res = mdr_grid_search(
                df_src=df_sel,
                target=cfg.target,
                include_grid=cfg.mdr_grid,
                min_obs=cfg.min_obs,
                k_neighbors=cfg.k_neighbors,  # <--- propagate k
                n_estimators=cfg.n_estimators,
                n_jobs=cfg.n_jobs,
                random_state=cfg.random_state,
                verbose=cfg.verbose,
            )
            mdr_path = os.path.join(cfg.outputs_dir, "mdr_grid_results.csv")
            mdr_res.to_csv(mdr_path, index=False)
            if cfg.verbose:
                print(f"MDR sweep -> {mdr_path}")

            if plot_inclusion_aggregate is not None:
                try:
                    plot_inclusion_aggregate(
                        mdr_res,
                        out_dir=os.path.join(cfg.outputs_dir, cfg.plots_dir),
                        title_suffix=cfg.title_tag,
                    )
                except Exception:
                    pass
        except Exception:
            # MDR is optional; if not available, skip silently
            pass

    # return a compact json-serializable report
    return {
        **report,
        "k_neighbors": cfg.k_neighbors,
        "outputs_dir": cfg.outputs_dir,
    }
