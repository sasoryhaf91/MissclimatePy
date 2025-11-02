from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional, List
import pandas as pd

from .io import load_any, save_parquet
from .prepare import enforce_schema, filter_period, missing_summary, select_stations
from .neighbors import neighbor_distances
from .api import MissClimateImputer
from .evaluate import evaluate_per_station
from .mdr import mdr_grid_search

# NUEVO:
from .viz import plot_station_series, plot_metrics_distribution
from tqdm.auto import tqdm

@dataclass
class QuickstartConfig:
    data_path: str
    target: str
    period: tuple[str,str] = ("1991-01-01","2020-12-31")
    stations: List[str] | None = None
    min_obs: int = 60
    k_neighbors: int = 12
    n_estimators: int = 300
    n_jobs: int = -1
    random_state: int = 42
    outputs_dir: str = "./outputs"
    do_metrics: bool = True
    train_frac: float = 0.7
    do_mdr: bool = False
    mdr_grid: dict | None = None

    # NUEVO:
    verbose: bool = True
    plot_sample_stations: int = 10     # cuántas estaciones graficar
    plots_dir: str = "plots"           # subcarpeta dentro de outputs
    title_tag: str = ""                # texto adicional en títulos

def run_quickstart(
    data_path: str, target: str,
    period=("1991-01-01","2020-12-31"),
    stations=None, min_obs=60,
    k_neighbors=12, n_estimators=300, n_jobs=-1, random_state=42,
    outputs_dir="./outputs",
    do_metrics=True, train_frac=0.7,
    do_mdr=False, mdr_grid=None,
    # NUEVO:
    verbose=True, plot_sample_stations=10, plots_dir="plots", title_tag=""
):
    cfg = QuickstartConfig(
        data_path=data_path, target=target, period=period, stations=stations,
        min_obs=min_obs, k_neighbors=k_neighbors, n_estimators=n_estimators,
        n_jobs=n_jobs, random_state=random_state, outputs_dir=outputs_dir,
        do_metrics=do_metrics, train_frac=train_frac, do_mdr=do_mdr,
        mdr_grid=mdr_grid,
        verbose=verbose, plot_sample_stations=plot_sample_stations,
        plots_dir=plots_dir, title_tag=title_tag,
    )
    return _run(cfg)

def _run(cfg: QuickstartConfig):
    outdir = Path(cfg.outputs_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_path = outdir / cfg.plots_dir
    plots_path.mkdir(parents=True, exist_ok=True)

    # 1) load & prepare
    if cfg.verbose: print("Loading:", cfg.data_path)
    df = load_any(cfg.data_path)
    if "altitude" in df.columns and "elevation" not in df.columns:
        df = df.rename(columns={"altitude":"elevation"})

    df = enforce_schema(df, cfg.target)
    df = filter_period(df, cfg.period[0], cfg.period[1])
    if cfg.verbose:
        print(f"Rows after period filter: {len(df):,} | stations: {df['station'].nunique():,}")

    # 2) missing summary
    miss = missing_summary(df, cfg.target)
    miss_path = str(outdir / f"missing_summary_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv")
    miss.to_csv(miss_path, index=False)
    if cfg.verbose:
        print("Missing summary ->", miss_path)

    # 3) station selection
    df_sel = select_stations(df, cfg.target, min_obs=cfg.min_obs, stations=cfg.stations)
    sel_stations = df_sel["station"].astype(str).unique().tolist()
    if cfg.verbose:
        print(f"Selected stations: {len(sel_stations):,} | rows: {len(df_sel):,}")

    # 4) imputation
    if cfg.verbose:
        print("Imputing with RF ...")
    imp = MissClimateImputer(
        engine="rf", target=cfg.target,
        k_neighbors=cfg.k_neighbors, min_obs_per_station=cfg.min_obs,
        n_estimators=cfg.n_estimators, n_jobs=cfg.n_jobs, random_state=cfg.random_state
    )
    df_imp = imp.fit_transform(df_sel)
    imp_path = str(outdir / "imputed.parquet")
    save_parquet(df_imp, imp_path)
    rep = imp.report(df_imp)
    if cfg.verbose:
        print("Imputed parquet ->", imp_path)
        print("Report:", rep)

    # 5) neighbor distances
    meta = df_sel.groupby("station")[["latitude","longitude"]].median().reset_index()
    nn_df = neighbor_distances(meta, k=cfg.k_neighbors)
    nn_path = str(outdir / f"neighbor_distances_k{cfg.k_neighbors}.csv")
    nn_df.to_csv(nn_path, index=False)
    if cfg.verbose:
        print("Neighbor distances ->", nn_path)

    # 6) per-station metrics with inclusion (opcional)
    metrics_path = None
    metrics_df = None
    if cfg.do_metrics:
        if cfg.verbose:
            print(f"Evaluating per-station with train_frac={cfg.train_frac} ...")
        metrics_df = evaluate_per_station(
            df_src=df_sel, target=cfg.target,
            train_frac=cfg.train_frac, min_obs=cfg.min_obs,
            k_neighbors=cfg.k_neighbors, n_estimators=cfg.n_estimators,
            n_jobs=cfg.n_jobs, random_state=cfg.random_state
        )
        metrics_path = str(outdir / f"metrics_trainfrac_{cfg.train_frac:.2f}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        if cfg.verbose: print("Metrics ->", metrics_path)

    # 7) plots automáticos: series y métricas
    # muestrario de estaciones (primeras N)
    if cfg.plot_sample_stations > 0:
        sample = sel_stations[: cfg.plot_sample_stations]
        if cfg.verbose:
            print(f"Plotting {len(sample)} stations -> {plots_path}")

        for st in tqdm(sample, disable=not cfg.verbose):
            out_png = str(plots_path / f"series_{cfg.target}_{st}.png")
            plot_station_series(df_sel, df_imp, st, cfg.target, out_png, cfg.title_tag)

    if cfg.do_metrics and metrics_df is not None and not metrics_df.empty:
        plot_metrics_distribution(
            metrics_df,
            out_png=str(plots_path / f"metrics_{cfg.target}.png"),
            title=f"Metrics ({cfg.target})"
        )

    # 8) MDR grid (opcional)
    mdr_path = None
    if cfg.do_mdr:
        grid = cfg.mdr_grid or {
            "missing_fracs": [0.1, 0.3],
            "grid_K": [5, 8, 12],
            "grid_min_obs": [30, 60, 120],
            "grid_trees": [200, 300],
            "metric_thresholds": {"RMSE": 2.0, "R2": 0.4},
        }
        if cfg.verbose: print("Running MDR grid ...")
        mdr_df = mdr_grid_search(df_sel, cfg.target, **grid, random_state=cfg.random_state)
        mdr_path = str(outdir / "mdr_results.parquet")
        save_parquet(mdr_df, mdr_path)
        if cfg.verbose: print("MDR ->", mdr_path)

    # 9) report json
    report = {
        "target": cfg.target,
        "period": cfg.period,
        "rows": int(len(df_sel)),
        "stations": int(len(sel_stations)),
        "imputation_report": rep,
        "paths": {
            "missing_summary_csv": miss_path,
            "imputed_parquet": imp_path,
            "neighbor_distances_csv": nn_path,
            "metrics_csv": metrics_path,
            "mdr_parquet": mdr_path,
            "plots_dir": str(plots_path),
        },
    }
    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    if cfg.verbose: print("Done.")
    return report
