# src/missclimatepy/quickstart.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json, pandas as pd

from .io import load_any, save_parquet
from .prepare import enforce_schema, filter_period, missing_summary, select_stations
from .api import MissClimateImputer
from .evaluate import evaluate_per_station
from .mdr import inclusion_sweep, recommend_min_inclusion
from .viz import plot_inclusion_aggregate

@dataclass
class QuickstartConfig:
    data_path: str
    target: str
    engine: str = "rf"
    period: Tuple[str,str] = ("1991-01-01","2020-12-31")
    stations: Optional[List[str]] = None
    min_obs: int = 60
    model_params: Optional[Dict[str, Any]] = None

    outputs_dir: str = "./outputs"
    verbose: bool = True

    do_metrics: bool = True
    train_frac: float = 0.7

    do_sweep: bool = False
    include_pcts: Optional[List[float]] = None
    thresholds: Dict[str, float] = None

    plot_sample_stations: int = 4
    plots_dir: str = "plots"
    title_tag: str = ""

def run_quickstart(**kwargs):
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)

def _run(cfg: QuickstartConfig):
    outdir = Path(cfg.outputs_dir); outdir.mkdir(parents=True, exist_ok=True)
    pplots = outdir / cfg.plots_dir; pplots.mkdir(parents=True, exist_ok=True)

    if cfg.verbose: print("Loading:", cfg.data_path)
    df = load_any(cfg.data_path)
    if "altitude" in df.columns and "elevation" not in df.columns:
        df = df.rename(columns={"altitude":"elevation"})
    df = enforce_schema(df, cfg.target)
    df = filter_period(df, cfg.period[0], cfg.period[1])
    if cfg.verbose: print(f"Rows after period filter: {len(df):,} | stations: {df['station'].nunique():,}")

    miss = missing_summary(df, cfg.target)
    miss_path = outdir / f"missing_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv"
    miss.to_csv(miss_path, index=False)
    if cfg.verbose: print("Missing summary ->", miss_path)

    df_sel = select_stations(df, cfg.target, min_obs=cfg.min_obs, stations=cfg.stations)
    S = df_sel["station"].astype(str).unique().tolist()
    if cfg.verbose: print(f"Selected stations: {len(S):,} | rows: {len(df_sel):,}")

    if cfg.verbose: print(f"Imputing with engine='{cfg.engine}' ...")
    imp = MissClimateImputer(engine=cfg.engine, target=cfg.target,
                             min_obs_per_station=cfg.min_obs,
                             model_params=cfg.model_params)
    df_imp = imp.fit_transform(df_sel)
    imp_path = outdir / "imputed.parquet"; save_parquet(df_imp, str(imp_path))
    rep = imp.report(df_imp)
    if cfg.verbose: print("Imputed parquet ->", imp_path); print("Report:", rep)

    metrics_path = None
    if cfg.do_metrics:
        if cfg.verbose: print(f"Evaluating per-station with train_frac={cfg.train_frac} ...")
        metr = evaluate_per_station(df_sel, cfg.target, train_frac=cfg.train_frac,
                                    min_obs=cfg.min_obs, engine=cfg.engine,
                                    model_params=cfg.model_params)
        if not metr.empty:
            metrics_path = outdir / f"metrics_{cfg.target}_{cfg.train_frac:.2f}.csv"
            metr.to_csv(metrics_path, index=False)
            if cfg.verbose: print("Metrics ->", metrics_path)

    sweep_path = None; rec_path = None
    if cfg.do_sweep:
        include_pcts = cfg.include_pcts or [0.0,0.04,0.1,0.2,0.4,0.6,0.8]
        if cfg.verbose: print("Running inclusion sweep:", include_pcts)
        sw = inclusion_sweep(df_sel, cfg.target, include_pcts=include_pcts,
                             period=cfg.period, stations=S, min_obs=cfg.min_obs,
                             engine=cfg.engine, model_params=cfg.model_params)
        sweep_path = outdir / "sweep.csv"; sw.to_csv(sweep_path, index=False)
        if cfg.verbose: print("Sweep ->", sweep_path)
        if not sw.empty:
            plot_inclusion_aggregate(sw, metric="RMSE",
                                     out_png=str(pplots / "agg_RMSE.png"))
            plot_inclusion_aggregate(sw, metric="R2",
                                     out_png=str(pplots / "agg_R2.png"))
            rec = recommend_min_inclusion(sw, thresholds=(cfg.thresholds or {"R2":0.5,"RMSE":2.0}))
            rec_path = outdir / "sweep_recommendation.csv"; rec.to_csv(rec_path, index=False)
            if cfg.verbose: print("Sweep recommendations ->", rec_path)

    report = {
        "target": cfg.target, "engine": cfg.engine, "period": cfg.period,
        "rows": int(len(df_sel)), "stations": int(len(S)),
        "imputation_report": rep,
        "paths": {
            "missing_summary_csv": str(miss_path),
            "imputed_parquet": str(imp_path),
            "metrics_csv": str(metrics_path) if metrics_path else None,
            "sweep_csv": str(sweep_path) if sweep_path else None,
            "sweep_recommendation_csv": str(rec_path) if rec_path else None,
            "plots_dir": str(pplots),
        },
    }
    with open(outdir / "report.json", "w") as f: json.dump(report, f, indent=2)
    if cfg.verbose: print("Done.")
    return report

