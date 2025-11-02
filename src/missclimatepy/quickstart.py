from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd

from .io import load_any, save_parquet
from .prepare import enforce_schema, filter_period, missing_summary, select_stations
from .neighbors import neighbor_distances
from .api import MissClimateImputer
from .evaluate import evaluate_per_station
from .mdr import mdr_grid_search

@dataclass
class QuickstartConfig:
    data_path: str
    target: str
    period: tuple[str,str] = ("1991-01-01","2020-12-31")
    stations: list[str] | None = None
    min_obs: int = 60
    k_neighbors: int = 12
    n_estimators: int = 300
    n_jobs: int = -1
    random_state: int = 42
    outputs_dir: str = "./outputs"
    do_metrics: bool = True
    train_frac: float = 0.7
    do_mdr: bool = False
    mdr_grid: dict | None = None  # keys: missing_fracs, grid_K, grid_min_obs, grid_trees, metric_thresholds

def run_quickstart(
    data_path: str, target: str,
    period=("1991-01-01","2020-12-31"),
    stations=None, min_obs=60,
    k_neighbors=12, n_estimators=300, n_jobs=-1, random_state=42,
    outputs_dir="./outputs",
    do_metrics=True, train_frac=0.7,
    do_mdr=False, mdr_grid=None
):
    cfg = QuickstartConfig(
        data_path=data_path, target=target, period=period, stations=stations,
        min_obs=min_obs, k_neighbors=k_neighbors, n_estimators=n_estimators,
        n_jobs=n_jobs, random_state=random_state, outputs_dir=outputs_dir,
        do_metrics=do_metrics, train_frac=train_frac, do_mdr=do_mdr,
        mdr_grid=mdr_grid
    )
    return _run(cfg)

def _run(cfg: QuickstartConfig):
    Path(cfg.outputs_dir).mkdir(parents=True, exist_ok=True)

    # 1) load & prepare
    df = load_any(cfg.data_path)
    if "altitude" in df.columns and "elevation" not in df.columns:
        df = df.rename(columns={"altitude":"elevation"})
    df = enforce_schema(df, cfg.target)
    df = filter_period(df, cfg.period[0], cfg.period[1])

    # 2) missing summary
    miss = missing_summary(df, cfg.target)
    miss_path = f"{cfg.outputs_dir}/missing_summary_{cfg.target}_{cfg.period[0][:4]}_{cfg.period[1][:4]}.csv"
    miss.to_csv(miss_path, index=False)

    # 3) station selection
    df_sel = select_stations(df, cfg.target, min_obs=cfg.min_obs, stations=cfg.stations)

    # 4) imputation
    imp = MissClimateImputer(
        engine="rf", target=cfg.target,
        k_neighbors=cfg.k_neighbors, min_obs_per_station=cfg.min_obs,
        n_estimators=cfg.n_estimators, n_jobs=cfg.n_jobs, random_state=cfg.random_state
    )
    df_imp = imp.fit_transform(df_sel)
    imp_path = f"{cfg.outputs_dir}/imputed.parquet"
    save_parquet(df_imp, imp_path)
    rep = imp.report(df_imp)

    # 5) neighbor distances
    meta = df_sel.groupby("station")[["latitude","longitude"]].median().reset_index()
    nn_df = neighbor_distances(meta, k=cfg.k_neighbors)
    nn_path = f"{cfg.outputs_dir}/neighbor_distances_k{cfg.k_neighbors}.csv"
    nn_df.to_csv(nn_path, index=False)

    # 6) per-station metrics with inclusion
    metrics_path = None
    if cfg.do_metrics:
        eval_df = evaluate_per_station(
            df_src=df_sel, target=cfg.target,
            train_frac=cfg.train_frac, min_obs=cfg.min_obs,
            k_neighbors=cfg.k_neighbors, n_estimators=cfg.n_estimators,
            n_jobs=cfg.n_jobs, random_state=cfg.random_state
        )
        metrics_path = f"{cfg.outputs_dir}/metrics_trainfrac_{cfg.train_frac:.2f}.csv"
        eval_df.to_csv(metrics_path, index=False)

    # 7) MDR grid (optional)
    mdr_path = None
    if cfg.do_mdr:
        grid = cfg.mdr_grid or {
            "missing_fracs": [0.1, 0.3],
            "grid_K": [5, 8, 12],
            "grid_min_obs": [30, 60, 120],
            "grid_trees": [200, 300],
            "metric_thresholds": {"RMSE": 2.0, "R2": 0.4},
        }
        mdr_df = mdr_grid_search(df_sel, cfg.target, **grid, random_state=cfg.random_state)
        mdr_path = f"{cfg.outputs_dir}/mdr_results.parquet"
        save_parquet(mdr_df, mdr_path)

    # 8) report json
    report = {
        "target": cfg.target,
        "period": cfg.period,
        "rows": int(len(df_sel)),
        "stations": int(df_sel["station"].nunique()),
        "imputation_report": rep,
        "paths": {
            "missing_summary_csv": miss_path,
            "imputed_parquet": imp_path,
            "neighbor_distances_csv": nn_path,
            "metrics_csv": metrics_path,
            "mdr_parquet": mdr_path,
        },
    }
    with open(f"{cfg.outputs_dir}/report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report
