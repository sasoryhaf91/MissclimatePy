# SPDX-License-Identifier: MIT
# src/missclimatepy/quickstart.py
"""
Quickstart pipeline for MissClimatePy
=====================================

This module offers a batteries-included entry point to:
1) load a daily station dataset,
2) enforce the expected schema (with elevation/altitude compatibility),
3) apply an optional period filter,
4) summarize missingness,
5) (optionally) compute a local-neighborhood map (k nearest stations),
6) train a global imputer on the *full* training universe, and
7) evaluate per-station on a (possibly small) subset for fast iteration.

Key principles
--------------
- **Train on all stations** that pass the period filter (the “training universe”).
- **Evaluate only** on the station list passed in `stations` (or all if None).
- **k_neighbors** constrains the *local training pool* for each evaluated station
  (neighbor restriction) but does **not** reduce the global missingness summary.
- Neighbor calculation accepts both return styles:
    * dict:  {station: [self, n1, n2, ...]}  (self may be present)
    * long DataFrame with columns ['station','neighbor','rank','distance_km']

Typical usage
-------------
>>> from missclimatepy.quickstart import run_quickstart
>>> report = run_quickstart(
...     data_path="/kaggle/input/datos.csv",
...     target="tmin",
...     period=("1991-01-01", "2020-12-31"),
...     stations=["15308","15122","20319"],   # evaluate these (train uses all)
...     k_neighbors=20,                       # local neighborhood
...     n_estimators=200,                     # RF quick override
...     do_metrics=True,                      # per-station masked evaluation
... )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import os
import pandas as pd

from .prepare import (
    enforce_schema,
    filter_period,
    missing_summary,
)
from .neighbors import neighbor_distances
from .evaluate import evaluate_per_station
from .api import MissClimateImputer


# ---------------------------------------------------------------------------
# Small internal utilities
# ---------------------------------------------------------------------------

def _load_dataframe(path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Load a DataFrame from CSV/Parquet/Feather, or pass through if already a DataFrame.
    Also harmonizes `elevation` -> `altitude` if only `elevation` is present.

    Parameters
    ----------
    path_or_df : str | DataFrame
        File path or in-memory DataFrame.

    Returns
    -------
    DataFrame
        Loaded and lightly harmonized table.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        path = str(path_or_df)
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".txt"):
            df = pd.read_csv(path)
        elif ext in (".parquet",):
            df = pd.read_parquet(path)
        elif ext in (".feather",):
            df = pd.read_feather(path)
        else:
            raise ValueError(f"Unsupported input extension: {ext}")

    # Schema compatibility: accept either 'altitude' or 'elevation'
    if "altitude" not in df.columns and "elevation" in df.columns:
        df = df.rename(columns={"elevation": "altitude"})
    return df


def _ensure_dir(path: Optional[str]) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class QuickstartConfig:
    # Input & target
    data_path: Union[str, pd.DataFrame]
    target: str = "tmin"

    # Period filter: (start, end) or None to skip
    period: Optional[Tuple[str, str]] = None

    # Which stations to EVALUATE (training uses all that survive the period filter)
    stations: Optional[Iterable[Union[str, int]]] = None

    # Minimum valid rows to include a station in the training universe
    min_obs: int = 60

    # Local neighborhood size (None or 0 = disable)
    k_neighbors: Optional[int] = None

    # RandomForest quick overrides (legacy-friendly)
    n_estimators: Optional[int] = None
    max_depth: Optional[int] = None
    min_samples_leaf: Optional[int] = None

    # Or pass a full parameter dict to the engine (preferred)
    rf_params: Optional[Dict[str, Any]] = None
    model_params: Optional[Dict[str, Any]] = None

    # Evaluation controls
    do_metrics: bool = True               # run masked evaluation
    train_frac: float = 0.8               # fraction kept for training in the mask-by-inclusion
    random_state: int = 42
    n_jobs: int = -1

    # Output / plotting
    outputs_dir: Optional[str] = None     # base output directory (CSV summaries etc.)
    plots_dir: Optional[str] = None       # optional figures directory (if your viz uses it)
    plot_sample_stations: int = 0         # how many stations to plot (if you plug plotting)
    title_tag: str = ""

    # Verbose
    verbose: bool = True


# ---------------------------------------------------------------------------
# Quickstart runner
# ---------------------------------------------------------------------------

def run_quickstart(**kwargs) -> Dict[str, Any]:
    """
    Build a `QuickstartConfig` from kwargs and execute the pipeline.

    Returns
    -------
    dict
        A report with counts and output file paths.
    """
    cfg = QuickstartConfig(**kwargs)
    return _run(cfg)


def _run(cfg: QuickstartConfig) -> Dict[str, Any]:
    # -----------------------------------------------------------------------
    # 1) Load & basic schema
    # -----------------------------------------------------------------------
    if cfg.verbose:
        print(f"Loading: {cfg.data_path}")
    df = _load_dataframe(cfg.data_path)

    # Enforce the package schema (expects 'altitude' after our harmonization)
    df = enforce_schema(df, target=cfg.target)

    # -----------------------------------------------------------------------
    # 2) Period filter (train universe)
    # -----------------------------------------------------------------------
    if cfg.period is not None:
        start, end = cfg.period
        df = filter_period(df, start=start, end=end)

    if cfg.verbose:
        print(f"Rows after period filter: {len(df):,} | stations: {df['station'].nunique():,}")

    # -----------------------------------------------------------------------
    # 3) Missingness summary (global insight, not restricted by evaluation subset)
    # -----------------------------------------------------------------------
    outputs_dir = cfg.outputs_dir or "/kaggle/working/mcp"
    _ensure_dir(outputs_dir)
    miss_path = os.path.join(
        outputs_dir, f"missing_summary_{cfg.target}_{(cfg.period[0] if cfg.period else 'all')}_{(cfg.period[1] if cfg.period else 'all')}.csv"
    )
    missing_summary(df, target=cfg.target).to_csv(miss_path, index=False)
    if cfg.verbose:
        print(f"Missing summary -> {miss_path}")

    # -----------------------------------------------------------------------
    # 4) Decide which stations to EVALUATE (train still uses ALL)
    # -----------------------------------------------------------------------
    if cfg.stations is None:
        eval_stations = sorted(df["station"].astype(str).unique().tolist())
    else:
        eval_stations = [str(s) for s in cfg.stations]

    # Optionally keep only stations with enough valid rows in the *training universe*
    if cfg.min_obs and cfg.min_obs > 0:
        valid_counts = (
            df.loc[df[cfg.target].notna(), ["station"]]
            .groupby("station").size().astype(int)
        )
        before = len(eval_stations)
        eval_stations = [s for s in eval_stations if int(valid_counts.get(s, 0)) >= int(cfg.min_obs)]
        if cfg.verbose:
            print(f"Selected stations: {before} → {len(eval_stations)} after min_obs={cfg.min_obs}")

    # -----------------------------------------------------------------------
    # 5) Optional neighbor map (used to restrict local training pool per station)
    #    Robust to either dict output or long-DataFrame output.
    # -----------------------------------------------------------------------
    neighbor_map: Optional[Dict[str, List[str]]] = None
    if cfg.k_neighbors is not None and cfg.k_neighbors > 0:
        stations_df = df[["station", "latitude", "longitude"]].drop_duplicates("station")

        # Compute k+1 so we can safely include self at rank 0 and then remove it
        neigh = neighbor_distances(
            stations_df,
            n_neighbors=int(cfg.k_neighbors) + 1,   # function signature is n_neighbors=...
        )

        if isinstance(neigh, dict):
            # New API: dict {station: [self?, n1, n2, ...]}
            neighbor_map = {
                str(s): [str(x) for x in ids if str(x) != str(s)][: int(cfg.k_neighbors)]
                for s, ids in neigh.items()
            }
        else:
            # Legacy long format: ['station','neighbor','rank','distance_km']
            nd = neigh.copy()
            nd["station"] = nd["station"].astype(str)
            nd["neighbor"] = nd["neighbor"].astype(str)
            neighbor_map = {}
            for st, rows in nd.groupby("station"):
                ids = rows.sort_values("distance_km")["neighbor"].tolist()
                ids = [i for i in ids if i != st]
                neighbor_map[st] = ids[: int(cfg.k_neighbors)]

    # -----------------------------------------------------------------------
    # 6) Train a GLOBAL model (full universe) and optionally evaluate per-station
    #    We expose `k_neighbors` via MissClimateImputer for local training,
    #    but the imputer still sees the whole DataFrame so it can subset per station.
    # -----------------------------------------------------------------------
    # Legacy single-parameter overrides for RF kept for convenience:
    rf_params = dict(cfg.rf_params or {})
    if cfg.n_estimators is not None:
        rf_params["n_estimators"] = cfg.n_estimators
    if cfg.max_depth is not None:
        rf_params["max_depth"] = cfg.max_depth
    if cfg.min_samples_leaf is not None:
        rf_params["min_samples_leaf"] = cfg.min_samples_leaf

    # Build the imputer (engine="rf" by default)
    imp = MissClimateImputer(
        engine="rf",
        target=cfg.target,
        k_neighbors=cfg.k_neighbors,
        min_obs_per_station=max(0, int(cfg.min_obs * cfg.train_frac)),  # for masked eval
        n_jobs=cfg.n_jobs,
        random_state=cfg.random_state,
        rf_params=rf_params,
        model_params=cfg.model_params,
    )

    # Fit once on the full training universe
    if cfg.verbose:
        print("Fitting global imputer on training universe ...")
    _ = imp.fit(df)

    # -----------------------------------------------------------------------
    # 7) Per-station masked evaluation (optional)
    # -----------------------------------------------------------------------
    metrics_df_path = None
    metrics_df = pd.DataFrame()
    if cfg.do_metrics:
        if cfg.verbose:
            print(f"Evaluating per-station (train_frac={cfg.train_frac:.2f}) ...")

        # Evaluate only on eval_stations, but still with global training.
        # The evaluate_per_station() in this package performs mask-by-inclusion
        # and recomputes the imputer internally when needed. If you prefer to
        # reuse the already-fitted `imp`, you can adapt evaluate_per_station
        # to accept a pre-fitted imputer; kept simple here for clarity.
        metrics_df = evaluate_per_station(
            df_src=df[df["station"].astype(str).isin(eval_stations)],
            target=cfg.target,
            train_frac=cfg.train_frac,
            min_obs=cfg.min_obs,
            engine="rf",
            k_neighbors=cfg.k_neighbors,
            model_params=dict(rf_params=rf_params, **(cfg.model_params or {})) if rf_params else cfg.model_params,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        metrics_df_path = os.path.join(
            outputs_dir,
            f"metrics_{cfg.target}_{(cfg.period[0] if cfg.period else 'all')}_{(cfg.period[1] if cfg.period else 'all')}.csv",
        )
        metrics_df.to_csv(metrics_df_path, index=False)

    # -----------------------------------------------------------------------
    # 8) Optional quick imputations for the evaluation subset (for inspection)
    # -----------------------------------------------------------------------
    # Note: This imputes the original df and then filters rows belonging to the
    # evaluation stations. Training was already done globally (fit above).
    if cfg.verbose:
        print(f"Imputing with RF (global training, local neighbors={cfg.k_neighbors}) ...")
    df_after = imp.transform(df)
    df_eval_imputed = df_after[df_after["station"].astype(str).isin(eval_stations)].copy()

    # Save a light preview if requested
    preview_path = os.path.join(
        outputs_dir,
        f"imputed_preview_{cfg.target}_{len(eval_stations)}stations.csv"
    )
    df_eval_imputed.head(100000).to_csv(preview_path, index=False)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report: Dict[str, Any] = {
        "n_rows_after_period": int(len(df)),
        "n_stations_after_period": int(df["station"].nunique()),
        "n_eval_stations": int(len(eval_stations)),
        "outputs_dir": outputs_dir,
        "missing_summary_csv": miss_path,
        "imputed_preview_csv": preview_path,
    }
    if cfg.do_metrics:
        report["metrics_csv"] = metrics_df_path
        report["metrics_table"] = metrics_df
    if neighbor_map is not None:
        # Only counts — neighbor_map can be large
        report["has_neighbor_map"] = True
        report["k_neighbors"] = int(cfg.k_neighbors or 0)

    if cfg.verbose:
        print(f"Training universe: rows={len(df):,} | stations={df['station'].nunique():,}")
        print(f"Evaluating only {len(eval_stations)} station(s).")

    return report
