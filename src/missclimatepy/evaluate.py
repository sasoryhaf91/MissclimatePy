# src/missclimatepy/evaluate.py
from __future__ import annotations

from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from .api import MissClimateImputer


def mask_by_inclusion(
    df: pd.DataFrame,
    target: str,
    train_frac: float,
    min_obs: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly mask a fraction of valid observations per-station to evaluate
    imputation quality (i.e., simulate missingness).

    Parameters
    ----------
    df : pd.DataFrame
        Long table with columns: ['station','date','latitude','longitude','elevation', target, ...].
    target : str
        Target column to impute/evaluate.
    train_frac : float
        Fraction of valid rows (per-station) kept as *visible* to the model.
        The complement (1 - train_frac) is masked and later used as validation.
        Use 0.0 for strict LOSO (no visible target at the station).
    min_obs : int
        Minimum number of valid observations required at the station to be considered.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    masked : pd.DataFrame
        Copy of `df` with a subset of the target set to NaN (validation part).
    info : pd.DataFrame
        Per-station summary with columns: ['station','n_total','n_train','n_valid'].
    """
    assert 0.0 <= train_frac <= 1.0, "train_frac must be in [0,1]"

    rng = np.random.default_rng(random_state)
    masked = df.copy()
    rows: List[Dict[str, int]] = []

    for st, sub in df.groupby("station"):
        idx_obs = sub.index[sub[target].notna()].to_numpy()
        if idx_obs.size < int(min_obs):
            continue

        n_train = int(idx_obs.size * train_frac)
        if n_train > 0:
            train_idx = rng.choice(idx_obs, size=n_train, replace=False)
            valid_idx = np.setdiff1d(idx_obs, train_idx)
        else:
            train_idx = np.empty(0, dtype=int)
            valid_idx = idx_obs

        # mask validation subset
        masked.loc[valid_idx, target] = np.nan

        rows.append(
            {
                "station": st,
                "n_total": int(idx_obs.size),
                "n_train": int(n_train),
                "n_valid": int(valid_idx.size),
            }
        )

    info = pd.DataFrame(rows, columns=["station", "n_total", "n_train", "n_valid"])
    return masked, info


def evaluate_per_station(
    df_src: pd.DataFrame,
    target: str,
    train_frac: float = 0.7,
    min_obs: int = 60,
    engine: str = "rf",
    k_neighbors: int = 20,
    model_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Evaluate imputation quality per-station by masking a fraction of valid
    values and comparing predictions vs. ground truth.

    This function fully supports LOSO: set `train_frac=0.0` to hide all target
    values of the station and rely only on spatio-temporal neighbors.

    Parameters
    ----------
    df_src : pd.DataFrame
        Long table with at least: ['station','date','latitude','longitude','elevation', target].
    target : str
        Target column to impute/evaluate.
    train_frac : float, default=0.7
        Fraction of valid rows per station kept as visible to the model.
        Use 0.0 for strict LOSO.
    min_obs : int, default=60
        Minimum number of valid observations required to include a station.
    engine : str, default="rf"
        Imputation engine name (currently 'rf').
    k_neighbors : int, default=20
        Number of spatio-temporal neighbors used to form the training subset
        for local predictions.
    model_params : dict, optional
        Extra hyperparameters forwarded to the underlying model/engine.
        Typical RF keys: {'n_estimators', 'max_depth', 'min_samples_leaf', ...}.
    random_state : int, default=42
        Random seed for masking and model reproducibility.
    n_jobs : int, default=-1
        Parallel jobs for the engine (if supported).

    Returns
    -------
    pd.DataFrame
        Per-station metrics with columns:
        ['station','n_valid','MAE','RMSE','R2'] sorted by RMSE.
    """
    masked, _ = mask_by_inclusion(df_src, target, train_frac, min_obs, random_state)

    # LOSO-friendly: require 0 visible obs at the target station when train_frac=0
    min_obs_train = max(0, int(min_obs * train_frac))

    # Prepare model params (ensure k_neighbors is set)
    mparams = dict(model_params or {})
    mparams.setdefault("k_neighbors", k_neighbors)

    imp = MissClimateImputer(
        engine=engine,
        target=target,
        min_obs_per_station=min_obs_train,
        model_params=mparams,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    imp_df = imp.fit_transform(masked)

    # Compute metrics per station, discarding any NaN predictions just in case
    rows = []
    for st, sub in df_src.groupby("station"):
        msub = masked.loc[sub.index]
        idx = msub.index[(msub[target].isna()) & (sub[target].notna())]
        if idx.size == 0:
            continue

        y_true = sub.loc[idx, target].to_numpy()
        y_pred = imp_df.loc[idx, target].to_numpy()

        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        if ok.sum() == 0:
            continue

        mae = float(mean_absolute_error(y_true[ok], y_pred[ok]))
        rmse = float(np.sqrt(np.mean((y_true[ok] - y_pred[ok]) ** 2)))
        r2 = float(r2_score(y_true[ok], y_pred[ok]))

        rows.append({"station": st, "n_valid": int(ok.sum()), "MAE": mae, "RMSE": rmse, "R2": r2})

    cols = ["station", "n_valid", "MAE", "RMSE", "R2"]
    return pd.DataFrame(rows, columns=cols).sort_values("RMSE") if rows else pd.DataFrame(columns=cols)
