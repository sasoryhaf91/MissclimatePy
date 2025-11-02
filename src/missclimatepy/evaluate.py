# src/missclimatepy/evaluate.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, r2_score

from .api import MissClimateImputer


# ---- utilities ----------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute RMSE without relying on sklearn's `squared=False`
    (keeps compatibility with older CI environments).
    """
    diff = y_true - y_pred
    return float(np.sqrt(np.nanmean(diff * diff)))


# ---- masking ------------------------------------------------------------------


def mask_by_inclusion(
    df: pd.DataFrame,
    target: str,
    train_frac: float,
    min_obs: int,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Randomly include a fraction of *observed* target values per station,
    and mask (set to NaN) the remaining observed values. This is used to
    simulate missingness for per-station evaluation.

    Parameters
    ----------
    df : DataFrame
        Long-format table with at least: ['station', 'date', 'latitude',
        'longitude', 'altitude', <target>].
    target : str
        Target column name to evaluate (e.g., 'tmin').
    train_frac : float
        Fraction in [0, 1]. Proportion of *observed* samples per station
        that will remain visible for training. The complement will be
        masked and used as a test fold. Use 0.0 for strict LOSO.
    min_obs : int
        Minimum number of observed rows required to consider a station
        for masking/evaluation. Stations with fewer valid rows are ignored.
    random_state : int
        Seed for the random generator.

    Returns
    -------
    masked_df : DataFrame
        Copy of `df` where the test subset (observed but not included)
        is set to NaN in `target`.
    info : DataFrame
        One row per station with counts: n_total, n_train, n_valid.
        Note that n_valid are the masked (test) items.
    """
    if not (0.0 <= train_frac <= 1.0):
        raise ValueError("train_frac must be within [0, 1].")

    rng = np.random.default_rng(random_state)
    masked = df.copy()
    info_rows = []

    for st, sub in df.groupby("station", sort=False):
        idx_obs = sub.index[sub[target].notna()].to_numpy()
        n_total = int(idx_obs.size)
        if n_total < int(min_obs):
            # not enough valid rows -> ignore in masking/eval
            continue

        # number of training items to keep visible
        n_train = int(round(train_frac * n_total))
        if n_train > 0:
            train_idx = rng.choice(idx_obs, size=min(n_train, n_total), replace=False)
            valid_idx = np.setdiff1d(idx_obs, train_idx, assume_unique=False)
        else:
            train_idx = np.empty(0, dtype=int)
            valid_idx = idx_obs  # all observed => test/validation fold

        # mask the test fold in the copy
        masked.loc[valid_idx, target] = np.nan

        info_rows.append(
            {
                "station": st,
                "n_total": n_total,               # observed items before masking
                "n_train": int(train_idx.size),   # included for training
                "n_valid": int(valid_idx.size),   # held-out for testing
            }
        )

    info = pd.DataFrame(info_rows, columns=["station", "n_total", "n_train", "n_valid"])
    return masked, info


# ---- evaluation ---------------------------------------------------------------


def evaluate_per_station(
    df_src: pd.DataFrame,
    target: str,
    train_frac: float = 0.7,
    min_obs: int = 60,
    engine: str = "rf",
    model_params: Optional[Dict[str, Any]] = None,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Per-station masked evaluation for local imputation models.

    The function:
      1) Selects stations with at least `min_obs` valid target values.
      2) Masks a (1 - train_frac) portion of *observed* rows per station
         to simulate missingness (the masked part becomes the test fold).
      3) Fits a global MissClimateImputer (x, y, z + calendar) on the masked
         dataset (so it only "sees" the included rows).
      4) Predicts the masked rows per station and computes MAE, RMSE and R2.

    Robustness:
      - If a station ends with no valid test pairs (after filtering NaNs), it
        will return NaN metrics for that station instead of failing.

    Parameters
    ----------
    df_src : DataFrame
        Long-format table with at least:
        ['station', 'date', 'latitude', 'longitude', 'altitude', <target>].
    target : str
        Target variable to impute/evaluate (e.g., 'tmin').
    train_frac : float, default 0.7
        Fraction of *observed* target values that remain visible for
        training. 0.0 means strict LOSO (the model will still train using
        neighbors from other stations).
    min_obs : int, default 60
        Minimum number of observed rows required for a station to be
        included in evaluation/masking.
    engine : str, default 'rf'
        Imputation engine ID, forwarded to `MissClimateImputer`.
    model_params : dict, optional
        Hyperparameters for the underlying model. Forwarded to
        `MissClimateImputer(engine=..., model_params=...)`.
    random_state : int, default 42
        Seed used both in masking and (through the imputer) model fitting.

    Returns
    -------
    DataFrame
        One row per evaluated station with columns:
        ['station', 'n_valid', 'MAE', 'RMSE', 'R2'].
        The frame is sorted by RMSE ascending when metrics are available.
    """
    # Basic column presence check
    required = {"station", "date", "latitude", "longitude", "altitude", target}
    missing_cols = required - set(df_src.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

    # Mask dataset according to the inclusion policy
    masked, _info = mask_by_inclusion(
        df_src, target=target, train_frac=train_frac, min_obs=min_obs, random_state=random_state
    )

    # Minimum number of *included* samples per station for the imputer.
    # For strict LOSO we allow 0.
    min_obs_train = max(0, int(round(min_obs * train_frac)))

    # Configure and fit the global imputer on the masked dataset
    imp = MissClimateImputer(
        engine=engine,
        target=target,
        min_obs_per_station=min_obs_train,
        model_params=model_params,
        random_state=random_state,
    )
    df_hat = imp.fit_transform(masked)

    # Compute per-station metrics on the masked (test) portion
    rows = []
    for st, sub in df_src.groupby("station", sort=False):
        # rows that were originally observed and got masked -> test fold
        sub_masked = masked.loc[sub.index]
        test_idx = sub_masked.index[sub_masked[target].isna() & sub[target].notna()].to_numpy()

        if test_idx.size == 0:
            # Nothing to evaluate for this station
            rows.append({"station": st, "n_valid": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
            continue

        y_true = sub.loc[test_idx, target].to_numpy()
        y_pred = df_hat.loc[test_idx, target].to_numpy()

        # guard against NaNs in either vector
        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        n_valid = int(ok.sum())
        if n_valid == 0:
            rows.append({"station": st, "n_valid": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan})
            continue

        mae = float(mean_absolute_error(y_true[ok], y_pred[ok]))
        rmse = _rmse(y_true[ok], y_pred[ok])
        # R2 requires at least 2 samples with variance
        r2 = float(r2_score(y_true[ok], y_pred[ok])) if n_valid >= 2 else np.nan

        rows.append({"station": st, "n_valid": n_valid, "MAE": mae, "RMSE": rmse, "R2": r2})

    result = pd.DataFrame(rows, columns=["station", "n_valid", "MAE", "RMSE", "R2"])
    if not result.empty and result["RMSE"].notna().any():
        result = result.sort_values("RMSE", na_position="last", ignore_index=True)
    return result
