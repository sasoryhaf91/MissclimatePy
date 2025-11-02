from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .api import MissClimateImputer

def mask_by_inclusion(df: pd.DataFrame, target: str,
                      train_frac: float, min_obs: int,
                      random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    masked = df.copy()
    info = []
    for st, sub in df.groupby("station"):
        idx_obs = sub[sub[target].notna()].index.values
        if len(idx_obs) < int(min_obs):
            continue
        n_train = max(1, int(len(idx_obs) * train_frac))
        train_idx = rng.choice(idx_obs, size=n_train, replace=False)
        valid_idx = np.setdiff1d(idx_obs, train_idx)
        masked.loc[valid_idx, target] = np.nan
        info.append({"station": st, "n_total": len(idx_obs),
                     "n_train": len(train_idx), "n_valid": len(valid_idx)})
    return masked, pd.DataFrame(info)

def evaluate_per_station(df_src: pd.DataFrame, target: str,
                         train_frac: float = 0.7,
                         min_obs: int = 60,
                         k_neighbors: int = 12,
                         n_estimators: int = 300,
                         n_jobs: int = -1,
                         random_state: int = 42,
                         rf_params: dict | None = None) -> pd.DataFrame:
    masked, _ = mask_by_inclusion(df_src, target, train_frac, min_obs, random_state)

    can_rf_params = ("rf_params" in MissClimateImputer.__init__.__code__.co_varnames)
    imp = MissClimateImputer(
        engine="rf", target=target,
        k_neighbors=k_neighbors, min_obs_per_station=min_obs,
        n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state,
        **({"rf_params": rf_params} if (rf_params and can_rf_params) else {})
    )
    imp_df = imp.fit_transform(masked)

    rows = []
    for st, sub in df_src.groupby("station"):
        msub = masked[masked["station"] == st]
        is_valid = msub[target].isna() & sub[target].notna()
        idx = msub[is_valid].index
        if len(idx) == 0:
            continue
        y_true = sub.loc[idx, target].values
        y_pred = imp_df.loc[idx, target].values
        rows.append({
            "station": st, "n_valid": len(idx),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": mean_squared_error(y_true, y_pred, squared=False),
            "R2": r2_score(y_true, y_pred),
        })
    return pd.DataFrame(rows).sort_values("RMSE")
