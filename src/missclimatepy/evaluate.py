# src/missclimatepy/evaluate.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .api import MissClimateImputer

def mask_by_inclusion(df: pd.DataFrame, target: str, train_frac: float,
                      min_obs: int, random_state: int=42):
    assert 0.0 <= train_frac <= 1.0
    rng = np.random.default_rng(random_state)
    masked = df.copy(); info = []
    for st, sub in df.groupby("station"):
        idx_obs = sub[sub[target].notna()].index.values
        if len(idx_obs) < int(min_obs):
            continue
        n_train = int(len(idx_obs)*train_frac)
        if n_train > 0:
            train_idx = rng.choice(idx_obs, size=n_train, replace=False)
            valid_idx = np.setdiff1d(idx_obs, train_idx)
        else:
            train_idx = np.array([], dtype=int)
            valid_idx = idx_obs
        masked.loc[valid_idx, target] = np.nan
        info.append({"station": st, "n_total": len(idx_obs),
                     "n_train": int(n_train), "n_valid": int(len(valid_idx))})
    return masked, pd.DataFrame(info)

def evaluate_per_station(df_src: pd.DataFrame, target: str, train_frac: float=0.7,
                         min_obs: int=60, engine: str="rf",
                         model_params: Optional[Dict[str,Any]]=None) -> pd.DataFrame:
    masked, _ = mask_by_inclusion(df_src, target, train_frac, min_obs)
    min_obs_train = max(0, int(min_obs*train_frac))  # 0 for LOSO

    imp = MissClimateImputer(engine=engine, target=target,
                             min_obs_per_station=min_obs_train,
                             model_params=model_params)
    imp_df = imp.fit_transform(masked)

    rows = []
    for st, sub in df_src.groupby("station"):
        msub = masked[masked["station"]==st]
        idx = msub[msub[target].isna() & sub[target].notna()].index
        if len(idx)==0: continue
        y_true = sub.loc[idx, target].to_numpy()
        y_pred = imp_df.loc[idx, target].to_numpy()
        ok = np.isfinite(y_true) & np.isfinite(y_pred)
        if ok.sum()==0: continue
        rows.append({
            "station": st, "n_valid": int(ok.sum()),
            "MAE": mean_absolute_error(y_true[ok], y_pred[ok]),
            "RMSE": mean_squared_error(y_true[ok], y_pred[ok], squared=False),
            "R2": r2_score(y_true[ok], y_pred[ok]),
        })
    cols = ["station","n_valid","MAE","RMSE","R2"]
    return pd.DataFrame(rows, columns=cols).sort_values("RMSE") if rows else pd.DataFrame(columns=cols)
