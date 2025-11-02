# src/missclimatepy/mdr.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Optional, Dict, Any, Tuple
from .evaluate import evaluate_per_station

def inclusion_sweep(df: pd.DataFrame, target: str,
                    include_pcts: Iterable[float]=(0.0,0.04,0.1,0.2,0.4,0.6,0.8),
                    period: Optional[Tuple[str,str]]=None,
                    stations: Optional[Iterable[str]]=None,
                    min_obs: int=60, engine: str="rf",
                    model_params: Optional[Dict[str,Any]]=None) -> pd.DataFrame:
    sub = df.copy()
    if period is not None:
        d0, d1 = period
        sub = sub[(sub["date"]>=d0) & (sub["date"]<=d1)]
    if stations is not None:
        S = set(map(str, stations))
        sub = sub[sub["station"].astype(str).isin(S)]

    rows = []
    for p in include_pcts:
        metr = evaluate_per_station(sub, target, train_frac=float(p),
                                    min_obs=min_obs, engine=engine,
                                    model_params=model_params)
        metr["include_pct"] = float(p)
        rows.append(metr)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def recommend_min_inclusion(sweep_df: pd.DataFrame,
                            thresholds: dict={"R2":0.5,"RMSE":2.0}) -> pd.DataFrame:
    out = []
    for st, g in sweep_df.groupby("station"):
        gg = g.sort_values("include_pct").reset_index(drop=True)
        ok = np.ones(len(gg), dtype=bool)
        if "R2" in thresholds:   
            ok &= (gg["R2"] >= thresholds["R2"])
        if "RMSE" in thresholds: 
            ok &= (gg["RMSE"] <= thresholds["RMSE"])
        out.append({"station": st,
                    "min_include_pct": float(gg.loc[ok,"include_pct"].min()) if ok.any() else np.nan})
    return pd.DataFrame(out)
