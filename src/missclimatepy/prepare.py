# src/missclimatepy/prepare.py
from __future__ import annotations
import pandas as pd
from typing import Optional
from typing import List

REQUIRED = ["station","date","latitude","longitude","elevation"]

def enforce_schema(df: pd.DataFrame, target: str) -> pd.DataFrame:
    need = set(REQUIRED + [target])
    missing = need - set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    out = df.copy()
    out["station"] = out["station"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    for c in ["latitude","longitude","elevation"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["date"]>=start) & (df["date"]<=end)].copy()

def missing_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    g = df.groupby("station")[target]
    return g.agg(total="size",
                 valid=lambda s: s.notna().sum(),
                 missing=lambda s: s.isna().sum(),
                 missing_rate=lambda s: s.isna().mean()).reset_index()

def select_stations(df: pd.DataFrame, target: str, min_obs: int=60,
                    stations: Optional[List[str]]=None) -> pd.DataFrame:
    if stations is not None:
        return df[df["station"].astype(str).isin([str(s) for s in stations])].copy()
    counts = df.groupby("station")[target].apply(lambda s: s.notna().sum())
    keep = counts[counts>=int(min_obs)].index.astype(str)
    return df[df["station"].isin(keep)].copy()
