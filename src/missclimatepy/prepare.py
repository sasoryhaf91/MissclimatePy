from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional, List

REQUIRED_BASE = ["station", "date", "latitude", "longitude", "elevation"]

def enforce_schema(df: pd.DataFrame, target: str) -> pd.DataFrame:
    need = set(REQUIRED_BASE + [target])
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = df.copy()
    out["station"] = out["station"].astype(str)
    out["date"] = pd.to_datetime(out["date"])
    for c in ["latitude", "longitude", "elevation"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def filter_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)].copy()

def missing_summary(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return (df.groupby("station")[target]
              .agg(total="size",
                   valid=lambda s: s.notna().sum(),
                   missing=lambda s: s.isna().sum(),
                   missing_rate=lambda s: s.isna().mean())
              .reset_index()
              .sort_values("missing_rate", ascending=False))

def select_stations(df: pd.DataFrame, target: str,
                    min_obs: int = 60,
                    stations: Optional[List[str]] = None) -> pd.DataFrame:
    if stations is not None:
        stations = [str(s) for s in stations]
        return df[df["station"].isin(stations)].copy()
    counts = df.groupby("station")[target].apply(lambda s: s.notna().sum())
    keep = counts[counts >= int(min_obs)].index.astype(str)
    return df[df["station"].isin(keep)].copy()
