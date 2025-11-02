from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_any(path: str, parse_dates=("date",)):
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=list(parse_dates))

def save_parquet(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
