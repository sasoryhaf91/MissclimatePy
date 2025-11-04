from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_any(path: str, parse_dates=("date",)):
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=list(parse_dates))
from __future__ import annotations
import pandas as pd
from typing import Iterable, Optional, Sequence

__all__ = ["read_csv", "require_columns"]

def require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def read_csv(path: str, *, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Thin wrapper around pandas.read_csv with robust date parsing and
    dtype safety (let pandas infer numerics where possible).
    """
    df = pd.read_csv(path)
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def save_parquet(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
