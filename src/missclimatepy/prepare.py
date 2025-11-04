from __future__ import annotations
import pandas as pd
from typing import Dict

__all__ = ["missing_summary"]

def missing_summary(
    df: pd.DataFrame,
    *,
    station_col: str,
    date_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Per-station missingness summary for the target variable.

    Returns
    -------
    DataFrame with: [station, n_rows, n_missing, pct_missing]
    """
    g = (df[[station_col, target_col]]
         .assign(_isna=lambda d: d[target_col].isna())
         .groupby(station_col)
         .agg(n_rows=(target_col, "size"), n_missing=("_isna", "sum"))
         .reset_index())
    g["pct_missing"] = g["n_missing"] / g["n_rows"]
    return g
