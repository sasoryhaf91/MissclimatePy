
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__all__ = ["safe_metrics", "aggregate_and_score"]

_FREQ_ALIAS = {"M": "ME", "A": "YE", "Y": "YE", "Q": "QE"}

def safe_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R² safely (R²=NaN when n<2 or zero variance).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    if y_true.size >= 2 and float(np.var(y_true)) > 0.0:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def aggregate_and_score(
    df_pred: pd.DataFrame,
    *,
    date_col: str,
    y_col: str = "y_true",
    yhat_col: str = "y_pred",
    freq: str = "M",
    agg: str = "sum",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Aggregate daily predictions to `freq` and compute metrics on overlap.

    Parameters
    ----------
    df_pred : DataFrame with [date_col, y_col, yhat_col]
    freq : Pandas offset alias; aliases M->ME, A/Y->YE, Q->QE to avoid warnings
    agg  : {"sum","mean","median"}

    Returns
    -------
    metrics_dict, aggregated_df
    """
    freq = _FREQ_ALIAS.get(freq, freq)
    aggfunc = {"sum": "sum", "mean": "mean", "median": "median"}[agg]

    df = df_pred[[date_col, y_col, yhat_col]].dropna(subset=[date_col]).copy()
    if df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, df

    df = df.set_index(date_col).sort_index()
    agg_df = df.resample(freq).agg({y_col: aggfunc, yhat_col: aggfunc}).dropna()
    if agg_df.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}, agg_df

    m = safe_metrics(agg_df[y_col].values, agg_df[yhat_col].values)
    return m, agg_df
