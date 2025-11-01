
import pandas as pd
import numpy as np

def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if not np.issubdtype(out[date_col].dtype, np.datetime64):
        out[date_col] = pd.to_datetime(out[date_col])
    doy = out[date_col].dt.dayofyear.astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    out["year"] = out[date_col].dt.year.astype(int)
    return out
