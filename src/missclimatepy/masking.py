from __future__ import annotations
import numpy as np
import pandas as pd

def mask_station_fraction(df_station: pd.DataFrame, target: str, frac: float, seed: int = 42) -> pd.DataFrame:
    """Mask a fraction of available target values on a single-station dataframe.
    Adds '<target>_true' with the original values for evaluation."""
    out = df_station.copy()
    rng = np.random.default_rng(seed)
    idx_valid = out[target].dropna().index.to_numpy()
    out[f"{target}_true"] = out[target]
    if len(idx_valid) == 0:
        return out
    n_mask = max(1, int(round(len(idx_valid) * frac)))
    mask_idx = rng.choice(idx_valid, size=n_mask, replace=False)
    out.loc[mask_idx, target] = np.nan
    return out
