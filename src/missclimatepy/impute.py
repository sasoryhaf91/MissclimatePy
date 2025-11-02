# src/missclimatepy/impute.py
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _ensure_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not np.issubdtype(out["date"].dtype, np.datetime64):
        out["date"] = pd.to_datetime(out["date"])
    out["doy"] = out["date"].dt.dayofyear.astype(int)
    out["year"] = out["date"].dt.year.astype(int)
    return out


def _feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df.loc[:, feature_cols].astype(float).values
    return X


def fit_transform_local_rf(
    df: pd.DataFrame,
    target: str,
    k_neighbors: int = 8,
    min_obs_per_station: int = 60,
    n_estimators: int = 300,
    n_jobs: int = -1,
    random_state: int = 42,
    rf_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Impute missing values per station using a local RandomForestRegressor trained
    on (x, y, z, t) features. If you already implement spatial neighbor selection,
    you can integrate it here; otherwise this is a minimal robust baseline that
    trains on each station's available data (and can be later extended to include
    spatio-temporal neighbors).

    Parameters
    ----------
    df : DataFrame
        Requires columns: station, date, latitude, longitude, elevation, <target>.
    rf_params : dict | None
        Extra params to override RandomForestRegressor defaults, e.g.
        {"max_depth": 20, "min_samples_leaf": 2}.

    Returns
    -------
    DataFrame with the same schema as `df` and target imputed where possible.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in df.")

    # final output
    out = df.copy()
    out = _ensure_calendar_features(out)

    feature_cols = ["latitude", "longitude", "elevation", "doy", "year"]

    # base RF params
    rf_kwargs = dict(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    if rf_params:
        rf_kwargs.update(rf_params)

    # Validate RF kwargs early with a dry instantiation
    try:
        _ = RandomForestRegressor(**rf_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Invalid RandomForestRegressor parameters in rf_params={rf_params}. "
            f"Original error: {e}"
        )

    # Iterate per station
    stations = out["station"].astype(str).unique()
    for st in stations:
        sub_idx = out["station"].astype(str) == str(st)
        sub = out.loc[sub_idx].copy()

        # training rows: target not null
        train_mask = sub[target].notna()
        n_train = int(train_mask.sum())
        if n_train < int(min_obs_per_station):
            # not enough data to train a local model
            continue

        # matrices
        X_train = _feature_matrix(sub.loc[train_mask], feature_cols)
        y_train = sub.loc[train_mask, target].astype(float).values

        pred_mask = ~train_mask
        if pred_mask.any():
            X_pred = _feature_matrix(sub.loc[pred_mask], feature_cols)

            model = RandomForestRegressor(**rf_kwargs)
            model.fit(X_train, y_train)
            y_hat = model.predict(X_pred)

            # write back
            sub.loc[pred_mask, target] = y_hat
            out.loc[sub_idx, target] = sub[target]

    return out
