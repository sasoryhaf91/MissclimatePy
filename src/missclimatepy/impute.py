from __future__ import annotations
import pandas as pd
import numpy as np
from .features import add_calendar_features
from .spatial import select_neighbors
from .models.rf import RFRegressor

X_COLS = ["latitude","longitude","elevation","doy_sin","doy_cos","year"]

def impute_local_station(
    df_all: pd.DataFrame,
    target_station: str,
    target_var: str,
    k_neighbors: int | None = 15,
    radius_km: float | None = None,
    min_obs_per_station: int = 50,
    train_frac: float = 1.0,
    n_estimators: int = 300,
    n_jobs: int = -1,
    random_state: int = 42,
):
    df = df_all.copy()
    df["date"] = pd.to_datetime(df["date"])

    meta = (df.groupby("station")
              .agg(latitude=("latitude","first"),
                   longitude=("longitude","first"),
                   elevation=("elevation","first"),
                   n_obs=(target_var, lambda s: s.notna().sum()))
              .reset_index())

    if (meta.station == target_station).sum() == 0:
        return None, None

    st_row = meta.loc[meta.station == target_station].iloc[0]
    neigh = select_neighbors(meta, st_row, k_neighbors=k_neighbors, radius_km=radius_km)
    neigh = neigh[neigh.n_obs >= min_obs_per_station]

    df_train = df[df.station.isin(neigh.station)].copy()
    df_train = df_train[df_train[target_var].notna()]
    if df_train.empty:
        return None, neigh

    if 0 < train_frac < 1.0:
        df_train = df_train.sample(frac=train_frac, random_state=random_state)

    df_train = add_calendar_features(df_train, "date")
    X = df_train[X_COLS].to_numpy()
    y = df_train[target_var].to_numpy()

    model = RFRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state).fit(X, y)

    df_target = df[df.station == target_station].copy()
    df_target = add_calendar_features(df_target, "date")
    miss_mask = df_target[target_var].isna()
    if miss_mask.any():
        Xmiss = df_target.loc[miss_mask, X_COLS].to_numpy()
        yhat = model.predict(Xmiss)
        df_target.loc[miss_mask, target_var] = yhat

    return df_target, neigh
