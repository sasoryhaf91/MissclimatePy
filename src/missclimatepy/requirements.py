from __future__ import annotations
import pandas as pd
import numpy as np
from .masking import mask_station_fraction
from .impute import impute_local_station
from .metrics import regression_report

def _meets(metrics: dict, thresholds: dict):
    for k, thr in thresholds.items():
        ku = k.upper()
        if ku in ("RMSE","MAE"):
            if metrics.get(k, np.inf) > thr: return False
        elif ku in ("R2","R^2"):
            if metrics.get("R2", -np.inf) < thr: return False
        else:
            if metrics.get(k, np.inf) > thr: return False
    return True

def _nan_report(keys=("MAE","RMSE","R2")):
    return {k: np.nan for k in keys}

def estimate_mdr(
    df: pd.DataFrame,
    target: str,
    metric_thresholds: dict,
    missing_fracs=(0.3,),
    grid_K=(3,5,8,12,15,20),
    grid_min_obs=(30,60,120),
    grid_train_frac=(0.05,0.10,0.20,0.40,0.60,0.80),
    random_state: int = 42,
) -> pd.DataFrame:
    stations = df["station"].unique().tolist()
    rows = []
    for st in stations:
        found = None
        for K in grid_K:
            for mn in grid_min_obs:
                for tf in grid_train_frac:
                    reps = []
                    for mf in missing_fracs:
                        st_df = df[df["station"] == st].copy()
                        st_masked = mask_station_fraction(st_df, target, frac=mf, seed=random_state)
                        imputed, _ = impute_local_station(
                            df_all=pd.concat([df[df.station!=st], st_masked], ignore_index=True),
                            target_station=st,
                            target_var=target,
                            k_neighbors=K,
                            radius_km=None,
                            min_obs_per_station=mn,
                            train_frac=tf,
                            n_estimators=300,
                            n_jobs=-1,
                            random_state=random_state,
                        )
                        if imputed is None:
                            reps = []
                            break
                        merged = imputed.merge(st_masked[["date", f"{target}_true"]], on="date", how="left")
                        msk = merged[f"{target}_true"].notna() & merged[target].notna()
                        if not msk.any():
                            reps = []
                            break
                        y = merged.loc[msk, f"{target}_true"].to_numpy()
                        yhat = merged.loc[msk, target].to_numpy()
                        reps.append(regression_report(y, yhat))
                    if reps:
                        avg = {k: float(np.mean([r[k] for r in reps])) for k in reps[0].keys()}
                        if _meets(avg, metric_thresholds):
                            found = {"station": st, "K": K, "min_obs": mn, "train_frac": tf, **avg}
                            break
                if found: break
            if found: break
        if found is None:
            nanrep = _nan_report()
            found = {"station": st, "K": None, "min_obs": None, "train_frac": None, **nanrep}
        rows.append(found)
    return pd.DataFrame(rows)

def mdr_curves(
    df: pd.DataFrame,
    target: str,
    K: int,
    min_obs: int,
    train_fracs=(0.05,0.10,0.20,0.40,0.60,0.80),
    missing_frac=0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    rows = []
    stations = df["station"].unique().tolist()
    for st in stations:
        st_df = df[df["station"] == st].copy()
        st_masked = mask_station_fraction(st_df, target, frac=missing_frac, seed=random_state)
        for tf in train_fracs:
            imputed, _ = impute_local_station(
                df_all=pd.concat([df[df.station!=st], st_masked], ignore_index=True),
                target_station=st,
                target_var=target,
                k_neighbors=K,
                radius_km=None,
                min_obs_per_station=min_obs,
                train_frac=tf,
                n_estimators=300,
                n_jobs=-1,
                random_state=random_state,
            )
            if imputed is None:
                continue
            merged = imputed.merge(st_masked[["date", f"{target}_true"]], on="date", how="left")
            msk = merged[f"{target}_true"].notna() & merged[target].notna()
            if not msk.any():
                continue
            y = merged.loc[msk, f"{target}_true"].to_numpy()
            yhat = merged.loc[msk, target].to_numpy()
            rep = regression_report(y, yhat)
            rep.update({"station": st, "train_frac": tf})
            rows.append(rep)
    return pd.DataFrame(rows)
