import pandas as pd
import numpy as np
from missclimatepy import MissClimateImputer
from missclimatepy.requirements import estimate_mdr, mdr_curves

def fake_df(n_stations=6, days=90, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stations):
        lat = 15 + 20*rng.random()
        lon = -110 + 20*rng.random()
        elev = 500 + 2000*rng.random()
        dates = pd.date_range("1991-01-01", periods=days, freq="D")
        base = 10 + 8*np.sin(2*np.pi*np.arange(days)/365.0) + rng.normal(0, 1.0, days)
        miss = rng.random(days) < 0.15
        base[miss] = np.nan
        for d, v in zip(dates, base):
            rows.append({"station": f"S{s:03d}", "date": d,
                         "latitude": lat, "longitude": lon, "elevation": elev,
                         "tmin": v})
    return pd.DataFrame(rows)

df = fake_df()

imp = MissClimateImputer(target="tmin", k_neighbors=8, min_obs_per_station=30, n_estimators=200, n_jobs=-1).fit(df)

mdr = estimate_mdr(
    df=df, target="tmin",
    metric_thresholds={"RMSE": 1.5, "R2": 0.4},
    missing_fracs=[0.1, 0.3],
    grid_K=[3,5,8,12],
    grid_min_obs=[20,30,60],
    grid_train_frac=[0.05,0.1,0.2,0.4,0.6],
    random_state=42
)
print(mdr.head())

cur = mdr_curves(
    df=df, target="tmin",
    K=8, min_obs=30,
    train_fracs=[0.05,0.1,0.2,0.4,0.6,0.8],
    missing_frac=0.3,
    random_state=42
)
print(cur.head())

