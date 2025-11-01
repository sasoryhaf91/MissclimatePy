
import pandas as pd
import numpy as np
from missclimatepy import MissClimateImputer

def fake_df(n_stations=3, days=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_stations):
        lat = 15 + 20*rng.random()
        lon = -110 + 20*rng.random()
        elev = 500 + 2000*rng.random()
        dates = pd.date_range("1991-01-01", periods=days, freq="D")
        tmin = 10 + 10*np.sin(2*np.pi*(np.arange(days))/365) + rng.normal(0, 1, days)
        m = rng.random(days) < 0.2
        tmin[m] = np.nan
        for d, v in zip(dates, tmin):
            rows.append({"station": f"S{s:03d}", "date": d,
                         "latitude": lat, "longitude": lon, "elevation": elev,
                         "tmin": v})
    return pd.DataFrame(rows)

def test_rf_fit_transform():
    df = fake_df()
    imp = MissClimateImputer(model="rf", target="tmin", n_estimators=50, n_jobs=-1)
    out = imp.fit_transform(df)
    assert out["tmin"].isna().sum() == 0
    rep = imp.report(out)
    assert {"MAE","RMSE","R2"} <= rep.keys()

def test_idw_fit_transform():
    df = fake_df()
    imp = MissClimateImputer(model="idw", target="tmin")
    out = imp.fit_transform(df)
    assert out["tmin"].isna().sum() == 0
