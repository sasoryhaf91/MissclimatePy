import pandas as pd
import numpy as np
from missclimatepy.neighbors import neighbor_distances
from missclimatepy.evaluate import evaluate_all_stations_fast, RFParams


def _toy_df():
    rng = pd.date_range("2001-01-01", periods=200, freq="D")
    rows=[]
    meta=[("S1",19.0,-99.0,2200,0.0),("S2",19.2,-99.2,2300,1.0),("S3",19.4,-99.4,2400,-0.5)]
    g = np.random.default_rng(0)
    for s,lat,lon,alt,bias in meta:
        y = 12 + 8*np.sin(2*np.pi*np.arange(len(rng))/365.25) + bias + g.normal(0,0.3,len(rng))
        rows += [{"station":s,"date":d,"latitude":lat,"longitude":lon,"altitude":alt,"tmin":v} for d,v in zip(rng,y)]
    return pd.DataFrame(rows)


def test_neighbors_basic():
    toy = pd.DataFrame({"station":["A","B","C"],"latitude":[19.0,19.2,19.4],"longitude":[-99.0,-99.2,-99.4]})
    nd = neighbor_distances(toy, k_neighbors=2, include_self=False)
    assert set(nd.columns) == {"station","neighbor","rank","distance_km"}
    assert len(nd) == 3*2


def test_evaluate_runs_and_metrics():
    df = _toy_df()
    rep = evaluate_all_stations_fast(
        df,
        id_col="station", date_col="date",
        lat_col="latitude", lon_col="longitude", alt_col="altitude",
        target_col="tmin",
        start="2001-01-01", end="2001-07-01",
        k_neighbors=2, include_target_pct=10.0,
        rf_params=RFParams(n_estimators=20, max_depth=12, n_jobs=-1, random_state=1),
        show_progress=False,
    )
    assert {"MAE_d","RMSE_d","R2_d"}.issubset(rep.columns)
    assert len(rep) == 3
