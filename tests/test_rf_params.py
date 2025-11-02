# tests/test_rf_params.py
import pandas as pd
import numpy as np
from missclimatepy import MissClimateImputer

def _fake_df():
    dates = pd.date_range("2000-01-01", periods=50)
    df = pd.DataFrame({
        "station": ["S1"]*50,
        "date": dates,
        "latitude": [19.5]*50,
        "longitude": [-99.1]*50,
        "elevation": [2300]*50,
        "tmin": np.sin(np.linspace(0, 6.28, 50)) + 10.0
    })
    df.loc[df.sample(frac=0.3, random_state=0).index, "tmin"] = np.nan
    return df

def test_rf_params_overrides():
    df = _fake_df()
    imp = MissClimateImputer(
        engine="rf",
        target="tmin",
        min_obs_per_station=20,
        n_estimators=10,
        rf_params={"max_depth": 2, "min_samples_leaf": 2},
        random_state=0,
    )
    out = imp.fit_transform(df)
    assert out["tmin"].isna().sum() == 0
