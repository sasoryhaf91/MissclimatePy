# tests/test_evaluate.py
import pandas as pd
from missclimatepy.evaluate import evaluate_stations, RFParams

def test_evaluate_stations_runs():
    # Tiny synthetic dataset: 2 stations, short time span
    df = pd.DataFrame({
        "station": ["A"]*5 + ["B"]*5,
        "latitude": [19.0]*10,
        "longitude": [-99.0]*10,
        "altitude": [2300]*10,
        "date": pd.date_range("2001-01-01", periods=5).tolist()*2,
        "tmin": [1.2, 2.1, None, 3.4, 2.9, 5.0, None, 5.5, 6.2, None],
    })

    rep, preds = evaluate_stations(
        df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmin",
        start="2001-01-01",
        end="2001-01-05",
        k_neighbors=1,
        include_target_pct=20,
        rf_params=RFParams(n_estimators=10, max_depth=4, n_jobs=1, random_state=42),
        show_progress=False,
    )

    # Basic checks
    assert not rep.empty
    assert "station" in rep.columns
    assert "R2_d" in rep.columns
    assert not preds.empty
    assert set(["station", "date", "latitude", "longitude", "altitude", "y_obs", "y_mod"]).issubset(preds.columns)
