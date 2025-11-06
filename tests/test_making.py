# tests/test_masking.py
import pandas as pd
from missclimatepy.masking import (
    percent_missing_between,
    gap_profile_by_station,
    missing_matrix,
    describe_missing,
    apply_random_mask_by_station
)

def _toy_df():
    return pd.DataFrame({
        "station": ["S1"]*5 + ["S2"]*5,
        "date": pd.date_range("2000-01-01", periods=5).tolist() * 2,
        "tmin": [1, None, 3, 4, None, 5, 6, None, 8, 9]
    })

def test_percent_missing_between():
    df = _toy_df()
    res = percent_missing_between(df,
                                  id_col="station",
                                  date_col="date",
                                  target_col="tmin",
                                  start="2000-01-01",
                                  end="2000-01-05")
    assert "percent_missing" in res.columns
    assert (res["percent_missing"].between(0, 100)).all()

def test_gap_profile_by_station():
    df = _toy_df()
    out = gap_profile_by_station(df,
                                 id_col="station",
                                 date_col="date",
                                 target_col="tmin")
    assert set(["station", "n_gaps", "mean_gap"]).issubset(out.columns)

def test_missing_matrix_and_describe():
    df = _toy_df()
    mat = missing_matrix(df, id_col="station", date_col="date", target_col="tmin")
    assert mat.values.dtype == "uint8"
    desc = describe_missing(df, id_col="station", date_col="date", target_col="tmin")
    assert "coverage" in desc.columns

def test_apply_random_mask_reproducibility():
    df = _toy_df()
    masked1 = apply_random_mask_by_station(df,
                                           id_col="station",
                                           date_col="date",
                                           target_col="tmin",
                                           percent_to_mask=20,
                                           random_state=123)
    masked2 = apply_random_mask_by_station(df,
                                           id_col="station",
                                           date_col="date",
                                           target_col="tmin",
                                           percent_to_mask=20,
                                           random_state=123)
    assert masked1.equals(masked2)
