# tests/test_imputate.py
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd

from missclimatepy.imputate import impute_dataset
from missclimatepy.evaluate import RFParams

def _build_station(station_id: str, start: str, *, periods: int, miss_ratio: float, seed: int = 0):
    """
    Build a single-station synthetic daily series with a sinusoidal target (tmin).
    `miss_ratio` fraction of rows are set to NaN (uniform at random).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="D")

    # Simple seasonal signal + noise
    doy = dates.dayofyear.to_numpy()
    tmin = 10 + 5 * np.sin(2 * np.pi * (doy / 365.25)) + rng.normal(0, 0.5, size=periods)

    # Randomly mask a fraction of rows
    miss_count = int(np.floor(miss_ratio * periods))
    miss_idx = rng.choice(np.arange(periods), size=miss_count, replace=False)
    tmin = pd.Series(tmin, index=np.arange(periods))  # mutable
    tmin.iloc[miss_idx] = np.nan

    df = pd.DataFrame({
        "station": station_id,
        "date": dates,
        "latitude": 19.5,
        "longitude": -99.1,
        "altitude": 2300.0,
        "tmin": tmin.values,
    })
    return df


def test_impute_dataset_basic_large_passes_mdr():
    """
    One large station with enough *observed* rows (>= 1826) must be imputed
    over the whole window. Output must be minimal schema and have no NaNs
    in the target column.
    """
    # 2000 daily rows; choose 5% missing → ~1900 observed >= 1826
    big = _build_station("S_BIG", "2000-01-01", periods=2000, miss_ratio=0.05, seed=1)

    df = big.copy()
    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmin",
        start="2000-01-01",
        end="2005-06-23",  # inclusive → 2001 days
        k_neighbors=None,  # use all other stations; target valid rows are fully included
        rf_params=RFParams(n_estimators=80, max_depth=20, n_jobs=1, random_state=42),
        show_progress=False,
    )

    # Schema checks (minimal)
    assert list(out.columns) == ["station", "date", "latitude", "longitude", "altitude", "tmin", "source"]

    # Full coverage for the window, single station
    expected_len = len(pd.date_range("2000-01-01", "2005-06-23", freq="D"))
    assert len(out) == expected_len
    assert out["station"].nunique() == 1

    # No missing values after imputation
    assert out["tmin"].isna().sum() == 0

    # 'source' must be only 'observed' or 'imputed'
    assert set(out["source"].unique()) <= {"observed", "imputed"}


def test_impute_dataset_mdr_filters_small_station():
    """
    When two stations are provided, only the one that satisfies MDR
    (observed >= 1826 rows in the window) is imputed and appears in output.
    The short station is excluded from the result.
    """
    # Large (passes MDR): 2000 periods, 5% missing → ~1900 observed
    big = _build_station("S_BIG", "2000-01-01", periods=2000, miss_ratio=0.05, seed=2)
    # Small (fails MDR): 120 periods, ~96 observed
    small = _build_station("S_SMALL", "2004-01-01", periods=120, miss_ratio=0.2, seed=3)

    df = pd.concat([big, small], ignore_index=True)

    out = impute_dataset(
        df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="tmin",
        start="2000-01-01",
        end="2005-06-23",
        # MDR uses observed-count >= max(1826, min_station_rows or 0)
        k_neighbors=None,
        rf_params=RFParams(n_estimators=60, max_depth=16, n_jobs=1, random_state=0),
        show_progress=False,
    )

    # Only the big station remains in output
    assert out["station"].nunique() == 1
    assert out["station"].unique()[0] == "S_BIG"

    # Output has full minimal schema and no NaNs in target for the kept station
    assert list(out.columns) == ["station", "date", "latitude", "longitude", "altitude", "tmin", "source"]
    assert out["tmin"].isna().sum() == 0
