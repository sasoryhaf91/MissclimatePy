import numpy as np
import pandas as pd

from missclimatepy.imputate import impute_dataset
from missclimatepy.evaluate import RFParams


def _build_station(station_id: str, start: str, periods: int, freq: str = "D",
                   lat: float = 19.5, lon: float = -99.1, alt: float = 2300.0,
                   miss_ratio: float = 0.1, seed: int = 0):
    """
    Helper: create a synthetic station with some missing target values.
    Ensures tmin is a mutable NumPy array (not a pandas Index) so we can
    assign NaNs for masking.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq=freq)

    # Use NumPy arrays to avoid immutable pandas Index
    doy = dates.dayofyear.to_numpy()
    base = 10.0 + 5.0 * np.sin(2.0 * np.pi * (doy / 365.25))
    noise = rng.normal(0.0, 0.7, size=dates.size)
    tmin = (base + noise).astype(float)

    # inject missing
    n_miss = int(len(dates) * miss_ratio)
    if n_miss > 0:
        miss_idx = rng.choice(len(dates), size=n_miss, replace=False)
        tmin[miss_idx] = np.nan

    df = pd.DataFrame({
        "station": station_id,
        "date": dates,
        "latitude": float(lat),
        "longitude": float(lon),
        "altitude": float(alt),
        "tmin": tmin,
    })
    return df


def test_impute_dataset_basic_large_passes_mdr():
    """
    One large station (> 1825 valid rows) must be imputed over the whole window,
    with no NaNs remaining in the target, and with 'source' marking observed/imputed.
    """
    # 2000 daily rows ~ 5.48 years -> passes MDR
    big = _build_station("S_BIG", "2000-01-01", periods=2000, miss_ratio=0.15, seed=1)

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
        end="2005-06-23",   # exactly 2000 days from start
        k_neighbors=None,   # use all-other stations (none), rely on include_target_pct
        include_target_pct=95.0,
        rf_params=RFParams(n_estimators=50, max_depth=20, n_jobs=1, random_state=42),
        show_progress=False,
    )

    # Schema checks
    assert {"station", "date", "latitude", "longitude", "altitude", "tmin", "source"} <= set(out.columns)

    # Full coverage for the window, single station
    expected_len = len(pd.date_range("2000-01-01", "2005-06-23", freq="D"))
    assert len(out) == expected_len
    assert out["station"].nunique() == 1

    # No missing values after imputation
    assert out["tmin"].isna().sum() == 0

    # Source must contain both observed and imputed (since we injected holes)
    assert set(out["source"].unique()) <= {"observed", "imputed"}
    assert (out["source"] == "imputed").sum() > 0


def test_impute_dataset_mdr_filters_small_station():
    """
    When two stations are provided, only the one that satisfies MDR (>1825) is imputed.
    The short station is filtered out even if the caller passes a small threshold,
    because the function enforces 1826 internally.
    """
    # Large (passes MDR)
    big = _build_station("S_BIG", "2000-01-01", periods=2000, miss_ratio=0.1, seed=2)
    # Small (fails MDR)
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
        # Pass a lenient threshold; function will still enforce 1826
        min_station_rows=10,
        k_neighbors=None,
        include_target_pct=95.0,
        rf_params=RFParams(n_estimators=40, max_depth=16, n_jobs=1, random_state=0),
        show_progress=False,
    )

    # Only the big station remains in output
    assert out["station"].nunique() == 1
    assert out["station"].unique()[0] == "S_BIG"

    # Complete coverage for the output station's requested window
    expected_len = len(pd.date_range("2000-01-01", "2005-06-23", freq="D"))
    assert len(out) == expected_len

    # No missing values after imputation
    assert out["tmin"].isna().sum() == 0
    assert set(out["source"].unique()) <= {"observed", "imputed"}
