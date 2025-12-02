# MissClimatePy

**Spatial–temporal imputation for daily climate station records using only XYZT (coordinates + calendar features).**

[![CI](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml/badge.svg)](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17794136.svg)](https://doi.org/10.5281/zenodo.17794136)

---

## 1. Overview

**MissClimatePy** is a lightweight Python package for **evaluating and imputing daily climate station records** using only:

- **Spatial coordinates:** latitude, longitude, altitude (X, Y, Z)  
- **Calendar features:** year, month, day-of-year (and optional sinusoidal transforms of day-of-year, T)

The package is designed for situations where:

- You need to **reconstruct complete daily series** (e.g. 1991–2020) at station level.
- You want to avoid dependency on **external covariates** (reanalyses, satellite products, gridded datasets).
- You require **transparent, station-wise diagnostics** suitable for Minimum Data Requirement (MDR) and interpolation studies.

MissClimatePy currently targets **continuous daily variables**, typically:

- precipitation (`prec`)
- minimum temperature (`tmin`)
- maximum temperature (`tmax`)
- evaporation (`evap`)

but any continuous daily variable can be used as the target.

The core entry points are:

- `evaluate_stations` — quantify how well an XYZT model reconstructs daily series for selected stations.
- `impute_dataset` — produce complete daily series with an explicit `"source"` flag (`"observed"` / `"imputed"`).

---

## 2. Installation

MissClimatePy uses a standard `src/` layout.

### From source (recommended while under active development)

```bash
git clone https://github.com/sasoryhaf91/MissclimatePy.git
cd MissclimatePy
pip install -e ".[dev]"
```

Once a PyPI release is available, installation will be simply:

```bash
pip install missclimatepy
```

You can check the installed version with:

```python
import missclimatepy
print(missclimatepy.__version__)
```

---

## 3. Data model

MissClimatePy assumes **daily, long-format climate data**. A minimal schema is:

```text
station    : station identifier (string or integer)
date       : daily timestamp (datetime-like)
latitude   : decimal degrees
longitude  : decimal degrees
altitude   : meters above sea level
<target>   : climate variable to impute (e.g., "prec", "tmin", "tmax", "evap")
```

Column names are passed explicitly to all functions; nothing is hard-coded.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "station":   ["S001"] * 3 + ["S002"] * 3,
    "date":      pd.to_datetime(["1991-01-01", "1991-01-02", "1991-01-03"] * 2),
    "latitude":  [19.5] * 6,
    "longitude": [-99.1] * 6,
    "altitude":  [2300.0] * 6,
    "tmin":      [8.0, None, 7.5, 9.0, None, 8.2],
})
```

---

## 4. Quickstart: impute daily series with XYZT

The simplest workflow is to **impute complete daily series for a subset of stations**.  
The example below uses an open Zenodo dataset of SMN daily stations in Mexico and imputes
minimum temperature for stations located in the **State of Mexico** (prefix `"15"`).

```python
import pandas as pd
import matplotlib.pyplot as plt
from missclimatepy.impute import impute_dataset
from missclimatepy.viz import plot_imputed_series

# 1) Load daily SMN data (1991–2020)
url = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"
df = pd.read_csv(url, parse_dates=["date"])

# 2) Impute tmin for stations starting with "15" (State of Mexico)
tmin_imputed = impute_dataset(
    data=df,
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="tmin",
    start="1991-01-01",
    end="2020-12-31",
    prefix=["15"],                 # State of Mexico stations
    k_neighbors=20,
    include_target_pct=50.0,       # % of each station's own history in training
    min_station_rows=365,          # Minimum observed days to attempt imputation
    model_kind="rf",
    model_params={"n_estimators": 15, "random_state": 42, "n_jobs": -1},
    show_progress=True,
)

# The output has exactly:
# [station, latitude, longitude, altitude, date, tmin, source]
print(tmin_imputed.head())

# 3) Visualise one station (e.g. 15017)
ax = plot_imputed_series(
    df=tmin_imputed,
    station=15017,
    id_col="station",
    date_col="date",
    target_col="tmin",
    source_col="source",
    title="Minimum temperature – imputed series (station 15017)",
)
plt.show()
```

Internally, `impute_dataset`:

1. Builds a daily grid for each selected station between `start` and `end`.
2. Derives calendar features (year, month, day-of-year, optional harmonic terms).
3. Trains a local `RandomForestRegressor` using neighbouring stations (or all others) plus
   an optional fraction of the station’s own history.
4. Predicts values for all days in the grid, **preserving original observations** and
   marking each row as `"observed"` or `"imputed"` in the `source` column.

Stations that do not meet `min_station_rows` or lack a valid training pool are skipped and
not included in the returned table.

---

## 5. Station-wise evaluation

To study how well XYZT-only models interpolate/reconstruct station series under different
conditions (e.g., MDR experiments), use **`evaluate_stations`**.

```python
import pandas as pd
from missclimatepy.evaluate import evaluate_stations

# df: same SMN daily data with tmin, lat, lon, alt, etc.
url = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"
df = pd.read_csv(url, parse_dates=["date"])

report, preds = evaluate_stations(
    data=df,
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="tmin",
    start="1991-01-01",
    end="2020-12-31",
    prefix=["15"],               # optional station filter (e.g., State of Mexico)
    min_station_rows=365,
    k_neighbors=20,
    include_target_pct=0.0,      # 0.0 ~ strict LOSO; >0 introduces controlled leakage
    model_kind="rf",
    model_params={"n_estimators": 15, "random_state": 42, "n_jobs": -1},
    baseline_kind="mcm_doy",     # optional mean-climatology baseline by day-of-year
    agg_for_metrics="mean",       # "sum" for precipitation, "mean" for temperatures
    show_progress=True,
)

print(report.head())
print(preds.head())
```

`report` contains **one row per evaluated station** with:

- training and test sizes,
- number of neighbours used,
- station coordinates and altitude,
- daily, monthly, and annual metrics (`MAE_d`, `RMSE_d`, `R2_d`, `KGE_d`, etc.),
- optional baseline metrics for the climatology model.

`preds` contains all evaluated observations with observed and modelled values, suitable
for plotting or further analysis.

---

## 6. Diagnostics and visualisation

MissClimatePy provides several helper modules:

- `metrics` – MAE, RMSE, $R^2$, KGE, and aggregation helpers (daily → monthly/annual).
- `masking` – coverage summaries, gap statistics, missingness matrices, and deterministic
  masking for controlled experiments.
- `neighbors` – Haversine K-nearest neighbours and ready-to-use neighbour maps.
- `viz` – small matplotlib wrappers for:
  - missingness matrices,
  - metric distributions,
  - parity plots,
  - time-series overlays,
  - spatial performance maps,
  - gap histograms,
  - **imputed series** (`plot_imputed_series`) and **imputation coverage**.

All functions are **schema-agnostic**: you always specify the column names explicitly.

---

## 7. Testing

MissClimatePy ships with a focused test suite based on `pytest`.

From the repository root:

```bash
pip install -e ".[dev]"
pytest
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the tests automatically
for each push and pull request.

---

## 8. Citation

If you use MissClimatePy in your research, please cite the Zenodo record and the
associated JOSS article (once published).

A provisional citation is:

> Antonio-Fernández, H., Vaquera-Huerta, H., Rosengaus-Moshinsky, M. M.,
> Pérez-Rodríguez, P., & Crossa, J. (2025). *MissClimatePy: Spatial–Temporal
> Imputation for Daily Climate Station Records in Python* (Version 0.1.1)
> [Software]. Zenodo. https://doi.org/10.5281/zenodo.17794136

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

---

## 9. License

MissClimatePy is released under the **MIT License**.  
See [`LICENSE`](LICENSE) for the full text.

© 2025 Hugo Antonio Fernández