# MissClimatePy

**Minimal and reproducible framework for climate–data imputation using only spatial coordinates (x, y, z) and calendar features (t).**

[![CI](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml/badge.svg)](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/TBA.svg)](https://doi.org/TBA)

---

## 1. Overview

**MissClimatePy** is a lightweight Python package for **imputing missing daily climate records** using only:

- **Spatial coordinates:** latitude (x), longitude (y), altitude (z)  
- **Temporal descriptors:** year, month, day‑of‑year (and optional harmonic sin/cos of day‑of‑year)

It is designed for situations where:

- You want to **reconstruct entire daily climate series** (e.g., 1991–2020) at station level.
- You **do not** want to depend on external covariates (reanalyses, gridded data, satellite products).
- You need **transparent, reproducible, station‑wise evaluation** based solely on (t, x, y, z).

Currently the package focuses on **regression problems** such as:

- Daily precipitation (`prec`)
- Minimum temperature (`tmin`)
- Maximum temperature (`tmax`)
- Evaporation (`evap`)

but any continuous daily variable can be used as the target.

---

## 2. What MissClimatePy actually does

The package exposes a **small, explicit API** – no hidden magic, no global state.

### 2.1 High‑level imputer

**`MissClimateImputer`** (`missclimatepy.api` / exported at top‑level):

- Fits **one global `RandomForestRegressor`** on all rows where the target is observed.
- Uses the minimal feature set:

  ```text
  [latitude, longitude, altitude, year, month, doy] (+ doy_sin, doy_cos if requested)
  ```

- Imputes missing values in a tidy, long‑format `pandas.DataFrame`.
- Provides a simple **diagnostic report** (MAE, RMSE, R²) on observed rows.

This is the main entry point when you want to **fill gaps in a full dataset**.

---

### 2.2 Station‑wise evaluation

**`evaluate_stations`** (`missclimatepy.evaluate` / exported at top‑level):

- Trains **one Random Forest per station**.
- Uses either:
  - All *other* stations as training pool, or
  - Only the **K nearest neighbors** in (latitude, longitude) via Haversine distance.
- Allows **controlled inclusion** of the target station into the training set via
  `include_target_pct`:

  - `include_target_pct = 0.0` → strict LOSO‑like evaluation (no leakage).
  - `include_target_pct > 0.0` → a stratified fraction of the station’s own valid rows
    is included in training (month × dry/wet for precipitation).

- Reports metrics at **three time scales**:
  - Daily (`MAE_d`, `RMSE_d`, `R2_d`)
  - Monthly aggregated (`MAE_m`, `RMSE_m`, `R2_m`)
  - Annual aggregated (`MAE_y`, `RMSE_y`, `R2_y`)

This function is what we use to **quantify how well the imputer interpolates / reconstructs
each station**, including experiments with different minimum data requirements.

---

### 2.3 Missing‑data diagnostics and masking

`missclimatepy.masking` provides utilities to **describe and simulate missingness**:

- `percent_missing_between` – percentage of missing days per station in a fixed window
  (e.g., 1991‑01‑01 to 2020‑12‑31).
- `gap_profile_by_station` – number, mean and maximum length of consecutive missing runs.
- `missing_matrix` – station × date matrix (1 = observed, 0 = missing).
- `describe_missing` – combined coverage + gap summary per station.
- `apply_random_mask_by_station` – deterministically mask a given percentage of values
  per station, useful for **controlled experiments**.

These tools help you **select stations** with enough information and **design masking
schemes** for validation.

---

### 2.4 Spatial neighbors

`missclimatepy.neighbors` exposes Haversine‑based neighbor search:

- `neighbor_distances` – tidy table of each station’s k nearest neighbors, with
  great‑circle distance in kilometers and optional altitude differences.
- `build_neighbor_map` – a convenient

  ```python
  {station_id: [neighbor_id_1, neighbor_id_2, ...]}
  ```

  mapping, ready for per‑station modeling loops.

`evaluate_stations` internally uses a similar Haversine KNN strategy; these helpers are
available when you want to **inspect or customize the neighborhood structure** explicitly.

---

### 2.5 Visualization helpers

`missclimatepy.viz` provides small plotting utilities built on **matplotlib**:

- `plot_missing_matrix` – heatmap‑like missingness matrix (stations × dates).
- `plot_metrics_distribution` – distributions of MAE/RMSE/R² across stations.
- `plot_parity_scatter` – observed vs modeled scatter with 1:1 line.
- `plot_time_series_overlay` – single‑station time series with optional model overlay.
- `plot_spatial_scatter` – scatter of stations colored by a performance metric.
- `plot_gap_histogram` – distribution of gap lengths (e.g. max gap per station).
- `plot_imputed_series` – observed vs imputed points for a single station
  (from an imputed dataset with a `source` column).
- `plot_imputation_coverage` – per‑station share of imputed values.

All functions are **schema‑agnostic**: you specify the column names.

---

## 3. Installation

MissClimatePy is a standard Python package with a `src/` layout.

Clone the repository:

```bash
git clone https://github.com/sasoryhaf91/MissclimatePy.git
cd MissclimatePy
```

Install for regular use:

```bash
pip install .
```

Or for development (recommended while preparing results / JOSS submission):

```bash
pip install -e ".[dev]"
```

This installs the package under the name:

```python
import missclimatepy
print(missclimatepy.__version__)
```

---

## 4. Data model

MissClimatePy assumes **daily, long‑format climate data**.  
A minimal schema looks like:

```text
station   : station identifier (string or integer)
date      : daily timestamp (datetime‑like)
latitude  : decimal degrees
longitude : decimal degrees
altitude  : meters above sea level
target    : climate variable to impute (e.g., "prec", "tmin", "tmax", "evap")
```

You are free to choose the column names; they are passed explicitly to the functions and
classes.

Example:

```python
import pandas as pd

df = pd.DataFrame({
    "station":  ["S001"] * 3 + ["S002"] * 3,
    "date":     pd.to_datetime(["1991-01-01", "1991-01-02", "1991-01-03"] * 2),
    "latitude": [19.5] * 6,
    "longitude": [-99.1] * 6,
    "altitude": [2300.0] * 6,
    "tmin":     [8.0, None, 7.5, 9.0, None, 8.2],
})
```

---

## 5. Quickstart: global RF imputation

The simplest way to use MissClimatePy is via **`MissClimateImputer`**.

```python
import pandas as pd
from missclimatepy import MissClimateImputer

# Example data frame (long format)
df = pd.DataFrame({
    "station":   ["S001"] * 3 + ["S002"] * 3,
    "date":      pd.to_datetime(["1991-01-01", "1991-01-02", "1991-01-03"] * 2),
    "latitude":  [19.5] * 6,
    "longitude": [-99.1] * 6,
    "altitude":  [2300.0] * 6,
    "tmin":      [8.0, None, 7.5, 9.0, None, 8.2],
})

imp = MissClimateImputer(
    target="tmin",
    # optional: schema if different from defaults
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    # feature configuration
    add_cyclic=False,      # set True to add sin/cos(doy)
    # RF hyper‑parameters
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

# Fit on observed rows, then impute missing ones
df_filled = imp.fit_transform(df)

# Diagnostic metrics on **observed rows only**
metrics = imp.report(df)
print(metrics)
# {'MAE': ..., 'RMSE': ..., 'R2': ...}
```

Under the hood:

- The imputer computes `year`, `month`, `doy` (and optionally `doy_sin`, `doy_cos`).
- Fits a `RandomForestRegressor` on rows where the target is not `NaN`.
- Predicts only where the target is missing and all required features are available.

---

## 6. Station‑wise evaluation (LOSO + leakage experiments)

To study how robust the imputer is under different levels of station coverage, you can
use **`evaluate_stations`**.

A typical use case (for example, daily temperature 1991–2020):

```python
import pandas as pd
from missclimatepy import RFParams, evaluate_stations

# df: long-format daily data for many stations and years
# Columns: station, date, latitude, longitude, altitude, tmin

rf = RFParams(
    n_estimators=200,
    max_depth=30,
    min_samples_leaf=1,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
)

report_loso, preds_loso = evaluate_stations(
    data=df,
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="tmin",
    start="1991-01-01",
    end="2020-12-31",
    # neighborhood: 20 nearest neighbors by Haversine on (lat, lon)
    k_neighbors=20,
    # strict leave‑one‑station‑out: no target data in training
    include_target_pct=0.0,
    rf_params=rf,
    agg_for_metrics="sum",   # sum for precipitation; for temperature, "mean" is also common
    show_progress=True,
)

print(report_loso.head())
print(preds_loso.head())
```

To study a **minimum data requirement (MDR)** experiment, you can repeat the evaluation
for different values of `include_target_pct` (e.g., 0, 4, 8, 16, 20, 40, 60, 80) and
compare metrics such as `MAE_d`, `RMSE_d`, `R2_d` across stations.

Example loop:

```python
results = []
for pct in (0, 4, 8, 16, 20, 40, 60, 80):
    rep, _ = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col="prec",
        start="1991-01-01",
        end="2020-12-31",
        k_neighbors=20,
        include_target_pct=float(pct),
        rf_params=rf,
        show_progress=False,
    )
    rep["include_target_pct"] = pct
    results.append(rep)

full_report = pd.concat(results, ignore_index=True)
```

This is the pattern used to produce the per‑variable summaries for **precipitation,
tmax, tmin, and evaporation**: the same station set, the same temporal window
(e.g., 1991–2020), and varying levels of station‑specific training inclusion.

---

## 7. Missing‑data description and masking

Before imputation or evaluation, you may want to **select stations** by coverage
and understand the structure of the gaps. The `masking` module is designed for this.

```python
from missclimatepy import masking

# 1) Coverage between 1991-01-01 and 2020-12-31
cov = masking.percent_missing_between(
    df,
    id_col="station",
    date_col="date",
    target_col="prec",
    start="1991-01-01",
    end="2020-12-31",
)
print(cov.head())

# 2) Gap profile per station
gaps = masking.gap_profile_by_station(
    df,
    id_col="station",
    date_col="date",
    target_col="prec",
)
print(gaps.head())

# 3) Combined description (coverage + gaps)
summary = masking.describe_missing(
    df,
    id_col="station",
    date_col="date",
    target_col="prec",
    start="1991-01-01",
    end="2020-12-31",
)
print(summary.head())

# 4) Apply random masking per station (e.g., mask 20% of existing values)
df_masked = masking.apply_random_mask_by_station(
    df,
    id_col="station",
    date_col="date",
    target_col="prec",
    percent_to_mask=20.0,
    random_state=123,
)
```

These functions are useful both for **pre‑selection** of stations and for
designing **synthetic missingness** experiments.

---

## 8. Spatial neighbors

To inspect or customize spatial neighborhoods explicitly, use the `neighbors` module.

```python
from missclimatepy import neighbors

# Station metadata: one row per station
meta = (
    df[["station", "latitude", "longitude", "altitude"]]
    .drop_duplicates("station")
    .reset_index(drop=True)
)

# Tidy table of neighbor distances
ndist = neighbors.neighbor_distances(
    meta,
    id_col="station",
    lat_col="latitude",
    lon_col="longitude",
    altitude_col="altitude",
    k_neighbors=10,
    max_radius_km=None,             # or e.g. 150.0
    max_abs_altitude_diff=None,     # or e.g. 500.0
    include_self=False,
)
print(ndist.head())

# Dict mapping station -> list of neighbor ids
nmap = neighbors.build_neighbor_map(
    meta,
    id_col="station",
    lat_col="latitude",
    lon_col="longitude",
    altitude_col="altitude",
    k_neighbors=10,
)
print(nmap["S001"])
```

This structure is compatible with the `neighbor_map` argument of `evaluate_stations`
if you prefer to **control neighbors yourself** instead of letting the evaluator
build them internally.

---

## 9. Visualization examples

A few examples using the `viz` module:

```python
import matplotlib.pyplot as plt
from missclimatepy import viz

# 1) Missingness matrix for precipitation
viz.plot_missing_matrix(
    df,
    id_col="station",
    date_col="date",
    target_col="prec",
    max_stations=40,
)
plt.show()

# 2) Metric distributions from station-wise evaluation
viz.plot_metrics_distribution(
    report_loso,                # output of evaluate_stations
    metric_cols=("MAE_d", "RMSE_d", "R2_d"),
    kind="hist",
)
plt.show()

# 3) Observed vs modeled parity scatter
viz.plot_parity_scatter(preds_loso, y_true_col="y_obs", y_pred_col="y_mod")
plt.show()

# 4) Time series overlay for one station
some_station = report_loso["station"].iloc[0]
viz.plot_time_series_overlay(
    preds_loso,
    station_id=some_station,
    id_col="station",
    date_col="date",
    y_true_col="y_obs",
    y_pred_col="y_mod",
)
plt.show()
```

For imputed datasets that carry a `"source"` column with values `"observed"` /
`"imputed"`, you can highlight which points were filled:

```python
viz.plot_imputed_series(
    df_filled_with_source,
    station="S001",
    id_col="station",
    date_col="date",
    target_col="tmin",
    source_col="source",
)
plt.show()
```

---

## 10. Testing and CI

MissClimatePy ships with a small but focused test suite.

Run tests locally from the repository root:

```bash
pip install -e ".[dev]"
pytest
```

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs the tests automatically
for each push and pull request.

---

## 11. Scope and limitations

To keep the framework **simple and transparent**, MissClimatePy intentionally:

- Uses **Random Forests only** (from `scikit‑learn`) for both the global imputer and
  station‑wise evaluation.
- Relies **only** on (t, x, y, z) features; no external covariates are used internally.
- Targets **daily, continuous variables**. Discrete variables or other temporal resolutions
  are outside the current scope.

Extensions such as additional models (e.g., MLPs, hybrids) or integration with external
gridded products can be built *on top* of the same data model and evaluation scheme, but
are **not** part of this package at the moment.

---

## 12. Citation

If you use MissClimatePy in your research, please cite the Zenodo record and/or the
associated article once published.

A provisional citation could be:

> Antonio-Fernández, H., Vaquera-Huerta, H., Rosengaus-Moshinsky, M. M., Pérez-Rodríguez, P., & Crossa, J.  (2025). *MissClimatePy: Minimal station‑wise imputation of daily
> climate records using spatial coordinates and calendar features (t, x, y, z only).*  
> Version 0.1.0. Zenodo. https://doi.org/10.5281/zenodo.TBA

See [`CITATION.cff`](CITATION.cff) for machine‑readable metadata.

---

## 13. License

MissClimatePy is released under the **MIT License**.  
See [`LICENSE`](LICENSE) for the full text.

© 2025 Hugo Antonio Fernández
