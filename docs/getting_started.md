# Getting started with MissClimatePy

MissClimatePy is a lightweight Python package for **evaluating and imputing daily climate station records** using only:

- **Spatial coordinates**: latitude, longitude, altitude  
- **Temporal features**: year, month, day-of-year (and optional harmonic features)

It is designed for **long station archives** (multi-decadal daily series) where you want to:

- Reconstruct **complete daily series** (e.g. 1991–2020) at station level.
- Avoid dependence on external covariates (reanalysis, satellite, gridded products).
- Work in a **transparent, reproducible XYZT framework**.

This page shows how to install the package, prepare your data, and run the two main workflows:

- **Station-wise evaluation**: `evaluate_stations`
- **Local station imputation**: `impute_dataset`

---

## 1. Installation

### 1.1 From source (recommended for now)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/sasoryhaf91/MissClimatePy.git
cd MissClimatePy

# create and activate your virtual env as you prefer, then:
pip install -e ".[dev]"
```

Check that the package imports correctly:

```python
import missclimatepy as mcp
print(mcp.__version__)
```

As soon as the package is on PyPI, you will also be able to do:

```bash
pip install missclimatepy
```

---

## 2. Data model

MissClimatePy works with **daily, long-format tables**. The minimal schema is:

| Column      | Type                          | Description                                     |
|------------|-------------------------------|-------------------------------------------------|
| `station`  | string or integer             | Station identifier                              |
| `date`     | datetime64 (daily)            | Daily timestamp                                 |
| `latitude` | float                          | Latitude in decimal degrees                     |
| `longitude`| float                          | Longitude in decimal degrees                    |
| `altitude` | float                          | Elevation above sea level (meters)              |
| `<target>` | float (with NaNs allowed)     | Climate variable to impute (e.g. `tmin`)        |

You can use **any column names** you like: you will pass them explicitly to the functions.

Example toy dataset:

```python
import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        "station":  ["S1"] * 5 + ["S2"] * 5,
        "date":     pd.date_range("2020-01-01", periods=5, freq="D").tolist() * 2,
        "latitude": [19.5] * 5 + [19.7] * 5,
        "longitude":[-99.1] * 5 + [-99.3] * 5,
        "altitude": [2300.0] * 10,
        "tmin":     [8.0, np.nan, 7.5, np.nan, 7.8,
                     9.0, 8.7, np.nan, 8.9, np.nan],
    }
)
```

---

## 3. Station-wise evaluation with `evaluate_stations`

Use `evaluate_stations` when you want to **quantify model performance** at each station:

- It trains **one model per station**.
- The training pool uses **other stations** (or only K nearest neighbours).
- You can choose:
  - the time window (`start` / `end`),
  - the Minimum Data Requirement (`min_station_rows`),
  - the fraction of the target station’s own history to include in training (`include_target_pct`),
  - the regression backend (`model_kind`, `model_params`).

### 3.1 Minimal example (synthetic data)

```python
import pandas as pd
from missclimatepy.evaluate import evaluate_stations

# Reuse the toy df from above
df["date"] = pd.to_datetime(df["date"])

report, preds = evaluate_stations(
    data=df,
    # schema
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="tmin",
    # time window
    start="2020-01-01",
    end="2020-01-05",
    # station selection (here: all)
    prefix=None,
    min_station_rows=3,     # at least 3 observed tmin per station
    # neighbours and leakage
    k_neighbors=1,          # 1 nearest neighbour (Haversine on lat/lon)
    include_target_pct=0.0, # pure LOSO-like: no target rows in training
    # model backend
    model_kind="rf",
    model_params={"n_estimators": 50, "random_state": 42, "n_jobs": -1},
    # metrics aggregation
    agg_for_metrics="mean",
    # UX
    show_progress=True,
)

print("Station-level report:")
print(report)

print("
Row-level predictions:")
print(preds.head())
```

What you get:

- `report`: one row per station, with metrics and metadata, e.g.

  - `MAE_d`, `RMSE_d`, `R2_d` (daily)
  - `MAE_m`, `RMSE_m`, `R2_m` (monthly aggregates)
  - `MAE_y`, `RMSE_y`, `R2_y` (annual aggregates)
  - `used_k_neighbors`, `include_target_pct`, `rows_train`, `rows_test`
  - `latitude`, `longitude`, `altitude` (station medoids)

- `preds`: one row per evaluated observation with columns like:

  - `station`, `date`, `latitude`, `longitude`, `altitude`
  - `y_obs`, `y_mod` (observed vs modelled target)

This is the core tool for **MDR experiments**, leave-one-station-out tests, and neighbour-based interpolation studies.

---

## 4. Local imputation with `impute_dataset`

Use `impute_dataset` when you want to **build complete daily series** for a subset of stations over a given period:

- It calibrates a **local model per station**, using neighbours (or all others) plus, optionally, a fraction of the station’s own observed history.
- It builds a **full daily date grid** for each station in the requested window.
- It returns **only the imputed view**, with a consistent schema:

  ```text
  [station, date, latitude, longitude, altitude, <target>, source]
  ```

  where `source` is `"observed"` or `"imputed"`.

### 4.1 Minimal example

```python
import pandas as pd
from missclimatepy.impute import impute_dataset

df["date"] = pd.to_datetime(df["date"])

filled = impute_dataset(
    data=df,
    # schema
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="tmin",
    # window for the series to reconstruct
    start="2020-01-01",
    end="2020-01-05",
    # which stations to impute (here: all)
    prefix=None,
    min_station_rows=1,        # allow stations with at least 1 observed value
    # neighbourhood
    k_neighbors=1,
    include_target_pct=50.0,   # include 50% of the station’s own observed history
    include_target_seed=42,
    # model backend
    model_kind="rf",
    model_params={"n_estimators": 50, "random_state": 42, "n_jobs": -1},
    # UX
    show_progress=True,
)

print(filled)
```

Typical output (simplified):

```text
  station       date  latitude  longitude  altitude  tmin   source
0      S1 2020-01-01      19.5     -99.1    2300.0   8.0  observed
1      S1 2020-01-02      19.5     -99.1    2300.0   7.9  imputed
2      S1 2020-01-03      19.5     -99.1    2300.0   7.5  observed
3      S1 2020-01-04      19.5     -99.1    2300.0   7.7  imputed
4      S1 2020-01-05      19.5     -99.1    2300.0   7.8  imputed
...
```

Key points:

- There is **one row per day per station** in the requested window.
- Observed values are preserved (`source="observed"`).
- Previously missing or non-existing days are filled (`source="imputed"`).
- Stations that **cannot be trained** (no neighbours, or too few observed values) are skipped and do **not** appear in the output.

---

## 5. Quick visual check of an imputed series

MissClimatePy includes a small `viz` module. One of the most useful plots is `plot_imputed_series`, which overlays observed and imputed values for a single station.

```python
import matplotlib.pyplot as plt
from missclimatepy.viz import plot_imputed_series

ax = plot_imputed_series(
    df=filled,
    station="S1",
    id_col="station",
    date_col="date",
    target_col="tmin",
    source_col="source",
    title="Minimum temperature – station S1",
)

plt.show()
```

By default, the function:

- Plots observed points and imputed points with different markers/alpha.
- Adds a legend explaining the `source` categories.
- Uses your specified `title`.

This is particularly helpful for **sanity checks** and for creating figures for reports or papers.

---

## 6. Next steps

Once you are comfortable with these basics, you can explore:

- **`docs/mdr_protocol.md`**  
  For step-by-step Minimum Data Requirement (MDR) experiments.
- **`docs/api.md`**  
  For a compact overview of functions and arguments.
- **`masking` utilities**  
  To describe and simulate missingness patterns before evaluation or imputation.
- **`neighbors` utilities**  
  To inspect and customise spatial neighbourhoods.

MissClimatePy is intentionally minimalist: everything is driven by **explicit arguments** and **long-format tables**, so it should be straightforward to integrate into larger workflows or reproduce full MDR and interpolation studies.
