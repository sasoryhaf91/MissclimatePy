# MissClimatePy

**MissClimatePy** is a lightweight Python package for **imputing and evaluating daily climate station records** using only:

- Spatial coordinates: **latitude, longitude, altitude**  
- Temporal information: **year, month, day-of-year** (optionally sine/cosine of day-of-year)

It enforces a simple **XYZT** representation:

> **X–Y**: latitude–longitude  **Z**: elevation  **T**: time (calendar features)

and intentionally ignores external covariates (reanalyses, satellite products, etc.). This makes workflows **reproducible, transparent, and easy to port** to other station networks and variables.

MissClimatePy is designed for:

- Reconstructing **complete daily series** (e.g. 1991–2020) at station level  
- Studying **Minimum Data Requirement (MDR)** scenarios  
- Evaluating how far a station network can be interpolated using **only its own geometry and calendar structure**

---

## Key features

- **Global XYZT imputer**  
  - `MissClimateImputer` fits a single `RandomForestRegressor` on all observed rows.  
  - Uses only `[lat, lon, alt, year, month, doy]` (plus optional cyclic `sin/cos(doy)`).
  - Produces a filled data frame and basic diagnostics (MAE, RMSE, R²) on observed rows.

- **Station-wise evaluation**  
  - `evaluate_stations` trains one model per station using **K-nearest spatial neighbours** (Haversine distance) or all other stations.  
  - Optional controlled leakage via `include_target_pct` for MDR experiments.  
  - Returns a **station report** (daily, monthly, yearly metrics; train/test sizes; neighbours) and a **prediction table**.

- **Local daily series reconstruction**  
  - `impute_dataset` builds a full daily grid for each selected station and window.  
  - Preserves original observations and fills gaps with model predictions.  
  - Output schema:  
    ```text
    [station, date, latitude, longitude, altitude, <target>, source]
    ```
    where `source` is `"observed"` or `"imputed"`.

- **Missing-data diagnostics**  
  - Coverage, gap profiles, missingness matrices, and deterministic masking scenarios (`masking` module).

- **Spatial neighbours**  
  - Haversine-based neighbour distances and reusable neighbour maps (`neighbors` module).

- **Visualisation helpers**  
  - Missingness matrices, metric distributions, parity plots, time-series overlays, spatial maps, and imputed-series plots (`viz` module).

MissClimatePy currently targets **daily continuous variables** such as precipitation (`prec`), minimum temperature (`tmin`), maximum temperature (`tmax`), and evaporation (`evap`), but any scalar daily variable can be used as the target.

---

## Installation

From a clone of the repository:

```bash
git clone https://github.com/sasoryhaf91/MissclimatePy.git
cd MissclimatePy
pip install .
```

For development (recommended if you want to run tests or edit the code/paper):

```bash
pip install -e ".[dev]"
pytest
```

---

## Minimal example: local daily imputation

The snippet below shows how to reconstruct daily minimum temperature (`tmin`) for stations in the State of Mexico (IDs starting with `"15"`) using only coordinates and calendar features:

```python
import pandas as pd
import matplotlib.pyplot as plt
from missclimatepy.impute import impute_dataset as impute
from missclimatepy.viz import plot_imputed_series as pis

url = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"
df = pd.read_csv(url, parse_dates=["date"])

# Impute tmin for stations whose ID starts with "15"
tmin_imputed = impute(
    df,
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="tmin",
    start="1991-01-01",
    end="2020-12-31",
    prefix=["15"],
    model_kind="rf",
    model_params={"n_estimators": 15, "random_state": 42, "n_jobs": -1},
)

# Visualise one station (e.g. 15017)
ax = pis(
    df=tmin_imputed,
    station=15017,
    id_col="station",
    date_col="date",
    target_col="tmin",
    source_col="source",
    title="Minimum temperature imputed – station 15017",
)

plt.show()
```

The resulting data frame `tmin_imputed` contains, for each selected station and day:

```text
[station, date, latitude, longitude, altitude, tmin, source]
```

where `source` indicates whether the value is observed or imputed.

---

## Documentation roadmap

Use the navigation bar to explore:

- **Quickstart** – step-by-step examples for the main workflows.  
- **Data model** – required columns and recommended preprocessing.  
- **Global imputer** – details and examples for `MissClimateImputer`.  
- **Station-wise evaluation** – MDR and interpolation experiments with `evaluate_stations`.  
- **Local imputation** – full daily reconstruction with `impute_dataset`.  
- **Missing-data diagnostics** – coverage, gaps, and masking.  
- **Neighbours & visualisation** – spatial KNN utilities and plotting helpers.  
- **API reference** – summary of the public functions and classes.

If you use MissClimatePy in your research, please see the **Citation** section in the project README or `CITATION.cff`.




