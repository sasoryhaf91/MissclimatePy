# Minimum Data Requirement (MDR) protocol with MissClimatePy

This document describes a practical protocol for **Minimum Data Requirement (MDR)** experiments using `MissClimatePy`. The goal is to quantify **how much local station history is needed** (in percentage of observed days) for the XYZT Random Forest models to deliver acceptable interpolation / reconstruction performance.

The workflow is fully reproducible and based on:

- A long-format daily station dataset.
- The station-wise evaluator `evaluate_stations`.
- The missing-data tools in `masking`.
- The neighbor utilities in `neighbors`.
- Standard metrics (MAE, RMSE, R², KGE) computed per station.

The protocol is agnostic to the country or network; here we use “precipitation” (`prec`) or “minimum temperature” (`tmin`) as examples, but any continuous daily variable is compatible.

---

## 1. Data assumptions

MissClimatePy expects a **long-format** daily table with at least:

- `station` – station identifier (string or integer)
- `date` – daily timestamp (datetime-like)
- `latitude` – decimal degrees
- `longitude` – decimal degrees
- `altitude` – meters above sea level
- `<target>` – variable to analyze (e.g. `prec`, `tmin`, `tmax`, `evap`)

You are free to choose the column names; they are passed explicitly to the functions.

```python
import pandas as pd

url = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"
df = pd.read_csv(url, parse_dates=["date"])

df.head()
```

Typical MDR experiments will:

- Fix a **time window**, e.g. 1991-01-01 to 2020-12-31.
- Focus on a **single target variable** (e.g. `prec` or `tmin`).
- Optionally restrict to a **region or subset of stations** (e.g. by ID prefix).

---

## 2. Step 1 – Describe missingness and select stations

Before running MDR experiments, we recommend:

1. **Describing coverage** (percentage of missing values per station).
2. **Describing gap structure** (number and length of consecutive gaps).
3. **Filtering stations** to ensure a reasonable minimum number of observed days.

All of this is handled by `missclimatepy.masking`.

```python
from missclimatepy import masking

START = "1991-01-01"
END   = "2020-12-31"
TARGET = "tmin"   # or "prec", "tmax", "evap"

# 2.1 Coverage: percentage of missing days per station
coverage = masking.percent_missing_between(
    df,
    id_col="station",
    date_col="date",
    target_col=TARGET,
    start=START,
    end=END,
)

# 2.2 Gap profile
gaps = masking.gap_profile_by_station(
    df,
    id_col="station",
    date_col="date",
    target_col=TARGET,
)

# 2.3 Combined descriptive summary
summary = masking.describe_missing(
    df,
    id_col="station",
    date_col="date",
    target_col=TARGET,
    start=START,
    end=END,
)

summary.head()
```

From this summary you can define station filters, e.g.:

- Minimum percentage of **observed** days (e.g. ≥ 60 % coverage).
- Maximum tolerated **maximum gap length** (e.g. ≤ 365 days).
- Region filters (e.g. station IDs starting with `"15"` for State of Mexico).

Example: stations with at least 25 full years of observations in 30 years:

```python
import numpy as np

min_obs_days = 25 * 365  # ~25 years
summary["n_obs_days"] = (1.0 - summary["pct_missing"] / 100.0) * summary["n_days"]
good_stations = summary.loc[summary["n_obs_days"] >= min_obs_days, "station"].tolist()

len(good_stations)
```

This station list can be passed to the evaluator as `station_ids=good_stations`.

---

## 3. Step 2 – Build spatial neighbors (optional but recommended)

MDR experiments benefit from a controlled definition of **spatial neighbors**. MissClimatePy uses **haversine distance** on latitude/longitude and optionally stores altitude differences for diagnostics.

```python
from missclimatepy import neighbors

# One row per station with coordinates
meta = (
    df[["station", "latitude", "longitude", "altitude"]]
    .drop_duplicates("station")
    .reset_index(drop=True)
)

# Tidy table of k nearest neighbors
ndist = neighbors.neighbor_distances(
    meta,
    id_col="station",
    lat_col="latitude",
    lon_col="longitude",
    altitude_col="altitude",
    k_neighbors=20,
    max_radius_km=None,             # or e.g. 150.0
    max_abs_altitude_diff=None,     # or e.g. 800.0
    include_self=False,
)

# Mapping: station -> list of neighbor IDs
neighbor_map = neighbors.build_neighbor_map(
    meta,
    id_col="station",
    lat_col="latitude",
    lon_col="longitude",
    altitude_col="altitude",
    k_neighbors=20,
)

len(neighbor_map)
```

You can either:

- Pass `k_neighbors=20` directly to `evaluate_stations` (let it build the map).
- Or pass `neighbor_map=neighbor_map` for explicit control and reproducibility.

---

## 4. Step 3 – Station-wise MDR experiments with `evaluate_stations`

The MDR protocol is based on **varying the fraction of the target station’s own history** included in the training set, via `include_target_pct`.

Intuition:

- `include_target_pct = 0.0`: **strict interpolation** (LOSO-like). The station is never seen in training; only neighbors contribute.
- Higher values: **local adaptation**. A stratified fraction (month × dry/wet for precipitation) of the station’s observed history is allowed into training, simulating different Minimum Data Requirements.

```python
from missclimatepy import evaluate_stations

mdr_grid = [0.0, 4.0, 8.0, 16.0, 20.0, 40.0, 60.0, 80.0]

all_reports = []

for pct in mdr_grid:
    report, preds = evaluate_stations(
        data=df,
        id_col="station",
        date_col="date",
        lat_col="latitude",
        lon_col="longitude",
        alt_col="altitude",
        target_col=TARGET,
        start=START,
        end=END,
        # station selection
        station_ids=good_stations,      # from Step 1
        # neighborhood (either k_neighbors or neighbor_map)
        k_neighbors=20,
        neighbor_map=None,
        # MDR control: fraction of target history used in training
        include_target_pct=pct,
        include_target_seed=42,
        # model configuration
        model_kind="rf",
        model_params={
            "n_estimators": 200,
            "max_depth": 30,
            "n_jobs": -1,
            "random_state": 42,
        },
        # aggregation for monthly / annual metrics
        agg_for_metrics="sum" if TARGET == "prec" else "mean",
        # baseline (optional; if enabled in your version)
        baseline_kind="mcm_doy",   # mean climatology by day-of-year
        # UX
        show_progress=True,
    )

    report["include_target_pct"] = pct
    all_reports.append(report)

mdr_report = pd.concat(all_reports, ignore_index=True)
mdr_report.head()
```

Each row in `mdr_report` typically contains:

- Station metadata: `station`, `latitude`, `longitude`, `altitude`.
- Training / test sizes: `rows_train`, `rows_test`, `n_rows`.
- Performance metrics:
  - Daily: `MAE_d`, `RMSE_d`, `R2_d`, `KGE_d` (if available).
  - Monthly: `MAE_m`, `RMSE_m`, `R2_m`, `KGE_m`.
  - Annual: `MAE_y`, `RMSE_y`, `R2_y`, `KGE_y`.
- Baseline metrics (suffix `_base`) if a baseline is enabled.
- Configuration: `include_target_pct`, `used_k_neighbors`, timing.

---

## 5. Step 4 – Summarize MDR curves

The core MDR question is:

> *How does performance change as we increase the fraction of local station history in training?*

A standard approach is to:

1. Aggregate metrics by `(include_target_pct, metric)` using medians.
2. Optionally stratify by region, altitude band, climate regime, etc.

```python
import numpy as np

summary_mdr = (
    mdr_report
    .groupby("include_target_pct")
    .agg(
        MAE_d_median=("MAE_d", "median"),
        RMSE_d_median=("RMSE_d", "median"),
        R2_d_median=("R2_d", "median"),
        KGE_d_median=("KGE_d", "median"),
    )
    .reset_index()
)

summary_mdr
```

You can then plot MDR curves, e.g. RMSE vs `include_target_pct`, or compare the Random Forest against the baseline (e.g. `RMSE_d` vs `RMSE_d_base`).

---

## 6. Step 5 – Visual diagnostics

`missclimatepy.viz` provides several ready-made plots to interpret MDR results and model behavior.

### 6.1 Metric distributions

```python
import matplotlib.pyplot as plt
from missclimatepy import viz

# Subset a single MDR level, e.g. 16 %
rep_16 = mdr_report[mdr_report["include_target_pct"] == 16.0]

viz.plot_metrics_distribution(
    rep_16,
    metric_cols=("MAE_d", "RMSE_d", "R2_d"),
    kind="hist",
)
plt.tight_layout()
plt.show()
```

### 6.2 Spatial patterns of performance

```python
viz.plot_spatial_scatter(
    rep_16,
    lat_col="latitude",
    lon_col="longitude",
    value_col="RMSE_d",
    title="Daily RMSE at include_target_pct=16%",
)
plt.show()
```

### 6.3 Time-series overlays or imputed series

Using the `preds` table from a particular MDR level, you can inspect individual stations:

```python
some_station = rep_16["station"].iloc[0]

viz.plot_time_series_overlay(
    preds,
    station_id=some_station,
    id_col="station",
    date_col="date",
    y_true_col="y_obs",
    y_pred_col="y_mod",
    title=f"Observed vs modeled – station {some_station}",
)
plt.show()
```

For fully imputed datasets produced by `impute_dataset` (with a `source` flag):

```python
from missclimatepy.viz import plot_imputed_series

ax = plot_imputed_series(
    df=imputed_df,
    station=some_station,
    id_col="station",
    date_col="date",
    target_col=TARGET,
    source_col="source",
    title=f"{TARGET} – observed vs imputed – station {some_station}",
)
plt.show()
```

---

## 7. Interpreting MDR results

Typical MDR findings include:

- A **baseline performance** at `include_target_pct=0.0`, representing pure spatial interpolation (LOSO-like).
- A **rapid improvement** in MAE/RMSE and KGE as the first 5–20 % of station history is included.
- A **plateau** beyond a certain threshold (e.g. 40–60 %), where additional history yields diminishing returns.
- Differences in MDR curves between regions (e.g. humid vs semi-arid climates, lowland vs highland stations).

These patterns can be:

- Reported as **MDR thresholds**, e.g. “a minimum of 20 % observed days is needed for median KGE ≥ 0.7”.
- Used to **classify stations** into categories (well-constrained vs under-observed).
- Incorporated into **data-quality guidelines** for operational networks and retrospective reconstructions.

---

## 8. Reproducibility notes

To ensure fully reproducible MDR experiments:

- Fix `include_target_seed` for stratified sampling of target rows.
- Fix the `random_state` in the model parameters (e.g. Random Forest).
- Version control:
  - The dataset (e.g. Zenodo DOI).
  - The exact MissClimatePy version (e.g. `v0.1.1`).
  - The MDR grid, station filters, and neighbor configuration.
- Prefer using an explicit `neighbor_map` saved to disk for cross-runs consistency.

This protocol aims to be a **template**: you can adapt it to different periods, variables, regions, or model backends, while keeping the core XYZT and MDR philosophy of `MissClimatePy`.
