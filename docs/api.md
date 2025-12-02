# API reference

MissClimatePy works on **daily, long-format climate tables** and exposes a small,
explicit set of functions oriented to:

- **Station-wise evaluation** of XYZT models, and  
- **Local imputation** of a single target variable per call.

Throughout the API you always pass **column names explicitly**, so the package
does not enforce a fixed schema.

---

## Data model

All core functions assume a long-format `pandas.DataFrame` with at least:

- `id_col` – station identifier (string or integer)
- `date_col` – daily timestamp (datetime-like)
- `lat_col` – station latitude in decimal degrees
- `lon_col` – station longitude in decimal degrees
- `alt_col` – station elevation (meters above sea level)
- `target_col` – climate variable to model/impute (e.g. `prec`, `tmin`, `tmax`, `evap`)

Minimal example:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "station":   ["S001"] * 3 + ["S002"] * 3,
        "date":      pd.to_datetime(["1991-01-01", "1991-01-02", "1991-01-03"] * 2),
        "latitude":  [19.5] * 6,
        "longitude": [-99.1] * 6,
        "altitude":  [2300.0] * 6,
        "tmin":      [8.0, None, 7.5, 9.0, None, 8.2],
    }
)
```

---

## Core evaluation: `evaluate_stations`

```python
from missclimatepy import evaluate_stations
```

`evaluate_stations` performs **station-wise model evaluation** using an XYZT
feature set:

- X–Y: latitude–longitude  
- Z: elevation  
- T: year, month, day-of-year (and optional harmonic transforms of DOY)

For each target station, it:

1. Builds a **training pool** from:
   - All other stations, or
   - A KNN subset defined by Haversine distance (lat–lon), or
   - A custom `neighbor_map`.
2. Applies an optional **Minimum Data Requirement** based on the number of
   non-missing target values in the chosen window.
3. Optionally includes a **fraction of the station’s own history**
   (`include_target_pct`) using a precipitation-friendly stratified sampler
   (month × dry/wet) when the target is rainfall.
4. Fits the chosen model backend (default: Random Forest).
5. Computes **daily, monthly, and annual metrics** on a held-out test set.
6. Optionally computes a **baseline** (Mean Climatology Model) to compare
   against the learned model.

### Signature (simplified)

```python
report, preds = evaluate_stations(
    data,
    *,
    # required schema
    id_col,
    date_col,
    lat_col,
    lon_col,
    alt_col,
    target_col,
    # temporal window
    start=None,
    end=None,
    # feature config
    add_cyclic=False,
    feature_cols=None,
    # station selection
    prefix=None,
    station_ids=None,
    regex=None,
    custom_filter=None,
    min_station_rows=None,
    # neighborhoods
    k_neighbors=20,
    neighbor_map=None,
    # leakage
    include_target_pct=0.0,
    include_target_seed=42,
    # model backend
    model_kind="rf",
    model_params=None,
    # baseline + metrics
    baseline_kind="mcm_doy",
    agg_for_metrics="sum",
    # UX / logging
    show_progress=False,
    log_csv=None,
    flush_every=20,
    # saving
    save_report_path=None,
    parquet_compression="snappy",
)
```

### Important parameters

- **Station selection**
  - `prefix`: list of string prefixes; keeps stations whose id starts with
    any of them (e.g. `["15"]` for State of Mexico in SMN codes).
  - `station_ids`: explicit list of station ids to evaluate.
  - `regex`: regular expression on station ids.
  - `custom_filter`: `callable(sid) → bool` to keep or discard ids.
  - All filters have **OR semantics**; if none are provided, all stations are
    considered.

- **Minimum Data Requirement**
  - `min_station_rows`: minimum number of **observed** target values in the
    `[start, end]` window. Stations below this threshold are skipped.

- **Neighborhood**
  - `k_neighbors`: if provided and `neighbor_map` is `None`, a Haversine KNN
    neighbor map is constructed internally from per-station median coordinates.
  - `neighbor_map`: custom precomputed mapping
    `{station_id: [neighbor_id_1, ...]}`. When given, it overrides
    `k_neighbors`.

- **Leakage / inclusion**
  - `include_target_pct`:
    - `0.0` → strict **LOSO-like** evaluation: no target rows in training.
    - `> 0` → include that percentage of the station’s own valid rows in
      training (using month × dry/wet stratification for rainfall).
  - `include_target_seed`: random seed for the stratified sampler.

- **Model backend**
  - `model_kind`: currently supports (exact set may expand):
    - `"rf"` – Random Forest (default)
    - `"etr"` – Extra Trees
    - `"gbr"` – Gradient Boosting
    - `"ridge"` – Ridge regression
    - `"svr"` – Support-Vector Regression
    - `"knn"` – K-Nearest Neighbors
    - `"mlp"` – Multi-layer Perceptron
  - `model_params`: dict of keyword arguments forwarded to the corresponding
    scikit-learn regressor (e.g. `{"n_estimators": 200, "max_depth": 30}`).

- **Baseline**
  - `baseline_kind`:
    - `"mcm_doy"` – **Mean Climatology Model**: for each station and day-of-year
      (1–366), take the mean of observed values across years; the test-day
      prediction is the corresponding DOY mean.
    - `None` – skip baseline computation.
  - Baseline metrics are reported alongside model metrics.

- **Metrics aggregation**
  - `agg_for_metrics`: `"sum"`, `"mean"`, or `"median"`:
    - `"sum"` is natural for precipitation (monthly / annual totals).
    - `"mean"` or `"median"` are typical for temperature.

### Return values

- `report`: `DataFrame` with one row per evaluated station, including:

  - Station id and metadata:
    - `station`
    - `latitude`, `longitude`, `altitude` (median coordinates in window)
    - `rows_train`, `rows_test`, `n_rows` (test rows used for metrics)
    - `seconds` (wall-clock time per station)
    - `used_k_neighbors`, `include_target_pct`
  - **Model metrics**:
    - `MAE_d`, `RMSE_d`, `R2_d` – daily metrics
    - `MAE_m`, `RMSE_m`, `R2_m` – monthly aggregated metrics
    - `MAE_y`, `RMSE_y`, `R2_y` – annual aggregated metrics
  - **Baseline metrics** (when `baseline_kind` is not `None`):
    - `MAE_d_base`, `RMSE_d_base`, `R2_d_base`
    - `MAE_m_base`, `RMSE_m_base`, `R2_m_base`
    - `MAE_y_base`, `RMSE_y_base`, `R2_y_base`

- `preds`: per-row predictions for the test splits, including:

  - `station`, `date`, `latitude`, `longitude`, `altitude`
  - `y_obs` – observed values
  - `y_mod` – model predictions
  - `y_base` – baseline predictions (if applicable)

---

## Core imputation: `impute_dataset`

```python
from missclimatepy import impute_dataset
```

`impute_dataset` performs **local imputation** of a single target variable:

- One **local model per station** (same XYZT feature set as `evaluate_stations`).
- Neighbors defined either by KNN (Haversine) or an explicit `neighbor_map`.
- Uses an optional `include_target_pct` to bring a fraction of the station’s
  own history into training.
- Generates a **complete daily grid** `[start, end]` for each imputed station.
- Preserves original observations and fills gaps with model predictions.
- Marks each day as `"observed"` or `"imputed"` via a `source` column.

### Signature (simplified)

```python
imputed = impute_dataset(
    data,
    *,
    id_col,
    date_col,
    lat_col,
    lon_col,
    alt_col,
    target_col,
    start=None,
    end=None,
    add_cyclic=False,
    feature_cols=None,
    prefix=None,
    station_ids=None,
    regex=None,
    custom_filter=None,
    min_station_rows=None,
    k_neighbors=20,
    neighbor_map=None,
    include_target_pct=0.0,
    include_target_seed=42,
    model_kind="rf",
    model_params=None,
    show_progress=False,
    save_table_path=None,
    parquet_compression="snappy",
)
```

The **station-selection**, **MDR**, **neighborhood**, **leakage**, and
**model_kind/model_params** arguments behave exactly as in `evaluate_stations`.

### Behavior

For each selected station `sid`:

1. Selects the station’s rows inside `[start, end]` and ensures a daily grid
   of dates in that window (even if the original table has gaps or entire dates
   missing).
2. Computes XYZT features for all rows where they can be constructed.
3. Builds a training pool from neighbors (or all other stations) plus an
   optional fraction of the station’s own observed history (`include_target_pct`).
4. If no valid training pool exists (e.g. no neighbors and no leakage), the
   station is **skipped** and does not appear in the output.
5. Otherwise, fits the chosen model and predicts the target for **all days**
   in the grid:
   - Where the original target is not missing → the original value is kept.
   - Where the original target is missing or absent → the model prediction is
     used.
6. Annotates each row with a `source` label:
   - `"observed"` – original non-missing observation.
   - `"imputed"` – filled by the model.

### Return value

A tidy `DataFrame` with **only the stations that could be imputed**, with columns:

- `station`
- `date`
- `latitude`
- `longitude`
- `altitude`
- `<target_col>` – imputed series
- `source` – `"observed"` or `"imputed"`

The rows are restricted to `[start, end]`. Stations that do not meet
`min_station_rows` or lack a valid training pool are silently omitted.

### Simple example

```python
import pandas as pd
from missclimatepy import impute_dataset

url = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"
df = pd.read_csv(url, parse_dates=["date"])

tmin_imputed = impute_dataset(
    df,
    id_col="station",
    date_col="date",
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    target_col="tmin",
    start="1991-01-01",
    end="2020-12-31",
    prefix=["15"],  # e.g. State of Mexico in SMN coding
    k_neighbors=20,
    include_target_pct=50.0,
    min_station_rows=365,
    model_kind="rf",
    model_params={"n_estimators": 200, "random_state": 42, "n_jobs": -1},
    show_progress=True,
)

print(tmin_imputed.head())
```

---

## Feature utilities: `features`

```python
from missclimatepy import features
```

Main user-facing helpers:

### `ensure_datetime_naive`

```python
from missclimatepy.features import ensure_datetime_naive

df["date"] = ensure_datetime_naive(df["date"])
```

- Parses a Series to `datetime64[ns]` and drops any timezone information.

### `add_calendar_features`

```python
from missclimatepy.features import add_calendar_features

df_with_time = add_calendar_features(
    df,
    date_col="date",
    add_cyclic=True,  # add sin/cos(doy)
)
```

Adds:

- `year` (int32)
- `month` (int16)
- `doy` – day-of-year (int16)
- If `add_cyclic=True`:
  - `doy_sin`, `doy_cos` – harmonic transforms of DOY (`2π * doy / 365.25`)

### `assemble_feature_columns`

```python
from missclimatepy.features import assemble_feature_columns

feat_cols = assemble_feature_columns(
    lat_col="latitude",
    lon_col="longitude",
    alt_col="altitude",
    add_cyclic=True,
    extra=["custom_feature_1"],
)
```

Returns the list of feature column names used for XYZT modeling.

### Station selection helpers

These are mostly used internally, but can be useful when building custom
workflows:

- `select_station_ids(data, id_col, prefix=None, station_ids=None, regex=None, custom_filter=None)`  
  → returns a **list** of station ids selected by the same OR semantics used in
  `evaluate_stations` / `impute_dataset`.

- `filter_by_min_station_rows(data, id_col, target_col, min_station_rows)`  
  → returns the subset of station ids with at least `min_station_rows` non-missing
  target values.

---

## Metrics: `metrics`

```python
from missclimatepy import metrics
```

Core functions:

### `compute_all_metrics`

```python
from missclimatepy.metrics import compute_all_metrics

scores = compute_all_metrics(y_true, y_pred, include_kge=True)
# -> {"MAE": ..., "RMSE": ..., "R2": ..., "KGE": ...}
```

Implements:

- MAE – mean absolute error
- RMSE – root mean square error
- R² – coefficient of determination
- KGE – Kling–Gupta Efficiency (optional)

All metrics are robust to empty or degenerate inputs; in such cases they return
`np.nan`.

### `aggregate_and_compute`

```python
from missclimatepy.metrics import aggregate_and_compute

metrics_monthly, df_monthly = aggregate_and_compute(
    df_pred,
    date_col="date",
    y_col="y_obs",
    yhat_col="y_mod",
    freq="M",
    agg="sum",
    include_kge=True,
)
```

- Resamples the series to a given `freq` (`"M"`, `"YS"`, `"Q"`, etc.).
- Applies an aggregation (`"sum"`, `"mean"`, `"median"`).
- Computes metrics between the aggregated observed and predicted series.

`evaluate_stations` uses this helper internally for monthly and annual metrics.

---

## Missing-data diagnostics: `masking`

```python
from missclimatepy import masking
```

Main helpers:

- `percent_missing_between(df, id_col, date_col, target_col, start, end)`  
  → percentage of missing days per station in a given window.

- `gap_profile_by_station(df, id_col, date_col, target_col)`  
  → number of gaps, mean and maximum gap length per station.

- `missing_matrix(df, id_col, date_col, target_col, start=None, end=None)`  
  → station × date matrix of 1 (observed) / 0 (missing).

- `describe_missing(df, id_col, date_col, target_col, start=None, end=None)`  
  → combined summary of coverage and gap statistics per station.

- `apply_random_mask_by_station(df, id_col, date_col, target_col, percent_to_mask, random_state=None)`  
  → deterministically mask a given fraction of existing values per station;
    useful for controlled validation experiments.

These tools are typically used **before** evaluation/imputation to select stations
and design synthetic missingness scenarios.

---

## Spatial neighbors: `neighbors`

```python
from missclimatepy import neighbors
```

Key functions:

### `neighbor_distances`

```python
ndist = neighbors.neighbor_distances(
    meta,
    id_col="station",
    lat_col="latitude",
    lon_col="longitude",
    altitude_col="altitude",
    k_neighbors=10,
    max_radius_km=None,
    max_abs_altitude_diff=None,
    include_self=False,
)
```

- `meta` should have one row per station.
- Returns a tidy table with columns like:
  - `station`, `neighbor`
  - `distance_km`
  - `altitude_diff`

### `build_neighbor_map`

```python
nmap = neighbors.build_neighbor_map(
    meta,
    id_col="station",
    lat_col="latitude",
    lon_col="longitude",
    altitude_col="altitude",
    k_neighbors=10,
    max_radius_km=None,
    max_abs_altitude_diff=None,
    include_self=False,
)
```

- Returns a dict `{station_id: [neighbor_id_1, ...]}`.
- Compatible with the `neighbor_map` argument of both
  `evaluate_stations` and `impute_dataset`.

---

## Visualisation: `viz`

```python
from missclimatepy import viz
```

Most functions return a `matplotlib.axes.Axes` instance so that you can further
customise them.

Common helpers:

- `plot_missing_matrix(df, id_col, date_col, target_col, max_stations=40, ax=None)`
- `plot_metrics_distribution(report, metric_cols=("MAE_d", "RMSE_d", "R2_d"), kind="hist", ax=None)`
- `plot_parity_scatter(df_pred, y_true_col="y_obs", y_pred_col="y_mod", ax=None)`
- `plot_time_series_overlay(df_pred, station_id, id_col, date_col, y_true_col, y_pred_col, ax=None)`
- `plot_spatial_scatter(meta, metric_col, lat_col="latitude", lon_col="longitude", ax=None)`
- `plot_gap_histogram(gaps, gap_col="max_gap", ax=None)`
- `plot_imputed_series(df, station, id_col, date_col, target_col, source_col, ax=None, title=None)`
- `plot_imputation_coverage(df, id_col, source_col="source", ax=None)`

Example:

```python
import matplotlib.pyplot as plt
from missclimatepy.viz import plot_imputed_series

ax = plot_imputed_series(
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

---

This reference is intentionally compact. For more narrative examples and
recommended MDR workflows, see:

- `README.md` (high-level overview + quickstart)
- `docs/mdr_protocol.md` (Minimum Data Requirement experiments)
- The JOSS paper in `paper/paper.md` (motivation and design rationale)
