---
title: "MissClimatePy: Spatial–Temporal Imputation for Daily Climate Station Records in Python"
tags:
  - Python
  - meteorology
  - climatology
  - reproducible research
  - missing data
  - machine learning
authors:
  - name: "Hugo Antonio-Fernández"
    orcid: "0000-0002-5355-8476"
    affiliation: "1, 2"
  - name: "Humberto Vaquera-Huerta"
    orcid: "0000-0002-2805-804X"
    affiliation: 1
  - name: "Moisés Michel Rosengaus-Moshinsky"
    affiliation: 3
  - name: "Paulino Pérez-Rodríguez"
    orcid: "0000-0002-3202-1784"
    affiliation: 1
  - name: "José Crossa"
    orcid: "0000-0001-9429-5855"
    affiliation: 4
affiliations:
  - name: "Colegio de Postgraduados, México"
    index: 1
  - name: "Universidad Mexiquense del Bicentenario, México"
    index: 2
  - name: "Independent Consultant"
    index: 3
  - name: "CIMMYT, México"
    index: 4
date: 12/01/2025
bibliography: paper.bib
---

# Summary

Daily climate station records often contain large gaps due to sensor failures, irregular maintenance, and decades of operational changes. In many national networks, missing fractions of 40–70 % are common for key variables such as precipitation and minimum and maximum temperature—inputs that underpin climate diagnostics, hydrology, agriculture, and climatological normals [@Morice2021; @Huang2024; @Dunn2023]. Gridded satellite and reanalysis products help, but spatial smoothing and methodological breaks often make them unsuitable for station-level reconstructions [@Cornes2018; @Hersbach2020].

Traditional interpolation methods, such as inverse-distance weighting and kriging, rely on distributional assumptions and relatively dense networks, and their performance degrades under sparse or highly incomplete observations [@LI2011228]. Modern machine-learning approaches offer greater flexibility [@Reichstein2019; @YANG2024120797], but usually depend on external covariates from remote sensing or reanalyses [@Beck2020; @Freitas2025], which may be unavailable, inconsistent, or hard to reproduce over long archives.

**MissClimatePy** is a lightweight Python package for evaluating and imputing daily station records using only station metadata—latitude, longitude, elevation—and calendar features (year, month, day-of-year, optional harmonic transforms). It enforces a simple XYZT representation (X–Y: latitude–longitude, Z: elevation, T: time) and intentionally ignores additional covariates, focusing on what can be learned from the station network itself. The package provides station-wise model evaluation, local Random-Forest-based imputation of target series, neighbour-based training pools, and diagnostics and visualisations tailored to Minimum Data Requirement (MDR) and interpolation studies.


# Statement of Need

Missing data remain a major limitation of meteorological networks, especially in regions with limited resources or complex topography [@Huang2024]. Many imputation frameworks require external datasets such as satellite precipitation, vegetation indices, or gridded temperatures, which are often unavailable historically or poorly representative of local station behaviour. Daily station archives also rarely include rich multivariate covariates per timestamp, which limits the applicability of multivariate imputation schemes such as MICE or MissForest [@Stekhoven2012]. MissClimatePy instead offers a covariate-independent, transparent, and reproducible station-level tool that scales to archives with thousands of stations and multiple decades while using only local metadata plus calendar structure and exposing per-station diagnostics suitable for MDR and interpolation studies.

# Functionality and Implementation

MissClimatePy is organised in small, focused submodules that operate directly on long-format `pandas` tables. The `features` utilities handle datetime parsing and calendar feature engineering, while remaining schema-agnostic through explicit column-name arguments. The `neighbors` utilities build spatial neighbour maps using haversine distance on latitude–longitude so that K-nearest-neighbour sets can be reused across workflows.

Model backends are provided by a compact registry in `models`, which wraps scikit-learn regressors under a single factory. Random Forests are the default, but alternative tree ensembles, k-nearest neighbours, multilayer perceptrons, support-vector regression, and linear models are also available. All operate on dense numeric XYZT feature matrices and share a common configuration interface.

The core station-wise evaluation protocol is implemented in `evaluate_stations` (module `evaluate`). For each target station, the function builds a training pool from all other stations or a KNN neighbour subset, applies a user-defined time window and a Minimum Data Requirement filter based on the number of observed target values, and optionally includes a controlled fraction of the station’s own valid rows via a precipitation-friendly stratified sampler over month × dry/wet conditions. It then fits the chosen backend and computes daily, monthly, and annual metrics on a held-out test set, returning (i) a station-level report with metrics, timing, training/test sizes, neighbour information, and median coordinates, and (ii) a prediction table for all evaluated observations.

Local imputation of a single target variable is handled by `impute_dataset` (module `impute`). For each selected station, it trains a `RandomForestRegressor` using neighbouring stations (or all others) plus an optional fraction of its own observed history. A full daily grid is created over the requested window, predictions are generated for all days, and observed values are preserved whenever available. The function returns a tidy table with exactly `[station, date, latitude, longitude, altitude, <target>, source]`, where `source` is `"observed"` or `"imputed"`, and can save results to CSV or Parquet, optionally partitioned by station.

Diagnostics are provided by `metrics`, `masking`, and `viz`. Metrics include mean absolute error (MAE), root mean square error (RMSE), coefficient of determination ($R^2$), and Kling–Gupta efficiency (KGE). Masking helpers compute percentage-missing profiles and gap statistics, build station × date missingness matrices, and generate deterministic masking scenarios for controlled experiments. Visualisation helpers produce missingness matrices, metric distributions, parity plots, time-series overlays, spatial performance maps, gap histograms, and imputation coverage charts, all returning matplotlib `Axes` objects. MissClimatePy depends only on `numpy`, `pandas`, `scikit-learn`, and `matplotlib`.


## Example

The following snippet illustrates a typical workflow using an open Zenodo dataset of daily SMN stations in Mexico [@AntonioFernandez2025SMN]. In this example, MissClimatePy imputes the stations located in the State of Mexico using only coordinates and calendar information; all other columns are ignored.

```python
import pandas as pd
import matplotlib.pyplot as plt
from missclimatepy.impute import impute_dataset as impute
from missclimatepy.viz import plot_imputed_series as pis

url = "https://zenodo.org/records/17636066/files/smn_mx_daily_1991_2020.csv"
df = pd.read_csv(url, parse_dates=["date"])

tmin_imputed = impute(
    df, id_col="station", date_col="date",
    lat_col="latitude", lon_col="longitude", alt_col="altitude",
    target_col="tmin", start="1991-01-01", end="2020-12-31",
    prefix = ["15"], model_kind="rf", 
    model_params={"n_estimators": 15, "random_state": 42, "n_jobs": -1},
)

ax = pis(
    df=tmin_imputed,station=15017, id_col="station", 
    date_col="date", target_col="tmin", source_col="source",
    title="Minimum temperature imputed – station 15017",
)

plt.show()
```
![Example of daily minimum temperature reconstruction for station 15017 (State of Mexico). Original observations are preserved, gaps are filled by the XYZT Random Forest model, and the `source` flag differentiates observed vs imputed days.](figures/imputed_tmin_15017.png){#fig-imputed-series}

# Related Work

Recent developments in climate data reconstruction have expanded the use of machine learning, geostatistical–AI hybrids, and deep-learning architectures [@Reichstein2019; @YANG2024120797]. Traditional interpolation techniques remain indispensable but degrade rapidly under sparse station density or high missingness [@LI2011228], and multivariate imputation algorithms such as MICE or MissForest assume the presence of multiple covariates per timestamp [@Stekhoven2012], which daily station archives rarely provide. MissClimatePy instead restricts its predictor space to XYZT only (coordinates and calendar structure), offering a reproducible way to test how far one can go in interpolating and reconstructing daily station records using only the station network itself.

# Acknowledgements

This work was supported by the Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI) through a doctoral scholarship to the first author. We acknowledge Colegio de Postgraduados and Universidad Mexiquense del Bicentenario for institutional support. We also thank the International Maize and Wheat Improvement Center (CIMMYT) for fostering collaboration in open climate and agricultural research.
