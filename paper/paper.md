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

Daily climate station records often contain large gaps due to sensor failures, irregular maintenance, and missing archives. In many national networks, missing fractions of 40–70 % are common for key variables such as precipitation and minimum and maximum temperature—inputs that underpin climate diagnostics and impact studies [@Morice2021; @Huang2024; @Dunn2023]. Gridded satellite and reanalysis products can help, but spatial smoothing and methodological breaks often make them unsuitable for station-level reconstructions [@Cornes2018; @Hersbach2020].

Traditional interpolation methods, such as inverse-distance weighting and kriging, rely on distributional assumptions and relatively dense networks, and their performance degrades under sparse or highly incomplete observations [@LI2011228]. Modern machine-learning approaches offer greater flexibility [@Reichstein2019; @YANG2024120797], but usually depend on external covariates from remote sensing or reanalyses [@Beck2020; @Freitas2025], which may be unavailable, inconsistent, or hard to reproduce.

**MissClimatePy** is a lightweight Python package for evaluating and imputing daily station records using only station metadata—latitude, longitude, elevation—and calendar features (year, month, day-of-year, optional harmonic transforms). It enforces a simple XYZT representation (X–Y: latitude–longitude, Z: elevation, T: time) and intentionally ignores additional covariates, focusing on what can be learned from the station network itself. The package provides station-wise model evaluation, local Random-Forest-based imputation of target series, neighbour-based training pools, and diagnostics tailored to Minimum Data Requirement (MDR) and interpolation studies.

# Statement of Need

Missing data remain a major limitation of meteorological networks, especially in regions with limited resources or complex topography [@Huang2024]. Many imputation frameworks require external datasets such as satellite precipitation, vegetation indices, or gridded temperatures, which are often unavailable historically or poorly representative of local station behaviour. Daily station archives also rarely include rich multivariate covariates per timestamp, which limits the applicability of multivariate schemes such as MICE or MissForest [@Stekhoven2012]. MissClimatePy instead offers a covariate-independent, transparent, and reproducible station-level tool that scales to archives with thousands of stations and multiple decades while using only local metadata plus calendar structure. It exposes per-station diagnostics suitable for MDR and interpolation studies, allowing users to quantify how far they can go in reconstructing station series relying only on XYZT features. Although motivated by national networks in Mexico, the package is general and can be applied to any climate network where station coordinates and dates are available.

# Functionality and Implementation

MissClimatePy operates on long-format `pandas` tables and is organised in a few small submodules. The `features` utilities parse datetimes and build calendar predictors; `neighbors` precomputes K-nearest-neighbour sets in geographic space; and `models` provides a compact registry that wraps scikit-learn regressors under a single factory, with random forests as the default. All models work on dense numeric XYZT matrices.

Station-wise evaluation is implemented in `evaluate_stations`. For each target station, the function builds a training pool from all other stations or a KNN subset, applies a user-defined time window and Minimum Data Requirement filter, optionally includes a fraction of the station’s own valid rows, fits the chosen backend, and returns multi-scale error metrics together with a prediction table containing observed and modelled values.

Local imputation of a single target variable is handled by `impute_dataset`. For each selected station, it trains a `RandomForestRegressor` using neighbouring stations (or all others) plus an optional portion of its observed history. It then builds a daily grid over the requested window, preserves observed values, fills gaps with predictions, and tags each row as observed or imputed.

Diagnostics are provided by the `metrics`, `masking`, and `viz` modules. `metrics` implements standard error scores (MAE, RMSE, $R^2$, KGE) and aggregation helpers; and `viz` offers thin `matplotlib` wrappers for missingness matrices, metric distributions, time-series overlays, spatial summaries of performance, and imputation coverage plots.

# Example

The following snippet illustrates a typical workflow using an open Zenodo dataset of daily SMN stations in Mexico [@AntonioFernandez2025SMN]. In this example, MissClimatePy imputes stations located in the State of Mexico using only coordinates and calendar information.


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

# References
