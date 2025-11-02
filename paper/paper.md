---
title: "MissclimatePy: A Minimal and Reproducible Local Framework for Imputing Climate Time Series Using Only Spatial and Temporal Predictors"
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
date: 2025-11-01
bibliography: paper.bib
---

# Summary
**MissclimatePy** is a minimal and reproducible framework for climate data imputation using only spatial coordinates (x, y, z) and calendar features (t).
It eliminates dependency on external covariates, gridded data, or auxiliary variables—focusing instead on the
spatio-temporal structure inherent in the observation network.

# Statement of Need
Many meteorological datasets present high missingness, and traditional methods such as IDW or kriging rely
on complete covariates or neighboring values at prediction time. MissclimatePy introduces a *local model*
approach, where Random Forests are trained for each target station using only spatio-temporal coordinates from
neighboring stations, allowing reconstruction even under severe data gaps.

# Features
- Minimal predictors: (x, y, z, t) only.
- Local Random Forest models for each station.
- Synthetic masking experiments to assess robustness.
- Minimum Data Requirement (MDR) estimation to quantify reliability under partial data.
- Full reproducibility: deterministic seeds, unit tests, and CI.

# Implementation
MissclimatePy is structured as independent modules:
- `spatial.py` – neighbor selection using haversine distances.
- `impute.py` – local model training and gap filling.
- `masking.py` – synthetic missingness generation.
- `requirements.py` – MDR estimation and evaluation.
- `api.py` – public interface (`fit`, `transform`, `report`).

The package depends only on `pandas`, `numpy`, and `scikit-learn`, ensuring lightweight reproducibility.

# Example
```python
from missclimatepy import MissClimateImputer
from missclimatepy.requirements import estimate_mdr

imp = MissClimateImputer(target="tmin", k_neighbors=8, min_obs_per_station=30)
imp.fit(df)
df_out = imp.fit_transform(df)

mdr = estimate_mdr(df=df, target="tmin",
                   metric_thresholds={"RMSE":1.5, "R2":0.5},
                   missing_fracs=[0.1,0.3,0.6],
                   grid_K=[3,5,8,12])
```

# Acknowledgements
We thank colleagues and institutions that supported the research and software development.

# References
