---
title: "MissClimatePy: Spatial–Temporal Random-Forest Imputation for Daily Climate Station Records"
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
date: 2025-11-16
bibliography: paper.bib
---

# Summary

Daily climate station records are frequently affected by substantial missingness resulting from sensor degradation, irregular maintenance, adverse weather conditions, and inconsistencies accumulated across decades of observation. In many national networks, gaps of 40–70% are common, particularly for precipitation, minimum temperature, and maximum temperature—variables essential for climate diagnostics, hydrological modelling, agricultural decision-making, and the estimation of climatological normals [@Morice2021; @Huang2024; @Dunn2023]. Although satellite- and reanalysis-based gridded products exist, they often introduce spatial smoothing, temporal discontinuities, or methodological shifts that hinder their use for station-level historical reconstructions [@Cornes2018; @Hersbach2020].

Traditional interpolation methods—such as inverse-distance weighting, kriging, splines, or linear regression—depend on distributional assumptions and dense networks, limiting their performance in sparse or highly incomplete observational settings [@LI2011228]. Modern machine-learning approaches, including Random Forests, gradient boosting, and deep neural networks, offer greater flexibility [@Reichstein2019; @YANG2024120797], but they typically rely on auxiliary predictors drawn from remote sensing, land-surface models, or reanalysis data [@Beck2020; @Freitas2025], which may be unavailable or unsuitable for long-term station reconstructions.

MissClimatePy introduces a minimalist, fully reproducible imputation framework that depends exclusively on universally available station metadata—latitude, longitude, elevation—and calendar features (year, month, day-of-year, optional harmonic transforms). Despite its constrained feature space, MissClimatePy trains a robust spatial–temporal Random Forest capable of reconstructing daily climate variables even when stations contain only a single year—or no years—of valid observations. The package incorporates controlled missingness experiments, neighbor-based benchmarking, gap diagnostics, and spatial error visualization, offering a transparent and scientifically defensible approach for climate data reconstruction.



# Statement of Need

Reliable daily climate observations form the backbone of drought monitoring, agricultural risk management, hydrological modelling, and climate-change assessments. Yet missing data continue to be one of the most persistent challenges across national meteorological networks, particularly in regions with limited resources or complex topography [@Huang2024]. Even well-maintained archives exhibit long gaps due to station relocation, technological transitions, or the integration of heterogeneous historical sources [@Dunn2023].

Many state-of-the-art imputation frameworks require external data sources such as satellite precipitation, remotely sensed vegetation indices, gridded temperature fields, or atmospheric reanalyses. These covariates are often unavailable historically, inconsistent across decades, or insufficiently representative of local station behavior. Consequently, there is an urgent need for a covariate-independent, transparent, and reproducible station-level imputation tool that:

- performs reliably under extreme missingness,
- requires only universally available metadata,
- produces deterministic and auditable results,
- scales to national archives comprising tens of millions of rows,
- and includes comprehensive diagnostic tools suitable for peer-reviewed research.

MissClimatePy fulfills this requirement by implementing a minimalist, yet powerful spatial–temporal machine-learning framework specifically designed for daily climate station reconstruction.


# Functionality and Implementation

## Minimal-Feature Random Forest Imputation

The MissClimateImputer class trains a Random Forest model using only:

- latitude, longitude, elevation,
- year, month, day-of-year,
- optional sin/cos day-of-year harmonics.

The model is fit exclusively on rows where the target variable is observed and then applied to fill all missing values. Importantly, because predictors derive solely from spatial coordinates and calendar structure, the imputer remains fully functional even when the target station contains no valid historical observations.


## Spatial Neighbor–Based Evaluation

The `evaluate_stations()` routine provides a rigorous and reproducible validation protocol. It employs:

- Haversine nearest neighbors (BallTree),
- deterministic neighbor ordering,
- optional radius and elevation constraints,
- user-defined masking fractions (0–95%) for Minimum Data Requirement (MDR) experiments,
- rainfall-specific wet/dry stratified sampling.

This design makes it possible to assess imputation performance at stations with extremely sparse or absent data, demonstrating the generalization capacity of the spatial–temporal model.

## Missingness and Gap Diagnostics

The `masking` module implements tools for:

- longest-gap and mean-gap metrics,  
- percentage-missing profiles over user-defined windows,  
- deterministic random masking for controlled experiments,  
- synthetic missingness generation suitable for MDR analysis.

These diagnostics follow best practices in observational climate data quality assessment and support reproducible MDR studies.

## Spatial Utilities

The `neighbors` module constructs deterministic neighbor maps using Haversine distance and optional altitude constraints. These maps support localized modelling, reproducibility, and the exploration of spatial error structures.

## Visualization Suite

The `viz` module provides intuitive visual diagnostics, including:

- missingness heatmaps,  
- parity scatterplots,  
- spatial RMSE maps,  
- observed–vs–imputed time-series overlays,  
- gap-length histograms,  
- imputation coverage plots.

All visualizations rely solely on `numpy`, `pandas`, and `matplotlib`, ensuring minimal dependencies and consistent rendering across environments.

## Lightweight, Reproducible Architecture

MissClimatePy is intentionally designed with minimal external dependencies ( `numpy`, `pandas`, and `scikit-learn`n) to ensure stability, simplicity, and long-term maintainability. The modular architecture enhances clarity, reproducibility, and extensibility for research and operational use.

## Example

```python
from missclimatepy import MissClimateImputer
from missclimatepy.requirements import estimate_mdr

imp = MissClimateImputer(target="tmin", k_neighbors=8, min_obs_per_station=30)
imp.fit(df)
df_out = imp.fit_transform(df)

mdr = estimate_mdr(
    df=df,
    target="tmin",
    metric_thresholds={"RMSE": 1.5, "R2": 0.5},
    missing_fracs=[0.1, 0.3, 0.6],
    grid_K=[3, 5, 8, 12]
)
```

# Related Work

Recent developments in climate data reconstruction have expanded the use of machine learning, hybrid geostatistical–AI frameworks, and deep-learning architectures [@Reichstein2019; @YANG2024120797]. Hybrid approaches combining kriging with ML or remote-sensing covariates have achieved high accuracy for temperature and precipitation interpolation [@Huang2024], while satellite–ML combinations have improved precipitation estimation in regions with limited ground observations [@Freitas2025]. However, these methods rely on external datasets that are often unavailable for historical reconstructions.

Traditional interpolation techniques remain indispensable but degrade rapidly under sparse station density or high missingness [@LI2011228]. Multivariate imputation algorithms like MICE or MissForest assume the presence of multiple covariates per timestamp [@Stekhoven2012], which daily station archives rarely provide.

MissClimatePy diverges fundamentally from these approaches by restricting its predictor space to spatial coordinates and calendar features alone, enabling imputation in data-sparse contexts and ensuring reproducibility across diverse historical periods.

# Acknowledgements

This work was supported by the Secretaría de Ciencia, Humanidades, Tecnología e Innovación (SECIHTI) through a doctoral scholarship granted to the first author under Mexico’s National Postgraduate System. We acknowledge the Comisión Nacional del Agua (CONAGUA) for maintaining the national meteorological database forming the foundation of this package, and the Colegio de Postgraduados and Universidad Mexiquense del Bicentenario for institutional support. We also thank the International Maize and Wheat Improvement Center (CIMMYT) for fostering collaboration in open climate and agricultural research.


# References