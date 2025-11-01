
# MissclimatePy

**Missing-Data Imputation for Climate Time Series (x, y, z + calendar only)**  
*A reproducible framework for local station imputation based on spatial and temporal features only.*

[![CI](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml/badge.svg)](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/TBA.svg)](TBA)

---

## 🧭 Philosophy

MissclimatePy focuses on **local imputation** of missing climate records using only:

- Spatial coordinates: **latitude, longitude, elevation**  
- Temporal descriptors: **year, month, day-of-year**, and optionally **harmonic sin/cos transforms**

This design ensures:
- Independence from external covariates (e.g., nearby variables or reanalysis data)
- Robust estimation even under high missingness
- Fully reproducible workflows through deterministic seeds and transparent validation

---

## ⚙️ Core Framework

MissclimatePy implements a **Minimum Data Requirement (MDR)** approach:
- Defines the **minimum percentage of valid data** required for model training.
- Evaluates imputation stability under varying inclusion levels (e.g., 10–80%).
- Uses **spatio-temporal neighbors** (based on t, x, y, z proximity) to reinforce local predictions.

---

## 📦 Install

```bash
pip install -U pip
pip install .
# or dev
pip install -e ".[dev]"
```

## Quickstart

```python
import pandas as pd
from missclimatepy import MissClimateImputer

# Example dataset
df = pd.DataFrame({
    "station": ["S001"]*3 + ["S002"]*3,
    "date": pd.to_datetime(["1991-01-01","1991-01-02","1991-01-03"]*2),
    "latitude": [19.5]*6, "longitude": [-99.1]*6, "elevation": [2300]*6,
    "tmin": [8.0, None, 7.5, 9.0, None, 8.2]
})

# Fit and impute
imp = MissClimateImputer(
    model="rf",
    target="tmin",
    n_estimators=100,
    min_data=0.3,       # minimum inclusion (30% valid data)
    n_neighbors=5,      # spatio-temporal neighbors
    n_jobs=-1
)

out = imp.fit_transform(df)
print(imp.report(out))

```
## API
- `MissClimateImputer.fit(df)`
- `MissClimateImputer.transform(df)`
- `MissClimateImputer.fit_transform(df)`
- `MissClimateImputer.report(df_valid)`
- `MissClimateImputer.plot_series(station)`

## Methods included
- `rf`: Random Forest baseline (AI).

## Reproducibility
- Tests (`pytest`)
- Deterministic seeds for replicability
- Includes examples/example_minimal.pyexample 

## Citation
See `CITATION.cff`.

## License
MIT. See `LICENSE`.