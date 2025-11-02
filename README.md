
# MissclimatePy

**Minimal and reproducible framework for climate data imputation using only spatial coordinates (x, y, z) and calendar features (t).**

[![CI](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml/badge.svg)](https://github.com/sasoryhaf91/MissclimatePy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/TBA.svg)](TBA)

---

## 🧭 Philosophy

**MissclimatePy** focuses on *local imputation* of missing climate records using only:

- **Spatial coordinates:** latitude, longitude, elevation  
- **Temporal descriptors:** year, month, day-of-year (and optionally harmonic sin/cos transforms)

This design ensures:

- Independence from external covariates or gridded data sources  
- Robust estimation even under high missingness  
- Fully reproducible workflows through deterministic seeds and transparent validation  

---

## ⚙️ Core Framework

MissclimatePy introduces a **Minimum Data Requirement (MDR)** approach for robust local imputation:

- Defines the *minimum percentage of valid data* required for model training.  
- Evaluates imputation *stability* under varying inclusion levels (e.g., 10–80%).  
- Uses *spatio-temporal neighbors* (based on t, x, y, z proximity) to reinforce local learning.  
- Quantifies uncertainty and reconstruction reliability across multiple masking scenarios.

Each model is trained **locally per station** using Random Forest regressors constrained to `(t, x, y, z)` inputs.

---

## 📦 Installation

```bash
pip install -U pip
pip install .
# or for development
pip install -e ".[dev]"
```

---

## 🚀 Quickstart

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

# Local RF-based imputation
imp = MissClimateImputer(
    engine="rf",
    target="tmin",
    k_neighbors=5,        # spatio-temporal neighbors
    min_obs_per_station=30,
    n_estimators=100,
    n_jobs=-1
)

out = imp.fit_transform(df)
print(imp.report(out))
```

---

## 🧩 API Overview

| Method | Description |
|:--|:--|
| `fit(df)` | Validates structure and prepares model. |
| `transform(df)` | Imputes missing values in all stations. |
| `fit_transform(df)` | Fits and imputes in one step. |
| `report(df)` | Returns MAE, RMSE, R² and coverage metrics. |
| `estimate_mdr(df, target, ...)` | Evaluates Minimum Data Requirement (MDR) through masking. |

---

## 🧠 Methods Included

- `rf`: Random Forest baseline (AI-based local model)
- (future) `mlp`: Multi-Layer Perceptron baseline
- (future) `rf+xgb`: Hybrid Random Forest–XGBoost

---

## 🔁 Reproducibility

- Unit tests via `pytest`  
- Deterministic seeds for consistent runs  
- Includes `/examples/example_minimal.py`  
- Continuous Integration (CI) via GitHub Actions  

---

## 🔬 Citation

See [`CITATION.cff`](CITATION.cff) for citation format.

> Fernández H.A. (2025). *MissclimatePy: Missing-Data Imputation for Climate Time Series (x, y, z + calendar only).*  
> Version 0.1.0. DOI: [10.5281/zenodo.TBA](https://doi.org/TBA)

---

## 📜 License
MIT License © 2025 Hugo Antonio Fernández  
See [`LICENSE`](LICENSE) for details.
