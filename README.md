
# MissclimatePy

**Missing-Data Imputation for Climate Time Series (x, y, z + calendar only).**  
Minimal, reproducible imputation with spatial coordinates and calendar harmonics only.

[![CI](https://github.com/youruser/missclimatepy/actions/workflows/ci.yml/badge.svg)](https://github.com/youruser/missclimatepy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/TBA.svg)](TBA)

## Philosophy
- Minimal inputs: latitude, longitude, elevation, and calendar features.
- Reproducible science: deterministic seeds, tests, open data schemas.
- Scalable to national networks.

## Install
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

df = pd.DataFrame({
    "station": ["S001"]*3 + ["S002"]*3,
    "date": pd.to_datetime(["1991-01-01","1991-01-02","1991-01-03"]*2),
    "latitude": [19.5]*6, "longitude": [-99.1]*6, "elevation": [2300]*6,
    "tmin": [8.0, None, 7.5, 9.0, None, 8.2]
})
imp = MissClimateImputer(model="rf", target="tmin", n_estimators=50, n_jobs=-1)
out = imp.fit_transform(df)
print(imp.report(out))
```

## API
- `MissClimateImputer.fit(df)`
- `MissClimateImputer.transform(df)`
- `MissClimateImputer.fit_transform(df)`
- `MissClimateImputer.report(df_valid)`

## Methods included
- `rf`: Random Forest baseline (AI).
- `idw`: Inverse Distance Weighting (traditional).

## Reproducibility
- Tests (`pytest`), seed control, example script.

## Citation
See `CITATION.cff`.

## License
MIT. See `LICENSE`.
