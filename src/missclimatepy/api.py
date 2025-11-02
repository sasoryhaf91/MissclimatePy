# src/missclimatepy/api.py
from dataclasses import dataclass
from typing import Optional, Dict, Any
import pandas as pd

from .impute import fit_transform_local_rf


@dataclass
class MissClimateImputer:
    """
    Public API for MissclimatePy.

    Parameters
    ----------
    engine : {"rf"}
        Imputation engine. Currently only "rf" (local Random Forest).
    target : str
        Target variable column to impute (e.g., "tmin", "tmax", "prec", "evap").
    k_neighbors : int
        Number of spatio-temporal neighbors used to train the local model.
        (If your neighbor selection is purely spatial, it still applies here.)
    min_obs_per_station : int
        Minimum number of valid observations required to train the local model.
    n_estimators : int
        Number of trees for RandomForestRegressor.
    n_jobs : int
        Parallel jobs for RandomForestRegressor.
    random_state : int
        Seed for reproducibility.
    rf_params : dict | None
        Optional hyperparameters forwarded directly to RandomForestRegressor,
        e.g. {"max_depth": 20, "min_samples_leaf": 2}.
    """
    engine: str = "rf"
    target: str = "tmin"
    k_neighbors: int = 8
    min_obs_per_station: int = 60
    n_estimators: int = 300
    n_jobs: int = -1
    random_state: int = 42
    rf_params: Optional[Dict[str, Any]] = None

    _fitted: bool = False

    def fit(self, df: pd.DataFrame):
        if self.engine not in {"rf"}:
            raise ValueError(f"Unsupported engine '{self.engine}'. Supported: 'rf'.")

        # basic column validation
        required = ["station", "date", "latitude", "longitude", "elevation", self.target]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call .fit(df) before .transform(df).")

        if self.engine == "rf":
            return fit_transform_local_rf(
                df=df,
                target=self.target,
                k_neighbors=self.k_neighbors,
                min_obs_per_station=self.min_obs_per_station,
                n_estimators=self.n_estimators,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                rf_params=self.rf_params,  # <- new
            )

        raise RuntimeError("Unexpected engine state.")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self, df_valid: pd.DataFrame) -> dict:
        """
        Minimal report after imputation.
        """
        rows = len(df_valid)
        stations = df_valid["station"].nunique()
        miss_after = df_valid[self.target].isna().sum()
        miss_rate = float(miss_after) / float(rows) if rows else 0.0
        return {
            "target": self.target,
            "rows": rows,
            "stations": stations,
            "missing_after": int(miss_after),
            "missing_rate_after": miss_rate,
        }
