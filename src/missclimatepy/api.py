# src/missclimatepy/api.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from .models import build_model

def _calendar_features(dates: pd.Series) -> pd.DataFrame:
    d = pd.to_datetime(dates)
    doy = d.dt.dayofyear.astype(int)
    year = d.dt.year.astype(int)
    month = d.dt.month.astype(int)
    sin1 = np.sin(2 * np.pi * doy / 365.25)
    cos1 = np.cos(2 * np.pi * doy / 365.25)
    return pd.DataFrame({"year": year, "month": month, "doy": doy, "sin1": sin1, "cos1": cos1})

@dataclass
class MissClimateImputer:
    """
    Local spatio-temporal imputer using only (x, y, z + calendar) features.
    Model-agnostic via registry (default: 'rf').

    Back-compat:
    - Accepts `model="rf"` as alias of `engine`.
    - Accepts top-level RF params: `n_estimators`, `n_jobs`, `random_state`.
    - Accepts `rf_params={...}` and merges with `model_params`.
    """
    # Current API
    engine: str = "rf"
    target: str = "tmin"
    k_neighbors: int = 12
    min_obs_per_station: int = 0
    model_params: Optional[Dict[str, Any]] = None

    # ---- Back-compat/legacy parameters ----
    model: Optional[str] = None  # alias for engine
    n_estimators: Optional[int] = None
    n_jobs: Optional[int] = None
    random_state: Optional[int] = None
    rf_params: Optional[Dict[str, Any]] = None
    # ---------------------------------------

    _model: Any = field(default=None, init=False)
    _fitted: bool = field(default=False, init=False)
    _resolved_params: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # Map legacy 'model' → 'engine'
        if self.model is not None:
            self.engine = self.model

        # Merge parameters in priority order (top-level wins):
        # model_params (base) <- rf_params <- top-level (n_estimators, n_jobs, random_state)
        params: Dict[str, Any] = {}
        if self.model_params:
            params.update(self.model_params)
        if self.rf_params:
            params.update(self.rf_params)
        if self.n_estimators is not None:
            params["n_estimators"] = self.n_estimators
        if self.n_jobs is not None:
            params["n_jobs"] = self.n_jobs
        if self.random_state is not None:
            params["random_state"] = self.random_state
        self._resolved_params = params

    @staticmethod
    def make_features(df: pd.DataFrame) -> pd.DataFrame:
        cal = _calendar_features(df["date"])
        return pd.concat(
            [
                df[["latitude", "longitude", "elevation"]].reset_index(drop=True),
                cal.reset_index(drop=True),
            ],
            axis=1,
        )

    def fit(self, df: pd.DataFrame) -> "MissClimateImputer":
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not found.")
        obs = df[df[self.target].notna()].copy()
        if len(obs) == 0:
            raise ValueError("No observed rows to fit. Provide neighbors with observations.")
        X = self.make_features(obs)
        y = obs[self.target].to_numpy()
        self._model = build_model(self.engine, **self._resolved_params)
        self._model.fit(X, y)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        out = df.copy()
        mask = out[self.target].isna()
        if mask.any():
            Xmiss = self.make_features(out.loc[mask])
            out.loc[mask, self.target] = self._model.predict(Xmiss)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self, df_after: pd.DataFrame) -> dict:
        miss = int(df_after[self.target].isna().sum())
        rows = int(len(df_after))
        return {
            "target": self.target,
            "rows": rows,
            "stations": int(df_after["station"].nunique()) if "station" in df_after.columns else None,
            "missing_after": miss,
            "missing_rate_after": float(miss / rows if rows else 0.0),
        }
