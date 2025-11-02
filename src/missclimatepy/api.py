# src/missclimatepy/api.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from .models import build_model


@dataclass
class _FitState:
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    model: Any = None


class MissClimateImputer:
    """
    High-level imputer that trains a global model with spatio-temporal features
    (t, x, y, z) and then predicts missing target values.

    Parameters
    ----------
    engine : str, default="rf"
        Model name (e.g., "rf").
    target : str, default="tmin"
        Target column to impute.
    k_neighbors : int, default=20
        Number of neighbors for neighbor-based features (pipeline-level; not a model arg).
    min_obs_per_station : int, default=50
        Minimum number of valid target rows per station to include in training.
    n_jobs : int, default=-1
        Parallelism for model training/prediction (passed to the estimator when applicable).
    random_state : int, default=42
        Random seed for reproducibility.
    rf_params : dict, optional
        Random Forest overrides.
    model_params : dict, optional
        Generic model overrides (merged on top of rf_params).
    """

    def __init__(
        self,
        engine: str = "rf",
        target: str = "tmin",
        k_neighbors: int = 20,
        min_obs_per_station: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        rf_params: Optional[Dict[str, Any]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.engine = engine
        self.target = target

        # Pipeline-level knobs (NO se pasan al modelo)
        self.k_neighbors = int(k_neighbors)
        self.min_obs_per_station = int(min_obs_per_station)

        # Modelo / ejecución
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Hyperparams (se fusionan y luego se filtran)
        self._rf_params = rf_params or {}
        self._model_params = model_params or {}

        # Estado de entrenamiento
        self._state = _FitState()

    # ----------------------------------------------------------------------
    # Feature builder simple (placeholder; aquí irán t, x, y, z y lo que
    # uses con vecinos cuando lo cableemos).
    def make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "lat": pd.to_numeric(df["latitude"], errors="coerce"),
                "lon": pd.to_numeric(df["longitude"], errors="coerce"),
                "elev": pd.to_numeric(df["elevation"], errors="coerce"),
                "doy": pd.to_datetime(df["date"], errors="coerce").dt.dayofyear.astype("Int64"),
                "year": pd.to_datetime(df["date"], errors="coerce").dt.year.astype("Int64"),
            },
            index=df.index,
        )
        # Codificación sen/cos opcional para estacionalidad
        with np.errstate(invalid="ignore"):
            out["sin_doy"] = np.sin(2 * np.pi * out["doy"] / 366.0)
            out["cos_doy"] = np.cos(2 * np.pi * out["doy"] / 366.0)
        return out

    # ----------------------------------------------------------------------
    def _resolve_model_params(self) -> Dict[str, Any]:
        """
        Merge rf_params and model_params, drop pipeline-only keys and None values.
        """
        params: Dict[str, Any] = {}
        params.update(self._rf_params)
        params.update(self._model_params)

        # No enviar knobs del pipeline al estimador
        EXCLUDE = {"k_neighbors", "min_obs_per_station"}

        cleaned = {k: v for k, v in params.items() if v is not None and k not in EXCLUDE}

        # Pasar n_jobs/random_state cuando el modelo los soporte (RF los soporta)
        cleaned.setdefault("n_jobs", self.n_jobs)
        cleaned.setdefault("random_state", self.random_state)
        return cleaned

    # ----------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "MissClimateImputer":
        # Filtrar filas con target observado
        obs = df[df[self.target].notna()]
        # Exigir mínimo de observaciones por estación
        if self.min_obs_per_station > 0:
            counts = obs.groupby("station")[self.target].count()
            keep = counts[counts >= self.min_obs_per_station].index
            obs = obs[obs["station"].isin(keep)]

        X = self.make_features(obs).to_numpy()
        y = pd.to_numeric(obs[self.target], errors="coerce").to_numpy()

        params = self._resolve_model_params()
        self._state.model = build_model(self.engine, **params)
        self._state.model.fit(X, y)

        self._state.X_train, self._state.y_train = X, y
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._state.model is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        out = df.copy()
        X_all = self.make_features(out).to_numpy()
        yhat = self._state.model.predict(X_all)
        # Imputa solo valores faltantes del target
        mask = out[self.target].isna()
        out.loc[mask, self.target] = yhat[mask]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def report(self, df_after: pd.DataFrame) -> dict:
        # Reporte mínimo post-imputación
        miss_after = int(df_after[self.target].isna().sum())
        rows = int(len(df_after))
        stations = int(df_after["station"].nunique())
        return {
            "target": self.target,
            "rows": rows,
            "stations": stations,
            "missing_after": miss_after,
            "missing_rate_after": (miss_after / rows) if rows else 0.0,
        }
