# src/missclimatepy/api.py
"""
missclimatepy.api
=================

High-level imputation API.

This module exposes a simple, reproducible imputer class that fills missing
climate values using only spatial coordinates (latitude, longitude, altitude)
and calendar features (year, month, day-of-year). It is intentionally minimal,
stateless across calls, and easy to test.

Design goals
------------
- No hard-coded column names: users provide column names.
- No external covariates required: uses (latitude, longitude, altitude, time).
- Fast, memory-lean workflow: single global RF per `fit()`; pure pandas/numpy ops.
- Reproducible: deterministic seeds; parameters explicitly exposed.

Public
------
- MissClimateImputer
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluate import RFParams


def _ensure_datetime_naive(series: pd.Series) -> pd.Series:
    """
    Parse to datetime (coerce errors) and drop timezone information if present,
    using non-deprecated dtype checks.
    """
    s = pd.to_datetime(series, errors="coerce")
    # Avoid deprecated is_datetime64tz_dtype; rely on dtype class instead.
    try:
        from pandas.api.types import DatetimeTZDtype  # pandas >= 1.0
        if isinstance(s.dtype, DatetimeTZDtype):
            s = s.dt.tz_localize(None)
    except Exception:
        # Fallback: best-effort—if .dt exists and has tz, localize away.
        with pd.option_context("mode.chained_assignment", None):
            try:
                _ = s.dt  # attribute may raise if not datetime64
                # If tz-aware, tz_localize(None) succeeds; otherwise it raises.
                s = s.dt.tz_localize(None)
            except Exception:
                pass
    return s


def _add_time_features(df: pd.DataFrame, date_col: str, add_cyclic: bool) -> pd.DataFrame:
    out = df.copy()
    out["year"] = out[date_col].dt.year.astype("int16")
    out["month"] = out[date_col].dt.month.astype("int8")
    out["doy"] = out[date_col].dt.dayofyear.astype("int16")
    if add_cyclic:
        # Use float32 to keep memory low; period ~ 365.25
        twopi = np.float32(2.0 * np.pi)
        out["doy_sin"] = np.sin(twopi * out["doy"].astype("float32") / np.float32(365.25))
        out["doy_cos"] = np.cos(twopi * out["doy"].astype("float32") / np.float32(365.25))
    return out


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)[:10]}...")


def _safe_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE / RMSE / R² without relying on sklearn.metric options that vary
    across versions. Returns NaN for degenerate cases.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0 or y_pred.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    diff = y_true - y_pred
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    var = float(np.var(y_true))
    if y_true.size < 2 or var == 0.0:
        r2 = np.nan
    else:
        ss_res = float(np.sum(diff * diff))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0.0 else np.inf)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


class MissClimateImputer:
    """
    Local imputer for climate series using only coordinates and calendar features.

    The imputer fits a single RandomForestRegressor on all rows where the target
    is observed, using (latitude, longitude, altitude, year, month, doy) and,
    optionally, cyclic features of the day of year. It then predicts and fills
    missing target values.

    Parameters
    ----------
    target : str
        Name of the target column to impute (e.g., "tmin", "tmax", "prec", "evap").
    engine : {"rf"}, optional
        Learning engine. Only "rf" is currently implemented. Alias: `model`.
    id_col, date_col, lat_col, lon_col, alt_col : str, optional
        Column names. Defaults: "station", "date", "latitude", "longitude", "altitude".
    add_cyclic : bool, optional
        Whether to include sin/cos transforms of day-of-year. Default False.
    k_neighbors : int or None, optional
        Reserved for future neighbor-constrained fitting. Currently unused by the
        high-level imputer (neighbor logic lives in `evaluate_stations`).
    min_obs_per_station : int, optional
        Minimum number of observed rows required in total to fit a model. Default 0.
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
    bootstrap, n_jobs, random_state : optional
        Common RandomForestRegressor hyperparameters. These populate RFParams.
    rf_params : dict or RFParams, optional
        Additional or overriding RF parameters.

    Notes
    -----
    - For station-wise neighbor training with controlled inclusion, use
      `missclimatepy.evaluate.evaluate_stations`.
    - This class is geared toward fast end-to-end imputation on a tidy DataFrame.

    Examples
    --------
    >>> from missclimatepy import MissClimateImputer
    >>> imp = MissClimateImputer(target="tmin", n_estimators=200, n_jobs=-1)
    >>> df_filled = imp.fit_transform(df)
    >>> metrics = imp.report(df_filled)
    """

    def __init__(
        self,
        target: str,
        *,
        engine: str = "rf",
        model: Optional[str] = None,  # accepted alias
        id_col: str = "station",
        date_col: str = "date",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
        alt_col: str = "altitude",
        add_cyclic: bool = False,
        k_neighbors: Optional[int] = None,  # reserved, not used here
        min_obs_per_station: int = 0,
        # RF exposed knobs (can be overridden by rf_params)
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[str, int, float]] = "sqrt",
        bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        rf_params: Optional[Union[Dict, RFParams]] = None,
    ) -> None:
        # Accept alias model= for historical usage
        chosen_engine = engine or model or "rf"
        if chosen_engine != "rf":
            raise ValueError(f"Only engine='rf' is supported at the moment, got {chosen_engine!r}.")

        self.target = target
        self.engine = chosen_engine

        # Schema
        self.id_col = id_col
        self.date_col = date_col
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.alt_col = alt_col
        self.add_cyclic = add_cyclic

        # Policy
        self.k_neighbors = k_neighbors
        self.min_obs_per_station = int(min_obs_per_station)

        # RF params (merge exposed knobs with rf_params overrides)
        base = RFParams(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        if rf_params is None:
            self.rf_params = base
        elif isinstance(rf_params, RFParams):
            # Override base with provided dataclass values if not None
            merged = base.__dict__.copy()
            for k, v in rf_params.__dict__.items():
                if v is not None or k in {"bootstrap", "n_jobs"}:
                    merged[k] = v
            self.rf_params = RFParams(**merged)
        else:
            # dict-like overrides
            merged = base.__dict__.copy()
            for k, v in dict(rf_params).items():  # type: ignore[arg-type]
                merged[k] = v
            self.rf_params = RFParams(**merged)

        # Runtime artifacts
        self._model: Optional[RandomForestRegressor] = None
        self._feature_columns: List[str] = []
        self._fitted_: bool = False
        self._observed_rows_: int = 0

    # ------------------------------------------------------------------ #
    # Core workflow
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "MissClimateImputer":
        """
        Fit a global RandomForest model on all rows where `target` is observed.

        Parameters
        ----------
        df : DataFrame
            Input long-format table with (id, date, latitude, longitude, altitude, target).

        Returns
        -------
        self : MissClimateImputer
        """
        required = [self.id_col, self.date_col, self.lat_col, self.lon_col, self.alt_col, self.target]
        _require_columns(df, required)

        work = df.copy()
        work[self.date_col] = _ensure_datetime_naive(work[self.date_col])
        work = work.dropna(subset=[self.date_col])

        # Time features
        work = _add_time_features(work, self.date_col, add_cyclic=self.add_cyclic)

        # Feature list
        feature_columns = [self.lat_col, self.lon_col, self.alt_col, "year", "month", "doy"]
        if self.add_cyclic:
            feature_columns += ["doy_sin", "doy_cos"]
        self._feature_columns = feature_columns

        # Training subset: observed target
        valid_mask = ~work[feature_columns + [self.target]].isna().any(axis=1)
        fit_df = work.loc[valid_mask, feature_columns + [self.target]]

        # Global minimum observations guard (sum across all stations)
        if len(fit_df) < max(1, self.min_obs_per_station):
            raise ValueError(
                f"Not enough observed rows to fit: {len(fit_df)} found, "
                f"min_obs_per_station={self.min_obs_per_station}."
            )

        X = fit_df[feature_columns].to_numpy(copy=False)
        y = fit_df[self.target].to_numpy(copy=False)

        # Build model
        params = self.rf_params.__dict__.copy()
        # scikit-learn >= 1.4 expects max_features in {"sqrt","log2"} or numeric; normalize "auto" to "sqrt"
        if params.get("max_features", None) == "auto":
            params["max_features"] = "sqrt"

        model = RandomForestRegressor(**params)
        model.fit(X, y)

        self._model = model
        self._observed_rows_ = int(len(fit_df))
        self._fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill missing `target` values using the fitted model.

        Parameters
        ----------
        df : DataFrame
            Input data.

        Returns
        -------
        DataFrame
            Copy of input with `target` imputed wherever it was missing.
        """
        if not self._fitted_ or self._model is None:
            raise RuntimeError("Imputer is not fitted. Call .fit(df) first.")

        required = [self.id_col, self.date_col, self.lat_col, self.lon_col, self.alt_col, self.target]
        _require_columns(df, required)

        out = df.copy()
        out[self.date_col] = _ensure_datetime_naive(out[self.date_col])
        out = out.dropna(subset=[self.date_col])

        out = _add_time_features(out, self.date_col, add_cyclic=self.add_cyclic)

        feature_columns = self._feature_columns
        need_pred_mask = out[self.target].isna() & (~out[feature_columns].isna().any(axis=1))
        if need_pred_mask.any():
            X_missing = out.loc[need_pred_mask, feature_columns].to_numpy(copy=False)
            preds = self._model.predict(X_missing)
            out.loc[need_pred_mask, self.target] = preds

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience wrapper for `.fit(df).transform(df)`.
        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def report(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute simple diagnostics (MAE/RMSE/R²) using the *observed* rows only.

        This applies the fitted model to rows where `target` is observed and
        compares predictions to observations. It does not mutate `df`.

        Parameters
        ----------
        df : DataFrame
            Input table (at least observed rows for the target).

        Returns
        -------
        dict
            {"MAE": float, "RMSE": float, "R2": float}
        """
        if not self._fitted_ or self._model is None:
            raise RuntimeError("Imputer is not fitted. Call .fit(df) first.")

        required = [self.id_col, self.date_col, self.lat_col, self.lon_col, self.alt_col, self.target]
        _require_columns(df, required)

        tmp = df.copy()
        tmp[self.date_col] = _ensure_datetime_naive(tmp[self.date_col])
        tmp = tmp.dropna(subset=[self.date_col])
        tmp = _add_time_features(tmp, self.date_col, add_cyclic=self.add_cyclic)

        feature_columns = self._feature_columns
        valid_mask = ~tmp[feature_columns + [self.target]].isna().any(axis=1)
        if not valid_mask.any():
            return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

        X = tmp.loc[valid_mask, feature_columns].to_numpy(copy=False)
        y_true = tmp.loc[valid_mask, self.target].to_numpy(copy=False)
        y_pred = self._model.predict(X)

        return _safe_metrics_numpy(y_true, y_pred)
