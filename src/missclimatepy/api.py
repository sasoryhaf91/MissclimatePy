# src/missclimatepy/api.py
"""
missclimatepy.api
=================

High-level imputation API.

This module exposes a simple, reproducible imputer class that fills missing
climate values using only spatial coordinates (latitude, longitude, altitude)
and calendar features (year, month, day-of-year, optionally cyclic sin/cos).

It is intentionally minimal, stateless across calls, and easy to test. The
low-level helpers for datetime handling and time-feature engineering are
shared with :mod:`missclimatepy.evaluate` to keep behavior consistent across
evaluation and imputation workflows.

Design goals
------------
- No hard-coded column names: users provide column names.
- No external covariates required: uses (latitude, longitude, altitude, time).
- Fast, memory-lean workflow: single global RF per ``fit()``; pure pandas/numpy ops.
- Reproducible: deterministic seeds; parameters explicitly exposed.

Public
------
- MissClimateImputer
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .evaluate import (
    RFParams,
    _require_columns,
    _ensure_datetime_naive,
    _add_time_features,
)


def _safe_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE / RMSE / R² using only NumPy.

    This avoids relying on :mod:`sklearn.metrics`, whose default options and
    signatures have changed slightly across versions. The function is robust
    to degenerate cases:

    - If either array is empty, all metrics are returned as NaN.
    - If the variance of ``y_true`` is zero or the length is < 2, R² is NaN.
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

    The imputer fits a single :class:`~sklearn.ensemble.RandomForestRegressor`
    on all rows where the target is observed, using:

    - spatial coordinates: latitude, longitude, altitude
    - calendar features: year, month, day-of-year
    - optionally, cyclic features of day-of-year (sin/cos)

    It then predicts and fills missing target values in a copy of the input
    :class:`pandas.DataFrame`.

    Parameters
    ----------
    target : str
        Name of the target column to impute (e.g. ``"tmin"``, ``"tmax"``,
        ``"prec"``, ``"evap"``).
    engine : {"rf"}, optional
        Learning engine. Only ``"rf"`` is currently implemented.
    model : str or None, optional
        Deprecated alias for ``engine`` kept for historical compatibility.
    id_col, date_col, lat_col, lon_col, alt_col : str, optional
        Column names. Defaults are ``"station"``, ``"date"``, ``"latitude"``,
        ``"longitude"``, ``"altitude"``.
    add_cyclic : bool, optional
        Whether to include sin/cos transforms of day-of-year. Default ``False``.
    k_neighbors : int or None, optional
        Reserved for future neighbor-constrained fitting. Currently unused by
        this high-level imputer (neighbor logic lives in
        :func:`missclimatepy.evaluate.evaluate_stations`).
    min_obs_per_station : int, optional
        Minimum number of observed rows (summed across all stations) required
        to fit a model. Default ``0``.
    n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
    bootstrap, n_jobs, random_state : optional
        Common :class:`~sklearn.ensemble.RandomForestRegressor` hyperparameters.
        These populate :class:`missclimatepy.evaluate.RFParams`.
    rf_params : dict or RFParams, optional
        Additional or overriding RF parameters. If a dict is provided, its keys
        override the defaults; if a :class:`RFParams` instance is provided, its
        non-``None`` attributes override the defaults.

    Notes
    -----
    - For station-wise neighbor training with controlled inclusion (LOSO-style
      experiments), use :func:`missclimatepy.evaluate.evaluate_stations`.
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
            raise ValueError(
                f"Only engine='rf' is supported at the moment, got {chosen_engine!r}."
            )

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
            merged = base.__dict__.copy()
            for k, v in rf_params.__dict__.items():
                # keep explicit booleans / n_jobs even if zero or False
                if v is not None or k in {"bootstrap", "n_jobs"}:
                    merged[k] = v
            self.rf_params = RFParams(**merged)
        else:
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
        Fit a global RandomForest model on all rows where ``target`` is observed.

        Parameters
        ----------
        df : DataFrame
            Input long-format table with at least
            ``(id_col, date_col, lat_col, lon_col, alt_col, target)``.

        Returns
        -------
        self : MissClimateImputer
        """
        required = [
            self.id_col,
            self.date_col,
            self.lat_col,
            self.lon_col,
            self.alt_col,
            self.target,
        ]
        _require_columns(df, required)

        work = df.copy()
        work[self.date_col] = _ensure_datetime_naive(work[self.date_col])
        work = work.dropna(subset=[self.date_col])

        # Time features (year, month, doy, optional sin/cos)
        work = _add_time_features(work, self.date_col, add_cyclic=self.add_cyclic)

        # Feature list (keep year explicitly: important for long series)
        feature_columns = [self.lat_col, self.lon_col, self.alt_col, "year", "month", "doy"]
        if self.add_cyclic:
            feature_columns += ["doy_sin", "doy_cos"]
        self._feature_columns = feature_columns

        # Training subset: observed target and complete features
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
        # scikit-learn >= 1.4 expects max_features in {"sqrt","log2"} or numeric;
        # normalize legacy "auto" to "sqrt".
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
        Fill missing ``target`` values using the fitted model.

        Parameters
        ----------
        df : DataFrame
            Input data frame.

        Returns
        -------
        DataFrame
            Copy of the input with ``target`` imputed wherever it was missing
            and features were available.
        """
        if not self._fitted_ or self._model is None:
            raise RuntimeError("Imputer is not fitted. Call .fit(df) first.")

        required = [
            self.id_col,
            self.date_col,
            self.lat_col,
            self.lon_col,
            self.alt_col,
            self.target,
        ]
        _require_columns(df, required)

        out = df.copy()
        out[self.date_col] = _ensure_datetime_naive(out[self.date_col])
        out = out.dropna(subset=[self.date_col])

        out = _add_time_features(out, self.date_col, add_cyclic=self.add_cyclic)

        feature_columns = self._feature_columns
        need_pred_mask = out[self.target].isna() & (
            ~out[feature_columns].isna().any(axis=1)
        )
        if need_pred_mask.any():
            X_missing = out.loc[need_pred_mask, feature_columns].to_numpy(copy=False)
            preds = self._model.predict(X_missing)
            out.loc[need_pred_mask, self.target] = preds

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience wrapper for ``.fit(df).transform(df)``.
        """
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def report(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute simple diagnostics (MAE/RMSE/R²) using the *observed* rows only.

        This applies the fitted model to rows where ``target`` is observed and
        compares predictions to observations. It does not mutate ``df`` and it
        does not change the model state.

        Parameters
        ----------
        df : DataFrame
            Input table (with at least the observed rows for the target).

        Returns
        -------
        dict
            A dictionary with keys ``{"MAE", "RMSE", "R2"}``.
        """
        if not self._fitted_ or self._model is None:
            raise RuntimeError("Imputer is not fitted. Call .fit(df) first.")

        required = [
            self.id_col,
            self.date_col,
            self.lat_col,
            self.lon_col,
            self.alt_col,
            self.target,
        ]
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
